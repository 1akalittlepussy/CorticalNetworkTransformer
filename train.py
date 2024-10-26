# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import copy
import pandas as pd
from collections import defaultdict
import torch
from transformer.utils import sensitivity_specificity
from sklearn.metrics import f1_score,roc_auc_score
#from torch_geometric.data.dataloader import DataLoader
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch_geometric import datasets
from transformer.models_1 import DiffGraphTransformer, GraphTransformer, GNNTransformer
from transformer.data import GraphDataset
from transformer.position_encoding_lpe import LapEncoding, FullEncoding, POSENCODINGS
from transformer.utils import count_parameters
from timeit import default_timer as timer
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from baseline.GAT import GAT
from baseline.GCN import GCN
from baseline.GIN import GIN
from baseline.GraphSAGE import GraphSAGE

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from load_data import FSDataset


def load_args():
    parser = argparse.ArgumentParser(
        description='Transformer baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--nb_heads', type=int, default=4)
    parser.add_argument('--nb_layers', type=int, default=3)
    parser.add_argument('--dim_hidden', type=int, default=64)
    # parser.add_argument('--pos_enc', choices=[None, 'diffusion', 'pstep', 'adj'], default='adj')
    parser.add_argument('--pos_enc', choices=[None, 'diffusion', 'pstep', 'adj'], default=None)
    parser.add_argument('--lappe', action='store_true', help='use laplacian PE', default=False)
    parser.add_argument('--lap_dim', type=int, default=8, help='dimension for laplacian PE')
    parser.add_argument('--p', type=int, default=3, help='p step random walk kernel')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='bandwidth for the diffusion kernel')
    parser.add_argument('--normalization', choices=[None, 'sym', 'rw'], default='sym',
                        help='normalization for Laplacian')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-05)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    parser.add_argument('--warmup', type=int, default=2000)
    parser.add_argument('--layer_norm', action='store_true', help='use layer norm instead of batch norm')
    parser.add_argument('--zero_diag', action='store_true', help='zero diagonal for PE matrix')
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.batch_norm = not args.layer_norm

    args.save_logs = False
    return args


def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    tic = timer()
    for i, (data, mask, pe, lap_pe, degree, adjs, labels) in enumerate(loader):
        if args.warmup is not None:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)
        if args.lappe:
            # sign flip as in Bresson et al. for laplacian PE
            sign_flip = torch.rand(lap_pe.shape[-1])
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            lap_pe = lap_pe * sign_flip.unsqueeze(0)

        if use_cuda:
            data = data.cuda()
            #data = np.reshape(data, (-1, 2))
            #scaler = StandardScaler()
            #data = torch.from_numpy(scaler.fit_transform(data.cpu()))
            #m, n = np.shape(data)
            #m = int(m/1922)
            #print(m, n)
            #data = data.reshape([m, 62, 62])
            #if m == 246016:
            #    data = data.reshape([128, 62, 62])
            #if m == 172980:
            #    data = data.reshape([90, 62, 62])
            #if m == 174902:
            #    data = data.reshape([91, 62, 62])
            #if m == 211420:
            #    data = data.reshape([110, 62, 62])
            #data = torch.tensor(data, dtype=torch.float).cuda()
            mask = mask.cuda()
            if pe is not None:
                pe = pe.cuda()
            if lap_pe is not None:
                lap_pe = lap_pe.cuda()
            if degree is not None:
                degree = degree.cuda()
            labels = labels.cuda()
            adjs = adjs.cuda()

        optimizer.zero_grad()
        output = model(data, adjs, mask, pe, lap_pe, degree)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        pred_auc = output.detach().cpu().numpy()
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # 指定归一化范围
        pred_auc = scaler.fit_transform(pred_auc)
        pred = output.data.argmax(dim=1)
        labels_auc = labels.cpu().numpy()
        # pred_auc=pred.cpu().numpy()
        # auc_cal=np.array([labels_auc.ravel(), pred_auc.ravel()])
        #np.savetxt('trandata.csv', labels_auc)
        #np.savetxt('trandata1.csv', pred_auc)
        running_loss += loss.item() * len(data)
        running_acc += torch.sum(pred == labels).item()

    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_acc = running_acc / n_sample
    print('Train loss: {:.4f} Acc: {:.4f} time: {:.2f}s'.format(
        epoch_loss, epoch_acc, toc - tic))
    return epoch_loss,epoch_acc


def eval_epoch(model, loader, criterion, use_cuda=False):
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    sen, sp ,f1,auc= 0, 0,0,0

    tic = timer()
    with torch.no_grad():
        for data, mask, pe, lap_pe, degree, adjs, labels in loader:
            if args.lappe:
                # sign flip as in Bresson et al. for laplacian PE
                sign_flip = torch.rand(lap_pe.shape[-1])
                sign_flip[sign_flip >= 0.5] = 1.0
                sign_flip[sign_flip < 0.5] = -1.0
                lap_pe = lap_pe * sign_flip.unsqueeze(0)

            if use_cuda:
                data = data.cuda()
                #data = np.reshape(data, (-1, 2))
                #scaler = StandardScaler()
                #data = torch.from_numpy(scaler.fit_transform(data.cpu()))
                #m, n = np.shape(data)
                #m = int(m / 1922)
                #print(m,n)
                #data = data.reshape([m, 62, 62])
                #data = torch.tensor(data, dtype=torch.float).cuda()
                mask = mask.cuda()
                if pe is not None:
                    pe = pe.cuda()
                if lap_pe is not None:
                    lap_pe = lap_pe.cuda()
                if degree is not None:
                    degree = degree.cuda()
                labels = labels.cuda()
                adjs = adjs.cuda()

            output = model(data, adjs, mask, pe, lap_pe, degree)
            loss = criterion(output, labels)
            pred_auc=output.cpu().numpy()
            #scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # 指定归一化范围
            #pred_auc = scaler.fit_transform(pred_auc)
            pred = output.data.argmax(dim=1)
            labels_auc=labels.cpu().numpy()
            #pred_auc=pred.cpu().numpy()
            #auc_cal=np.array([labels_auc.ravel(), pred_auc.ravel()])
            #p.savetxt('gtdata.csv', labels_auc)
            #np.savetxt('gtdata1.csv', pred_auc)
            sensitivity, specificity = sensitivity_specificity(labels.cpu(), pred.cpu())
            test_f1 = f1_score(labels.cpu(), pred.cpu())
            test_auc = roc_auc_score(labels.cpu(), pred.cpu())

            sen += sensitivity * len(data)
            sp += specificity * len(data)

            f1 += test_f1 * len(data)
            auc += test_auc * len(data)

            running_loss += loss.item() * len(data)
            running_acc += torch.sum(pred == labels).item()
    toc = timer()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_acc = running_acc / n_sample

    print('Eval loss: {:.4f} Acc: {:.4f} time: {:.2f}s'.format(
        epoch_loss, epoch_acc, toc - tic))
    return epoch_acc, epoch_loss, sen / n_sample, sp / n_sample, f1/n_sample, auc/n_sample


def main():
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)

    n_tags = None

    train_acc_list, train_loss_list, val_acc_list,val_loss_list = [], [], [] ,[]
    best_acc_list, best_sp_list, best_sen_list,best_f1_list,best_auc_list= [], [], [],[],[]
    bestmeanacc_list, bestmeansp_list, bestmeansen_list,bestmeanf1_list,bestmeanauc_list = [], [], [],[],[]
    random_s = np.array([25, 50, 100, 125, 150, 175, 200, 225, 250, 275], dtype=int)

    for i in range(10):

      #dataset = FSDataset(r'E:\Users\HP\Desktop\results', folds=10, seed=random_s[i])
      dataset = FSDataset(r'F:\graphmix-master-new\codes_graph\HCP62ROI', folds=10, seed=random_s[i])

      for i in range(10):
        # dataset = FSDataset(r'..\data_graph\MDD\ROISignals_FunImgARglobalCWF',folds=10,seed=1)
        train_dset, val_dset, test_dset, train_split, valid_split, test_split = dataset.kfold_split(test_index=i)
        # dataset = dataset.fc
        train_dset = GraphDataset(train_dset, n_tags, degree=True)
        input_size = train_dset.input_size()

        train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=train_dset.collate_fn())

        val_dset = GraphDataset(val_dset, n_tags, degree=True)
        val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dset.collate_fn())

        pos_encoder = None
        if args.pos_enc is not None:
            pos_encoding_method = POSENCODINGS.get(args.pos_enc, None)
            pos_encoding_params_str = ""
            if args.pos_enc == 'diffusion':
                pos_encoding_params = {
                    'beta': args.beta
                }
                pos_encoding_params_str = args.beta
            elif args.pos_enc == 'pstep':
                pos_encoding_params = {
                    'beta': args.beta,
                    'p': args.p
                }
                pos_encoding_params_str = "{}_{}".format(args.p, args.beta)
            else:
                pos_encoding_params = {}

            if pos_encoding_method is not None:
                'E:\graphmix-master-new\cache'
                # pos_cache_path = 'E:/graphmix-master/cache/pe/{}_{}_{}.pkl'.format(args.pos_enc, args.normalization, pos_encoding_params_str)
                pos_cache_path = 'F:/graphmix-master-new/cache/pe12/{}_{}_{}.pkl'.format(args.pos_enc, args.normalization,
                                                                                    pos_encoding_params_str)
                pos_encoder = pos_encoding_method(pos_cache_path, normalization=args.normalization,
                                                  zero_diag=args.zero_diag, **pos_encoding_params)

            print("Position encoding...")
            pos_encoder.apply_to(dataset, split='all')
            train_dset.pe_list = [dataset.pe_list[i] for i in train_split]
            val_dset.pe_list = [dataset.pe_list[i] for i in valid_split]

        if args.lappe and args.lap_dim > 0:
            lap_pos_encoder = LapEncoding(args.lap_dim, normalization='sym')
            lap_pos_encoder.apply_to(train_dset)
            lap_pos_encoder.apply_to(val_dset)

        if args.pos_enc is not None:
            model = DiffGraphTransformer(in_size=input_size,
                                         nb_class=2,
                                         d_model=args.dim_hidden,
                                         dim_feedforward=2 * args.dim_hidden,
                                         dropout=args.dropout,
                                         nb_heads=args.nb_heads,
                                         nb_layers=args.nb_layers,
                                         batch_norm=args.batch_norm,
                                         lap_pos_enc=args.lappe,
                                         lap_pos_enc_dim=args.lap_dim)
        else:
            model = GraphTransformer(in_size=input_size,
                                   nb_class=2,
                                   d_model=args.dim_hidden,
                                   dim_feedforward=2 * args.dim_hidden,
                                   dropout=args.dropout,
                                   nb_heads=args.nb_heads,
                                   nb_layers=args.nb_layers,
                                   lap_pos_enc=args.lappe,
                                   lap_pos_enc_dim=args.lap_dim)
        if args.use_cuda:
            model.cuda()
        print("Total number of parameters: {}".format(count_parameters(model)))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.warmup is None:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        else:
            lr_steps = (args.lr - 1e-6) / args.warmup
            decay_factor = args.lr * args.warmup ** .5

            def lr_scheduler(s):
                if s < args.warmup:
                    lr = 1e-6 + s * lr_steps
                else:
                    lr = decay_factor * s ** -.5
                return lr

        test_dset = GraphDataset(test_dset, n_tags, degree=True)
        test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=test_dset.collate_fn())
        if pos_encoder is not None:
            pos_encoder.apply_to(test_dset, split='test')

        if args.lappe and args.lap_dim > 0:
            lap_pos_encoder.apply_to(test_dset)

        print("Training...")
        best_val_acc = 0
        # best_model = None
        best_epoch = 0
        start_time = timer()
        for epoch in range(args.epochs):
            print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
            train_loss,train_acc = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, args.use_cuda)
            val_acc, val_loss, val_sen, val_sp,val_f1, val_auc = eval_epoch(model, val_loader, criterion, args.use_cuda)
            # test_acc, test_loss, sensitivity, specificity = eval_epoch(model, test_loader, criterion, args.use_cuda)

            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            #val_acc_list.append(val_acc)
            #val_loss_list.append(val_loss)

            #name = ['train_acc']
            #test = pd.DataFrame(columns=name, data=train_acc_list)  # 数据有三列，列名分别为one,two,three
            #test.to_csv('E:/train_acc.csv', encoding='gbk')

            if args.warmup is None:
                lr_scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc, best_sp, best_sen,best_f1,best_auc = val_acc, val_sp, val_sen,val_f1,val_auc
                best_epoch = epoch
                #best_weights = copy.deepcopy(model.state_dict())

        total_time = timer() - start_time
        print("best epoch: {} best val acc: {:.4f}".format(best_epoch, best_val_acc))

        #torch.save(model.load_state_dict(best_weights),'best_weight.pth')

        # print("Testing...")
        # test_acc, test_loss, sensitivity, specificity = eval_epoch(model, val_loader, criterion, args.use_cuda)

        # print("test Acc {:.4f}".format(test_acc))
        best_acc_list.append(best_val_acc)
        best_sp_list.append(best_sp)
        best_sen_list.append(best_sen)
        best_f1_list.append(best_f1)
        best_auc_list.append(best_auc)

        #name = ['layer3_train_acc']
        #test = pd.DataFrame(columns=name, data=best_acc_list)  # 数据有三列，列名分别为one,two,three
        #test.to_csv('E:\layer3_acc.csv', encoding='gbk')

        #name = ['layer3_train_sp']
        #test = pd.DataFrame(columns=name, data=best_sp_list)  # 数据有三列，列名分别为one,two,three
        #test.to_csv('E:\layer3_sp.csv', encoding='gbk')

        #name = ['layer3_train_sen']
        #test = pd.DataFrame(columns=name, data=best_sen_list)  # 数据有三列，列名分别为one,two,three
        #test.to_csv('E:\layer3_sen.csv', encoding='gbk')

        print("10 fold test average: {:.2f}±{:.2f} sensitivity:{:.2f}±{:.2f} specificity:{:.2f}±{:.2f} f1:{:.2f}±{:.2f} auc:{:.2f}±{:.2f}".format(
            np.mean(best_acc_list) * 100, np.std(best_acc_list) * 100,
            np.mean(best_sen_list) * 100, np.std(best_sen_list) * 100,
            np.mean(best_sp_list) * 100, np.std(best_sp_list) * 100,
            np.mean(best_f1_list) * 100, np.std(best_f1_list) * 100,
            np.mean(best_auc_list) * 100, np.std(best_auc_list) * 100,))

        best_acc_mean = np.mean(best_acc_list) * 100
        best_sen_mean = np.mean(best_sen_list) * 100
        best_sp_mean = np.mean(best_sp_list) * 100
        best_f1_mean = np.mean(best_f1_list) * 100
        best_auc_mean = np.mean(best_auc_list) * 100

        bestmeanacc_list.append(best_acc_mean)
        bestmeansen_list.append(best_sen_mean)
        bestmeansp_list.append(best_sp_mean)
        bestmeanf1_list.append(best_f1_mean)
        bestmeanauc_list.append(best_auc_mean)

    print("10 number 10 fold test average: {:.2f}±{:.2f} 10 number 10 fold specificity:{:.2f}±{:.2f} 10 number 10 fold sensitivity:{:.2f}±{:.2f} 10 number 10 fold f1:{:.2f}±{:.2f} 10 number 10 fold auc:{:.2f}±{:.2f} ".format(
              np.mean(bestmeanacc_list), np.std(bestmeanacc_list),
              np.mean(bestmeansen_list), np.std(bestmeansen_list),
              np.mean(bestmeansp_list), np.std(bestmeansp_list),
              np.mean(bestmeanf1_list), np.std(bestmeanf1_list),
              np.mean(bestmeanauc_list), np.std(bestmeanauc_list)))


if __name__ == "__main__":
    main()