import argparse
import numpy as np
import os
import copy
import pandas as pd
from collections import defaultdict
import torch
from sklearn import svm
#from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score,roc_auc_score
from sklearn import preprocessing
from transformer.utils import sensitivity_specificity

import torch.nn.functional as F
from torch_geometric.data.dataloader import DataLoader
# import DataLoader
from transformer.utils import count_parameters
from timeit import default_timer as timer
from torch import nn, optim

from baseline.GAT import GAT
from baseline.GCN import GCN
from baseline.GIN import GIN
from baseline.GraphSAGE import GraphSAGE
from baseline.FGDN import FGDN
from baseline.GraphTrans import GraphTrans
#from baseline.newGIN import GIN
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
    parser.add_argument('--fc_features', type=int, default=90, help='fc_features')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='the numbers of convolution layers')  # 64
    parser.add_argument('--num_nodes', type=int, default=90, help='num_nodes')
    parser.add_argument('--nb_heads', type=int, default=4)
    parser.add_argument('--nb_layers', type=int, default=3)
    parser.add_argument('--dim_hidden', type=int, default=64)
    parser.add_argument('--pos_enc', choices=[None, 'diffusion', 'pstep', 'adj'], default='adj')
    parser.add_argument('--lappe', action='store_true', help='use laplacian PE', default=True)
    parser.add_argument('--lap_dim', type=int, default=8, help='dimension for laplacian PE')
    parser.add_argument('--p', type=int, default=1, help='p step random walk kernel')
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
    # parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
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
    for i, data in enumerate(loader):
        if args.warmup is not None:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)

        if use_cuda:
            data = data.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        pred = output.data.argmax(dim=1)
        running_loss += loss.item() * data.num_graphs
        running_acc += torch.sum(pred == data.y).item()

    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_acc = running_acc / n_sample
    print('Train loss: {:.4f} Acc: {:.4f} time: {:.2f}s'.format(
        epoch_loss, epoch_acc, toc - tic))
    return epoch_loss


def eval_epoch(model, loader, criterion, use_cuda=False):
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    sen, sp ,f1,auc= 0, 0,0,0

    tic = timer()
    with torch.no_grad():
        for data in loader:
            if use_cuda:
                data = data.cuda()

            output = model(data)
            loss = criterion(output, data.y)
            pred_auc = output.detach().cpu().numpy()
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # 指定归一化范围
            pred_auc = scaler.fit_transform(pred_auc)
            pred = output.data.argmax(dim=1)
            labels_auc = data.y.cpu().numpy()
            # pred_auc=pred.cpu().numpy()
            # auc_cal=np.array([labels_auc.ravel(), pred_auc.ravel()])
            #np.savetxt('gatdata.csv', labels_auc)
            #np.savetxt('gatdata1.csv', pred_auc)

            try:
                sensitivity, specificity = sensitivity_specificity(data.y.cpu(), pred.cpu())
            except IndexError:
                pass
            #sensitivity, specificity = sensitivity_specificity(data.y.cpu(), pred.cpu())
            test_f1 = f1_score(data.y.cpu(), pred.cpu())
            try:
                test_auc = roc_auc_score(data.y.cpu(), pred.cpu())
            except ValueError:
                pass

            #sensitivity, specificity = sensitivity_specificity(data.y.cpu(), pred.cpu())
            #test_f1 = f1_score(data.y.cpu(), pred.cpu())
            #test_auc = roc_auc_score(data.y.cpu(), pred.cpu())

            sen += sensitivity * data.num_graphs
            sp += specificity * data.num_graphs

            f1 += test_f1 * data.num_graphs
            auc += test_auc * data.num_graphs

            running_loss += loss.item() * data.num_graphs
            running_acc += torch.sum(pred == data.y).item()
    toc = timer()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_acc = running_acc / n_sample

    print('Eval loss: {:.4f} Acc: {:.4f} time: {:.2f}s'.format(
        epoch_loss, epoch_acc, toc - tic))
    return epoch_acc, epoch_loss, sen / n_sample, sp / n_sample,f1/n_sample, auc/n_sample


def main( ):
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)

    n_tags = None

    test_acc_list, test_sp_list, test_sen_list = [], [], []
    best_acc_list, best_sp_list, best_sen_list,best_f1_list,best_auc_list = [], [], [],[],[]
    bestmeanacc_list, bestmeansp_list, bestmeansen_list,bestmeanf1_list,bestmeanauc_list = [],[],[], [], []
    #random_s = np.array([25, 50, 100, 125, 150, 175, 200, 225, 250, 275], dtype=int)
    #random_s = np.array([50, 100, 200, 300, 400, 500, 600, 700, 800, 900], dtype=int)
    random_s = np.array([25, 50, 100, 125, 150, 175, 200, 225, 250, 275], dtype=int)

    best_acc_list1, best_sp_list1, best_sen_list1 = [], [], []
    for i in range(7):

        #dataset = FSDataset(r'..\Resting1', folds=10, seed=i)
        #dataset = FSDataset(r'C:\Users\HP\Desktop\ROISignals_FunImgARCWF', folds=10, seed=random_s[i])
        #dataset = FSDataset(r'E:\graphmix-master-new\codes_graph\results', folds=10, seed=random_s[i])
        #dataset = FSDataset(r'F:\graphmix-master-new\codes_graph\HCP62ROI', folds=10, seed=random_s[i])
        dataset = FSDataset(r'F:/graphmix-master-new/codes_graph/ZDXX62pkl', folds=10, seed=random_s[i])
        for i in range(10):

            train_dset, val_dset, test_dset, train_split, valid_split, test_split = dataset.kfold_split(test_index=i)
            # dataset = dataset.fc

            # train_dset = GraphDataset(train_dset, n_tags, degree=True)
            input_size = train_dset[0].x.size(0)
            train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)

            model = FGDN(in_size=input_size, nb_class=2, d_model=args.dim_hidden, dropout=args.dropout,nb_layers=args.nb_layers)
            #model = GCN(in_size=input_size, nb_class=2, d_model=args.dim_hidden,nb_heads=args.nb_heads,dim_feedforward=2 * args.dim_hidden, dropout=args.dropout,nb_layers=args.nb_layers)
            #model = svm.SVC(C=10, kernel='sigmoid')

            if args.use_cuda:
                model.cuda()
            print("Total number of parameters: {}".format(count_parameters(model)))

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            #optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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

            print("Training...")
            best_val_acc = 0
            # best_model = None
            best_epoch = 0
            start_time = timer()
            for epoch in range(args.epochs):
                print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
                train_loss = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, args.use_cuda)
                val_acc, val_loss, val_sen, val_sp,val_f1,val_auc = eval_epoch(model, val_loader, criterion, args.use_cuda)
                # test_acc, test_loss, sensitivity, specificity = eval_epoch(model, test_loader, criterion, args.use_cuda)

                if args.warmup is None:
                    lr_scheduler.step(val_loss)

                if val_acc > best_val_acc:
                    best_val_acc, best_sp, best_sen,best_f1,best_auc = val_acc, val_sp, val_sen,val_f1,val_auc
                    best_epoch = epoch
                    # best_weights = copy.deepcopy(model.state_dict())

            total_time = timer() - start_time
            print("best epoch: {} best val acc: {:.4f}".format(best_epoch, best_val_acc))
            # model.load_state_dict(best_weights)

            # print("Testing...")
            # test_acc, test_loss, sensitivity, specificity = eval_epoch(model, val_loader, criterion, args.use_cuda)

            # print("test Acc {:.4f}".format(test_acc))

            best_acc_list.append(best_val_acc)
            best_sp_list.append(best_sp)
            best_sen_list.append(best_sen)
            best_f1_list.append(best_f1)
            best_auc_list.append(best_auc)
            print(
                "10 fold test average: {:.2f}±{:.2f} sensitivity:{:.2f}±{:.2f} specificity:{:.2f}±{:.2f} f1:{:.2f}±{:.2f} auc:{:.2f}±{:.2f}".format(
                    np.mean(best_acc_list) * 100, np.std(best_acc_list) * 100,
                    np.mean(best_sen_list) * 100, np.std(best_sen_list) * 100,
                    np.mean(best_sp_list) * 100, np.std(best_sp_list) * 100,
                    np.mean(best_f1_list) * 100, np.std(best_f1_list) * 100,
                    np.mean(best_auc_list) * 100, np.std(best_auc_list) * 100, ))

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

        print(
            "10 number 10 fold test average: {:.2f}±{:.2f} 10 number 10 fold specificity:{:.2f}±{:.2f} 10 number 10 fold sensitivity:{:.2f}±{:.2f} 10 number 10 fold f1:{:.2f}±{:.2f} 10 number 10 fold auc:{:.2f}±{:.2f} ".format(
                np.mean(bestmeanacc_list), np.std(bestmeanacc_list),
                np.mean(bestmeansen_list), np.std(bestmeansen_list),
                np.mean(bestmeansp_list), np.std(bestmeansp_list),
                np.mean(bestmeanf1_list), np.std(bestmeanf1_list),
                np.mean(bestmeanauc_list), np.std(bestmeanauc_list)))



if __name__ == "__main__":
    main()
