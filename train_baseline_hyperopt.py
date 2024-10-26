# -*- coding: utf-8 -*-
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

from transformer.utils import sensitivity_specificity

import torch.nn.functional as F
from torch_geometric.data.dataloader import DataLoader
# import DataLoader
from transformer.utils import count_parameters
from timeit import default_timer as timer
from torch import nn, optim

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL

from baseline.GAT import GAT
from baseline.GCN import GCN
from baseline.GIN import GIN
from baseline.GraphSAGE import GraphSAGE
from baseline.FGDN import FGDN

import matplotlib
import pickle
import atexit
from munch import DefaultMunch

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
    parser.add_argument('--device', type=str, default='cpu', help='specify cuda devices')
    args = parser.parse_args()
    args.use_cuda = False
        # torch.cuda.is_available()
    args.batch_norm = not args.layer_norm

    args.save_logs = False
    return args


def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch=10, use_cuda=False):
    model.train()

    running_loss = 0.0
    running_acc = 0.0

    tic = timer()
    for i, data in enumerate(loader):
        if args["warmup"] is not None:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)

        #if use_cuda:
        #    data = data.cuda()

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

            #optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)

            pred = output.data.argmax(dim=1)
            running_loss += loss.item() * len(data)
            running_acc += torch.sum(pred == data.y).item()
        toc = timer()

        n_sample = len(loader.dataset)
        epoch_loss = running_loss / n_sample
        epoch_acc = running_acc / n_sample
        print('Eval loss: {:.4f} Acc: {:.4f} time: {:.2f}s'.format(
            epoch_loss, epoch_acc, toc - tic))
        return epoch_acc, epoch_loss


def main(args):
    # global args
    args = load_args()
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #print(args)

    print(args)
    args = DefaultMunch.fromDict(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n_tags = None

    test_acc_list, test_sp_list, test_sen_list = [], [], []
    best_acc_list, best_sp_list, best_sen_list, best_f1_list, best_auc_list = [], [], [], [], []
    bestmeanacc_list, bestmeansp_list, bestmeansen_list, bestmeanf1_list, bestmeanauc_list = [], [], [], [], []
    random_s = np.array([25, 50, 100, 125, 150, 175, 200, 225, 250, 275], dtype=int)

    best_acc_list1, best_sp_list1, best_sen_list1 = [], [], []
    #for i in range(10):

        # dataset = FSDataset(r'..\Resting1', folds=10, seed=i)
        # dataset = FSDataset(r'C:\Users\HP\Desktop\ROISignals_FunImgARCWF', folds=10, seed=random_s[i])
        # dataset = FSDataset(r'E:\graphmix-master-new\codes_graph\results', folds=10, seed=random_s[i])
        #dataset = FSDataset(r'F:\graphmix-master-new\codes_graph\HCP62ROI', folds=10, seed=random_s[i])
    for i in range(10):
            dataset = FSDataset(r'F:\graphmix-master-new\codes_graph\HCP62ROI', folds=10, seed=random_s[i])
            train_dset, val_dset, test_dset, train_split, valid_split, test_split = dataset.kfold_split(test_index=i)
            # dataset = dataset.fc

            # train_dset = GraphDataset(train_dset, n_tags, degree=True)
            input_size = train_dset[0].x.size(0)
            train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)

            # model = GCN(in_size=input_size, nb_class=2, d_model=args.dim_hidden, dropout=args.dropout,nb_layers=args.nb_layers)
            model = GCN(in_size=input_size, nb_class=2, d_model=args.dim_hidden, dropout=args.dropout,
                        nb_layers=args.nb_layers)
            # model = svm.SVC(C=10, kernel='sigmoid')

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

            print("Training...")
            best_val_acc = 0
            # best_model = None
            best_epoch = 0
            logs = defaultdict(list)
            start_time = timer()
            for epoch in range(args.epochs):
                print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
                train_loss = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch)
                val_acc, val_loss = eval_epoch(model, val_loader, criterion)
                test_acc, test_loss = eval_epoch(model, test_loader, criterion)

                if args.warmup is None:
                    lr_scheduler.step(val_loss)

                logs['train_loss'].append(train_loss)
                logs['val_acc'].append(val_acc)
                logs['val_loss'].append(val_loss)
                logs['test_acc'].append(test_acc)
                logs['test_loss'].append(test_loss)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_weights = copy.deepcopy(model.state_dict())
            total_time = timer() - start_time
            print("best epoch: {} best val acc: {:.4f}".format(best_epoch, best_val_acc))
            model.load_state_dict(best_weights)

            print()
            print("Testing...")
            #test_acc, test_loss = eval_epoch(args, model, test_loader, criterion, args.use_cuda)

            print("test Acc {:.4f}".format(test_acc))
            return {
                "loss": -best_val_acc,
                'status': STATUS_OK,
                'params': args
            }


if __name__ == "__main__":
    def save_result(result_file, trials):

        print("正在保存结果...")
        with open(result_file, "w+") as f:
            for result in trials.results:
                if 'loss' in result and result['loss'] <= trials.best_trial['result']['loss']:
                    print(result, file=f)
        print("结果已保存 {:s}".format(result_file))
        # print(trials.best_trial)


    def initial_hyperopt(trial_file, result_file, max_evals):
        try:
            with open(trial_file, "rb") as f:
                trials = pickle.load(f)
            current_process = len(trials.results)
            print("使用已有的trial记录, 现有进度: {:d}/{:d} {:s}".format(current_process, max_evals, trial_file))
        except:
            trials = Trials()
            print("未找到现有进度, 从0开始训练 0/{:d} {:s}".format(max_evals, trial_file))
        atexit.register(save_result, result_file, trials)
        return trials

    max_evals = 200
    args = vars(load_args())
    #args['pos_enc'] = hp.choice('pos_enc',[None,'diffusion','pstep','adj'])
    #args['nb_heads'] = hp.choice('nb_heads', [4, 8])
    args['nb_layers'] = hp.choice('nb_layers', [1,2,3,4,5])
    args['lr'] = hp.choice('lr', [0.01, 0.005, 0.001])
    args['weight_decay'] = hp.choice('weight_decay', [0, 1e-4, 5e-4, 1e-5])
    args['dropout'] = hp.choice('dropout', [0, 0.3, 0.5, 0.8])

    save_root = os.path.join("hyperopt")
    result_file = os.path.join(save_root, f"result_gcn.log")
    trial_file = os.path.join(save_root, f"result_gcn.trial")

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    trials = initial_hyperopt(trial_file, result_file, max_evals)
    best = fmin(fn=main, space=args, algo=tpe.suggest, max_evals=max_evals,
        trials=trials, trials_save_file=trial_file)





