from glob import glob

import torch
from sklearn import metrics
import numpy as np
from datetime import datetime
import pickle
import os
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import DataLoader, Data
from glob import glob
import re
import code_utils


def verify_dir_exists(dirname):
    if os.path.isdir(os.path.dirname(dirname)) == False:
        os.makedirs(os.path.dirname(dirname))


def print_ext(*args):
    print(str(datetime.now()), *args)


def preprocess(path, ):
    ###
            ###
            # reading labels
            #label.append(label_set[label_files]]

            data_files = glob(os.path.join(path, "**", "*.pkl"))
            data_files.sort()
            lable_all=[]
            fc_features=[]
            fc_adj=[]
            feature1=[]
            degree_all=[]
            threshold = 0.2
            for file in data_files:
                with open(file, "rb") as f:
                    data = pickle.load(f)
                #adj = data.adjacency_mat
                feature = data.source_mat.T
                #feature = np.corrcoef(feature)
                feature = code_utils.partialCorrelationMatrix(feature)
                feature1.append(feature)
                feature2=np.array(feature1)
                #feature2=feature2[np.triu_indices(90)]
                nsamples, nx, ny = feature2.shape
                d2_train_dataset = feature2.reshape((nsamples, nx * ny))

                thresh_val = code_utils.get_thresh_val(feature)
                adj = code_utils.convert_binary_by_thresh_val(feature, thresh_val)

                subj_fc_adj = adj
                fc_adj.append(subj_fc_adj)
                fc_adj_array = np.array(fc_adj)
                nsamples, nx, ny = fc_adj_array.shape
                train_dataset = fc_adj_array.reshape((nsamples, nx * ny))
                # 取ROISignals_S(1)-1-0001的括号部分，1是MDD，2是NC，减一变成二分类用的0,1标签
                lable=int(re.match(r".*cortical_.+-(\d)-.+?\.pkl", file).group(1))-1
                #fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
                lable_all.append(lable)
                #subj_fc_mat_list = feature.reshape((-1))
                #subj_fc_feature = (feature - min(subj_fc_mat_list)) / (
                #            max(subj_fc_mat_list) - min(subj_fc_mat_list))
                #fc_features.append(np.transpose(subj_fc_feature))
                #adj1.append(adj)
                degree = data.degree
                degree_all.append(degree)

            return np.array(degree_all), np.array(train_dataset), np.array(lable_all)


def compute_metrics(test_y, pre):
    accuracy = metrics.accuracy_score(test_y, pre)
    f1 = metrics.f1_score(test_y, pre)
    auc = metrics.roc_auc_score(test_y, pre)

    confusion_matrix = metrics.confusion_matrix(test_y, pre)
    tp = confusion_matrix[1][1]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    tn = confusion_matrix[0][0]
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)

    return accuracy, sensitivity, specificity, f1, auc


def mean_metrics(acc, sen, spe, f1_score, auc_score):
    acc = np.array(acc)
    sen = np.array(sen)
    spe = np.array(spe)
    f1_score = np.array(f1_score)
    auc_score = np.array(auc_score)
    mean_acc = np.mean(acc) * 100
    std_acc = np.std(acc) * 100
    mean_sensitivity = np.mean(sen) * 100
    std_sensitivity = np.std(sen) * 100
    mean_specificity = np.mean(spe) * 100
    std_specificity = np.std(spe) * 100
    mean_f1_score = np.mean(f1_score) * 100
    std_f1_score = np.std(f1_score) * 100
    mean_auc = np.mean(auc_score) * 100
    std_auc = np.std(auc_score) * 100
    return mean_acc, std_acc, mean_sensitivity, std_sensitivity, mean_specificity, std_specificity, mean_f1_score, std_f1_score, mean_auc, std_auc


def print_each_iter(no_folds, mean_acc, std_acc, mean_sensitivity, std_sensitivity, mean_specificity, std_specificity,
                    mean_f1_score, std_f1_score, mean_auc, std_auc):
    print('Result is: %.2f (± %.2f)' % (mean_acc, std_acc))
    print('sensitivity is: %.2f (± %.2f)' % (mean_sensitivity, std_sensitivity))
    print('specificity is: %.2f (± %.2f)' % (mean_specificity, std_specificity))
    print('f1 is: %.2f (± %.2f)' % (mean_f1_score, std_f1_score))
    print('auc is: %.2f (± %.2f)' % (mean_auc, std_auc))
    verify_dir_exists('results/')
    with open('results/%s.txt' % 'svm_every_iter', 'a+') as file:
        file.write('%s\t%d-fold\t\t%.2f (± %.2f)\t%.2f (± %.2f)\t%.2f (± %.2f)\t%.2f (± %.2f)\t%.2f (± %.2f)\n' % (
            str(datetime.now()), no_folds, mean_acc, std_acc, mean_sensitivity, std_sensitivity, mean_specificity,
            std_specificity, mean_f1_score, std_f1_score, mean_auc, std_auc))


def print_final(proportion, iter_time, acc_set, std_set, sen_set, sen_std_set, spe_set, spe_std_set, f1_set, f1_std_set,
                auc_set, auc_std_set):
    acc_mean = np.mean(acc_set)
    acc_std = np.std(acc_set)
    sen_mean = np.mean(sen_set)
    sen_std = np.std(sen_set)
    spe_mean = np.mean(spe_set)
    spe_std = np.std(spe_set)
    f1_mean = np.mean(f1_set)
    f1_std = np.std(f1_set)
    auc_mean = np.mean(auc_set)
    auc_std = np.std(auc_set)
    print_ext('finish!')
    verify_dir_exists('results/')
    with open('results/svm_final.txt', 'a+') as file:
        for iter_num in range(iter_time):
            print_ext('acc %d :    %.2f   sen :    %.2f   spe :    %.2f   f1 :    %.2f   auc :    %.2f' % (
                iter_num, acc_set[iter_num], sen_set[iter_num], spe_set[iter_num], f1_set[iter_num], auc_set[iter_num]))
            file.write(
                '%s\tacc %d :   \t%.2f (± %.2f)\tsen :   \t%.2f (± %.2f)\tspe :   \t%.2f (± %.2f)f1 :   \t%.2f (± %.2f)auc :   \t%.2f (± %.2f)\n' % (
                    str(datetime.now()), iter_num, acc_set[iter_num], std_set[iter_num], sen_set[iter_num],
                    sen_std_set[iter_num], spe_set[iter_num], spe_std_set[iter_num], f1_set[iter_num],
                    f1_std_set[iter_num], auc_set[iter_num], auc_std_set[iter_num]))
        print_ext(
            'acc:     %.2f(±%.2f)   sen:     %.2f(±%.2f)   spe:     %.2f(±%.2f)   f1:     %.2f(±%.2f)   auc:     %.2f(±%.2f)v' % (
                acc_mean, acc_std, sen_mean, sen_std, spe_mean, spe_std, f1_mean, f1_std, auc_mean, auc_std))
        file.write(
            '%s\t %.2f acc  :   \t%.2f (± %.2f)  sen  :   \t%.2f (± %.2f)  spe  :   \t%.2f (± %.2f)  f1  :   \t%.2f (± %.2f)  auc  :   \t%.2f (± %.2f)\n' % (
                str(datetime.now()), proportion, acc_mean, acc_std, sen_mean, sen_std, spe_mean, spe_std, f1_mean,
                f1_std, auc_mean, auc_std))