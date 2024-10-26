import pickle
import os
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import torch
import scipy.io as scio
from scipy import io
from torch.utils.data import Subset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import DataLoader, Data
from glob import glob
import re
import torch.nn as nn
import pandas as pd
import code_utils
import seaborn as sns
# from sympy import re
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import pingouin as pg

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def K_Fold(folds, dataset, seed):
    skf = KFold(folds, shuffle=True, random_state=seed)
    test_indices = []
    for _, index in skf.split(torch.zeros(len(dataset))):
        test_indices.append(index)

    return test_indices


class FSDataset(object):
    def __init__(self, root_dir, folds, seed):
        data_files = glob(os.path.join(root_dir,"**","*.pkl"))
        data_files.sort()
        self.fc = []
        for file in data_files:
            with open(file, "rb") as f:
                data = pickle.load(f)
            #adj = data.adjacency_mat

           # degree = data.degree
           #  degree1 = np.array(degree).reshape((90,1))
           #  degree1 =degree1.tolist()
           # clustering=data.clustering
           # clustering1 = np.array(clustering).reshape((90,1))
           # clustering1 = clustering1.tolist()
           # nodal_efficiency=data.nodal_efficiency
           # nodal_efficiency1= np.array(nodal_efficiency).reshape((90,1))
           # nodal_efficiency1 = nodal_efficiency1.tolist()

            #feature = data.source_mat.T
            #feature = np.hstack((feature,degree1))
            #feature =np.hstack((feature,clustering1))
            #feature =np.hstack((feature,nodal_efficiency1))


            feature = data.source_mat.T
            #feature =  np.delete(feature,1,axis = 1)
            #feature = np.delete(feature, 0, axis=1)
            #feature=torch.from_numpy(feature).float()
            #feature_nor = (feature- feature.min(axis=0)) / (feature.max(axis=0) - feature.min(axis=0))
            #BatchNorm = nn.BatchNorm1d(in_size)
            #feature = BatchNorm(feature)
            #feature = feature.detach().numpy()
            #feature_nor=torch.tensor(feature_nor)
            #feature = np.corrcoef(feature)
            feature = code_utils.partialCorrelationMatrix(feature)
            #feature=code_utils.cor2pcor(feature)
            #in_size=62
            #BatchNorm = nn.BatchNorm1d(in_size)
            #feature = BatchNorm(feature)
            #feature = feature.detach().numpy()

            #scaler = StandardScaler()
            #res_2 = scaler.fit_transform(feature)
            #res_2 = torch.tensor(res_2, dtype=torch.float).cuda()

            #偏相关
            #df = pd.DataFrame(feature)
            #df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
            #df2 = np.array(df2.pcorr())
            #sns.heatmap(df2, cmap='Blues', annot=False)
            #plt.matshow(df2)
            #plt.savefig('test7.png')

            #data = np.array(feature)
            #cov = np.cov(feature)

            #spearman
            #df = pd.DataFrame(feature)
            #df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
            #df2 = np.array(df.corr('spearman'))

            thresh_val=code_utils.get_thresh_val(feature)
            adj=code_utils.convert_binary_by_thresh_val(feature,thresh_val)

            #scaler = StandardScaler()
            #res_2 = scaler.fit_transform(feature)
            #res_2 = torch.tensor(res_2, dtype=torch.float).cuda()

            #thresh_val=code_utils.get_thresh_val(df2)
            #adj=code_utils.convert_binary_by_thresh_val(df2,thresh_val)

            #adj=code_utils.knn_generate_graph(data.source_mat)

            #feature = np.hstack((feature, feature1))
            # 取ROISignals_S(1)-1-0001的括号部分，1是MDD，2是NC，减一变成二分类用的0,1标签
            #label = int(re.match(r".*ROISignals_.+-(\d)-.+?\.pkl", file).group(1)) - 1
            label =int(re.match(r".*cortical_.+-(\d)-.+?\.pkl", file).group(1)) - 1
            #label=int(re.match(r".*cortical.+\d+", file).group(1))-1
            #label = 0 if file.split('/')[-3] == "female" else 1
            fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
            self.fc.append(Data(
                x=torch.from_numpy(feature).float(), edge_index=fcedge_index, y=torch.as_tensor(label).long()
                ))

        data_files_fake = glob(os.path.join(root_dir, "*.npz"))
        data_files_fake.sort()
        for file in data_files_fake:
            with open(file, "rb") as f:
               file_r = np.load(f)
               file_r.files
               fake_X=file_r['fake_X']
               #fake_A=file_r['fake_A']
               #fake_X= np.corrcoef(fake_X)
               #df = pd.DataFrame(fake_X)
               #df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
               #df2 = np.array(df2.pcorr())
               #fake_X = code_utils.partialCorrelationMatrix(fake_X)
               thresh_val = code_utils.get_thresh_val(fake_X)
               fake_A = code_utils.convert_binary_by_thresh_val(fake_X, thresh_val)
               #thresh_val = code_utils.get_thresh_val(fake_X)
               #fake_A = code_utils.convert_binary_by_thresh_val(fake_X, thresh_val)
               fcedge_index, _ = dense_to_sparse(torch.from_numpy(fake_A.astype(np.int16)))
               lst = file.split("/")[3].split("\\")[1]
               lst=int(lst[10])
               if lst==0:
                   label=0
               else:
                   label=1
               self.fc.append(Data(
                   x=torch.from_numpy(fake_X).float(), edge_index=fcedge_index, y=torch.as_tensor(label).long()
               ))



        self.k_fold = folds
        self.k_fold_split = K_Fold(self.k_fold, self.fc, seed)

    def kfold_split(self, test_index):
        assert test_index < self.k_fold
        # valid_index = (test_index + 1) % self.k_fold
        valid_index = test_index
        test_split = self.k_fold_split[test_index]
        valid_split = self.k_fold_split[valid_index]

        train_mask = np.ones(len(self.fc))
        train_mask[test_split] = 0
        train_mask[valid_split] = 0
        train_split = train_mask.nonzero()[0]

        train_subset = Subset(self.fc, train_split.tolist())
        valid_subset = Subset(self.fc, valid_split.tolist())
        test_subset = Subset(self.fc, test_split.tolist())

        return train_subset, valid_subset, test_subset, train_split,valid_split,test_split

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        return iter(self.fc)

if __name__ == "__main__":

   #data = io.loadmat(r"..\data_graph\MDD\ROISignals_FunImgARCWF\ROISignals_S1-1-0005\ROISignals_S1-1-0005.mat")
   data = io.loadmat(r"D:\Project\python\graphmix\data_graph\MDD\data_90_230\len_90\ROISignals_S5-1-0001.mat")
   print(len(data['ROISignals'][0]),len(data['ROISignals']))