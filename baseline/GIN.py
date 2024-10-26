import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from sklearn.preprocessing import StandardScaler

class GIN(torch.nn.Module):
    def __init__(self, in_size, nb_class, d_model, dropout=0.1, nb_layers=4):
        super(GIN, self).__init__()
        self.features = in_size
        self.hidden_dim = d_model
        self.num_layers = nb_layers
        self.num_classes = nb_class
        self.dropout = dropout
        self.conv1 = GINConv(
            Sequential(
                Linear(self.features, self.hidden_dim),
                ReLU(),
                Linear(self.hidden_dim, self.hidden_dim),
                ReLU(),
                BN(self.hidden_dim),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(self.hidden_dim, self.hidden_dim),
                        ReLU(),
                        Linear(self.hidden_dim, self.hidden_dim),
                        ReLU(),
                        BN(self.hidden_dim),
                    ), train_eps=True))

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)
        self.bn1 = nn.BatchNorm1d(in_size)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.bn3 = nn.BatchNorm1d(d_model)

    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x

    def forward(self, data, *args, **kwargs):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = torch.ones(edge_index.size(1))
        #adj = to_dense_adj(edge_index, batch)
        #batch_x, mask = to_dense_batch(x, batch)
        #res_1 = normalize(x)
        #x = self.bn1(x)
        #x = F.relu(self.conv1(x, edge_index))
        #x= self.bn2(x)
        #for conv in self.convs:
        #    x = F.relu(conv(x, edge_index))
        #x = self.bn3(x)
        #x = global_add_pool(x, batch)
        scaler = StandardScaler()
        res_2 = torch.from_numpy(scaler.fit_transform(x.cpu()))
        res_2 = torch.tensor(res_2,dtype=torch.float).cuda()

        res_2 = F.relu(self.conv1(res_2, edge_index))
        for conv in self.convs:
            res_2 = F.relu(conv(res_2, edge_index))

        res_2 = global_add_pool(res_2, batch)

        res_2 = self.fc_forward(res_2)

        return res_2
    def __repr__(self):
        return self.__class__.__name__
