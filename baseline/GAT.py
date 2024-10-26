import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GATConv, SAGEConv, GATConv
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_dense_batch, to_dense_adj


def normalize(data):
    _mean = np.mean(data, axis=0)
    _std = np.std(data, axis=0)
    new_data = (data - _mean) / _std
    return new_data

class GAT(nn.Module):
    def __init__(self, in_size, nb_class, d_model, dropout=0.3, nb_layers=4):
        super(GAT, self).__init__()
        self.features = in_size
        self.hidden_dim = d_model
        self.num_layers = nb_layers
        self.num_classes = nb_class
        self.dropout = dropout

        self.conv1 = GATConv(self.features, self.hidden_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(GATConv(self.hidden_dim, self.hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
        )

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)
        self.bn1 = nn.BatchNorm1d(in_size)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.bn3 = nn.BatchNorm1d(d_model)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x

    def forward(self, data, *args, **kwargs):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.bn1(x)
        x = F.relu(self.conv1(x, edge_index))
        x= self.bn2(x)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.bn3(x)
        x = global_add_pool(x, batch)
        x = self.fc_forward(x)

        return x

    def __repr__(self):
        return self.__class__.__name__
