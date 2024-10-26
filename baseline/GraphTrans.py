import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


def normalize(data):
    _mean = np.mean(data, axis=0)
    _std = np.std(data, axis=0)
    new_data = (data - _mean) / _std
    return new_data

class GraphTrans(nn.Module):
    def __init__(self, in_size, nb_class, d_model,nb_heads,
                 dim_feedforward=2048,dropout=0.1,nb_layers=4):
        super(GraphTrans, self).__init__()
        self.features = in_size
        self.hidden_dim = d_model
        self.num_layers = nb_layers
        self.num_classes = nb_class
        self.dropout = dropout

        self.conv1 = GCNConv(self.features, self.hidden_dim)
        self.pool1 = SAGPool(self.hidden_dim, ratio=0.8)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
        self.pool2 = SAGPool(self.hidden_dim, ratio=0.8)

        self.bn1 = nn.BatchNorm1d(in_size)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
        )

        self.embedding = nn.Linear(in_features=in_size,
                                   out_features=d_model,
                                   bias=False)

        self.embedding1 = nn.Linear(in_features=d_model,
                                   out_features=in_size,
                                   bias=False)

        self.fc1 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nb_heads, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, nb_layers)

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
        #adj = to_dense_adj(edge_index, batch)
        #batch_x, mask = to_dense_batch(x, batch)
        #res_1 = normalize(x)
        #scaler = StandardScaler()
        #res_2 = torch.from_numpy(scaler.fit_transform(x.cpu()))
        #res_2 = torch.tensor(res_2,dtype=torch.float).cuda()

        #res_2 = F.relu(self.conv1(res_2, edge_index))
        #for conv in self.convs:
        #    res_2 = F.relu(conv(res_2, edge_index))

        #res_2 = global_add_pool(res_2, batch)
        m, n = np.shape(x)
        m = int(m/62)
        # 将输入特征矩阵赋值给identity（副分支的输出值）
        x = x.reshape([m, 62, 62])
        #res_2 = self.fc_forward(res_2)
        x = self.bn1(x)
        x = x.permute(1, 0, 2)
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.embedding1(x)
        x = x.reshape([m*62,62])
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x=x1+x2
        #x = global_add_pool(x, batch)

        x = self.fc_forward(x)
        return x

        #return res_2

    def __repr__(self):
        return self.__class__.__name__

class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch,perm
