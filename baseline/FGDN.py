import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import ChebConv

class FGDN(nn.Module):
    def __init__(self,in_size, nb_class, d_model, dropout=0.1, nb_layers=4):
        super(FGDN, self).__init__()
        #self.args = args
        self.features = in_size
        self.hidden_dim = d_model
        self.num_layers = nb_layers
        self.num_classes = nb_class
        self.dropout =dropout

        self.conv1 = ChebConv(self.features, self.hidden_dim,K=1)
        self.prelu_a1 = nn.Parameter(torch.Tensor([0.25]))
        self.prelu_a2 = nn.Parameter(torch.Tensor([0.25]))
        self.prelu_a3 = nn.Parameter(torch.Tensor([0.25]))

        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(ChebConv(self.hidden_dim, self.hidden_dim,K=1))

        #self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        #self.fc2 = nn.Linear(self.hidden_dim, 1)

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)

        self.bn1 = nn.BatchNorm1d(in_size)
        #self.bn1 = nn.BatchNorm1d(in_size)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.bn3 = nn.BatchNorm1d(d_model)

    def fc_forward(self, x):
        x = F.prelu(self.fc1(x),self.prelu_a3)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, data):
        #x, edge_index,batch = data.x, data.edge_index,data.batch
        #scaler = StandardScaler()
        #res_2 = torch.from_numpy(scaler.fit_transform(x.cpu()))
        #res_2 = torch.tensor(res_2,dtype=torch.float).cuda()
        #res_2 = F.prelu(self.conv1(res_2, edge_index),self.prelu_a1)
        #for conv in self.convs:
        #    res_2 = F.relu(conv(res_2, edge_index),self.prelu_a2)
        #res_2 = global_add_pool(res_2, batch)
        #res_2 = self.fc_forward(res_2)
        #x = F.log_softmax(x, dim=-1)
        #res_2 = F.log_softmax(res_2, dim=-1)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.bn1(x)
        x = F.relu(self.conv1(x, edge_index))
        #x = self.bn2(x)
        #x = self.bn1(x)
        #x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.bn3(x)
        x = global_add_pool(x, batch)

        x = self.fc_forward(x)
        return x
        #return res_2

    def __repr__(self):
        return self.__class__.__name__
