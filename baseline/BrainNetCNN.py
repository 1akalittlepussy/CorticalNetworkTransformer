import torch
import numpy as np

import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class Edge2Edge(nn.Module):
    def __init__(self, channel):
        super(Edge2Edge, self).__init__()
        self.channel = channel
        #self.dim = dim
        #self.filters = filters
        #self.row_conv = nn.Conv2d(channel, filters, (1, dim))
        self.row_conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=2)
        #self.col_conv = nn.Conv2d(channel, filters, (dim, 1))
        self.col_conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=2)

    # implemented by two conv2d with line filter
    def forward(self, x):
        #size = x.size()
        row = self.row_conv(x)
        col = self.col_conv(x)
        #row_ex = row.expand(size[0], self.filters, self.dim, self.dim)
        #col_ex = col.expand(size[0], self.filters, self.dim, self.dim)
        return row+col


# BrainNetCNN edge to node layer
class Edge2Node(nn.Module):
    def __init__(self, channel):
        super(Edge2Node, self).__init__()
        self.channel = channel
        #self.dim = dim
        #self.filters = filters
        #self.row_conv = nn.Conv2d(channel, filters, (1, dim))
        #self.col_conv = nn.Conv2d(channel, filters, (dim, 1))
        self.row_conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=2)
        self.col_conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=2)


    def forward(self, x):
        row = self.row_conv(x)
        col = self.col_conv(x)
        return row+col


# BrainNetCNN node to graph layer
class Node2Graph(nn.Module):
    def __init__(self, channel):
        super(Node2Graph, self).__init__()
        self.channel = channel
        #self.dim = dim
        #self.filters = filters
        self.conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=2)
        #self.conv = nn.Conv2d(channel, filters, (dim, 1))

    def forward(self, x):
        return self.conv(x)


# BrainNetCNN network
class BCNN(nn.Module):
    def __init__(self, e2e, e2n, n2g, dropout,out_channel,nb_class):
        super(BCNN, self).__init__()
        self.n2g_filter = n2g
        self.e2e = Edge2Edge(e2e)
        self.e2n = Edge2Node(e2e)
        self.dropout = nn.Dropout(p=dropout)
        self.n2g = Node2Graph(e2n)
        self.fc = nn.Linear(n2g, 1)
        self.BatchNorm = nn.BatchNorm1d(n2g)
        self.bn1 = nn.BatchNorm1d(e2e)
        self.bn2 = nn.BatchNorm1d(e2e-1)
        self.bn3 = nn.BatchNorm1d(e2e-2)
        self.bn4 = nn.BatchNorm1d(e2e-3)
        self.fc1 = nn.Linear(e2e, out_channel)
        self.fc2 = nn.Linear(out_channel, out_channel // 2)
        self.fc3 = nn.Linear(out_channel // 2, nb_class)
        self.fc4 = nn.Linear(e2e - 3, e2e)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)


    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x=self.bn1(x)
        m, n = np.shape(x)
        m = int(m/62)
        # 将输入特征矩阵赋值给identity（副分支的输出值）
        x = x.reshape([m, 62, 62])
        x = self.e2e(x)
        x = self.dropout(x)
        x = x.reshape(m * 62, 61)
        x = self.bn2(x)
        x = x.reshape(m ,62, 61)
        x = self.e2n(x)
        x = self.dropout(x)
        x = x.reshape(m * 62, 60)
        x = self.bn3(x)
        x = x.reshape(m, 62, 60)
        x = self.n2g(x)
        x = self.dropout(x)
        x = x.reshape(m * 62, 59)
        #x = self.bn3(x)
        #x = x.view(-1, self.n2g_filter)
        #x = x.reshape(m * 62, 59)
        x=self.fc4(x)
        #x=self.bn1(x)
        x = global_add_pool(x, batch)
        x = self.fc_forward(x)
        #x = self.fc(self.BatchNorm(x))

        return x
