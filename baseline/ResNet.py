import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

# 定义18层和34层网络所用的残差结构
class ResNet(nn.Module):
    # 对应残差分支中主分支采用的卷积核个数有无发生变化，无变化设为1（即每一个残差结构主分支的第二层卷积层卷积核个数是第一层卷积层卷积核个数的1倍）
    expansion = 1

    # 输入特征矩阵深度、输出特征矩阵深度（主分支卷积核个数）、步长默认取1、下采样参数默认为None（对应虚线残差结构）
    def __init__(self, in_channel,nb_class,batch_size, out_channel, stride=1, downsample=None, dropout=0.1,**kwargs):
        super(ResNet, self).__init__()
        # 每一个残差结构中主分支第一个卷积层，注意步长是要根据是否需要改变channel而取1或取2的，不使用偏置（BN处理）
        #self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
        #                       kernel_size=1, stride=stride, padding=1, bias=False)
        self.conv1 = nn.Conv1d(in_channels=in_channel,out_channels=in_channel, kernel_size=2)
        # BN标准化处理，输入特征矩阵为conv1的out_channel
        self.bn1 = nn.BatchNorm1d(in_channel)
        # 激活函数
        self.relu = nn.ReLU()
        # 每一个残差结构中主分支第二个卷积层，输入特征矩阵为bn1的out_channel，该卷积层步长均为1，不使用偏置
        #self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
        #                       kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=in_channel, out_channels=in_channel, kernel_size=1)
        # BN标准化处理，输入特征矩阵为conv2的out_channel
        self.bn2 = nn.BatchNorm1d(in_channel)
        # 下采样方法，即侧分支为虚线
        self.downsample = downsample
        self.dropout = dropout

        #self.fc11 = nn.Linear(batch_size, out_channel)
        #self.fc12 = nn.Linear(out_channel, batch_size)

        self.fc11 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1)
        self.fc12 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)

        self.fc1 = nn.Linear(in_channel, out_channel)
        self.fc2 = nn.Linear(out_channel, out_channel // 2)
        self.fc3 = nn.Linear(out_channel // 2, nb_class)
        self.fc4 = nn.Linear(in_channel-1, in_channel)

    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x

    # 正向传播过程
    def forward(self, data, *args, **kwargs):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        m, n = np.shape(x)
        m = int(m/62)
        # 将输入特征矩阵赋值给identity（副分支的输出值）
        x = x.reshape([m, 62, 62])
        identity = x
        # 如果需要下采样方法，将输入特征矩阵经过下采样函数再赋值给identity
        if self.downsample is not None:
            identity = self.downsample(x)

        # 主分支的传播过程
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity =  self.conv1(identity)
        identity = self.bn1(identity)
        identity = self.conv2(identity)
        identity = self.bn2(identity)

        w = F.avg_pool2d(out, out.size(2))
        #w = w.reshape(1 , m)
        out_channal=64
        #fc11=nn.Linear(m, out_channal)
        #fc12=nn.Linear(out_channal, m)
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        w = F.relu(self.fc11(w))
        w = F.sigmoid(self.fc12(w))
        # Excitation
        #w=w.reshape(m,1,1)
        #w=w.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        #w=w.cuda()
        #out=out.to(device)
        out = out * w

        # 将主分支和副分支的输出相加再经过激活函数
        out += identity
        out = self.relu(out)
        out =out.reshape(m*62,61)
        out=self.fc4(out)
        out = global_add_pool(out, batch)
        out = self.fc_forward(out)
        #o,p,q = np.shape(out)
        return out
