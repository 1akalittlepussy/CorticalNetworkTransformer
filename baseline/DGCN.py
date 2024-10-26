# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
#import torch_geometric.nn.Linear as Linear_pyg
from .layers import DiffTransformerEncoderLayer
import torch_geometric.nn as tnn
from torch_geometric.nn import GATConv, SAGEConv, GINConv
from torch.nn.modules.utils import _triple, _pair, _single
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
#import adapool_cuda
import numpy

class GraphTransformer(nn.Module):
    def __init__(self, in_size, nb_class, d_model, nb_heads,
                 dim_feedforward=2048, dropout=0.1, nb_layers=4,
                 lap_pos_enc=False, lap_pos_enc_dim=0):
        super(GraphTransformer, self).__init__()

        self.embedding = nn.Linear(in_features=in_size,
                                   out_features=d_model,
                                   bias=False)

        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        if lap_pos_enc and lap_pos_enc_dim > 0:
            # We embed the pos. encoding in a higher dim space
            # as Bresson et al. and add it to the features.
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nb_heads, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, nb_layers)
        self.pooling = GlobalSum1D()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
        )
        self.bn1 = nn.BatchNorm1d(in_size)
        self.fc1 = nn.Linear(62, 128)
        self.fc2 = nn.Linear(62, 128)
        self.fc3 = nn.Linear(62, self.part_num)

    def forward(self, x, adj, masks, x_pe, x_lap_pos_enc=None, degree=None):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        #x = self.bn1(x)
        x = x.permute(1, 0, 2)
        q = self.fc1(x)
        k = self.fc2(x)
        v = self.fc3(x)
        corr_matrix = self.mha(q, k, v)

        output = self.embedding(x)
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            output = output + x_lap_pos_enc
        output = self.encoder(output, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)
        # we make sure to correctly take the masks into account when pooling
        output = self.pooling(output, masks)
        # we only do mean pooling for now.
        return self.classifier(output)

class DiffTransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, pe, degree=None, mask=None, src_key_padding_mask=None, JK=False):
        output = src
        xs = []
        for mod in self.layers:
            output,attn = mod(output, pe=pe, degree=degree, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)
            xs.append(output)
        if self.norm is not None:
            output = self.norm(output)
        if JK:
            output = torch.cat(xs, -1)
        return output,attn

class DiffGraphTransformer(nn.Module):
    # This is a variant of the GraphTransformer, where the node positional
    # information is injected in the attention score instead of being
    # added to the node features. This is in the spirit of relative
    # pos encoding rather than Vaswani et al.
    def __init__(self, in_size, nb_class, d_model, nb_heads,
                 dim_feedforward=2048, dropout=0.1, nb_layers=2,
                 batch_norm=False, lap_pos_enc=False, lap_pos_enc_dim=0):
        super(DiffGraphTransformer, self).__init__()
        def mlp(in_features, hid, out_features):
            return nn.Sequential(
                nn.Linear(in_features, hid),
                nn.ReLU(inplace=True),
                nn.Linear(hid, out_features),
            )

        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        self.dropout = dropout
        if lap_pos_enc and lap_pos_enc_dim > 0:
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)
        self.degree_encoding = nn.Embedding(50, d_model)


        self.embedding = nn.Linear(in_features=in_size,
                                   out_features=d_model,
                                   bias=False)

        encoder_layer = DiffTransformerEncoderLayer(
            d_model, nb_heads, dim_feedforward, dropout, batch_norm=batch_norm)
        self.encoder = DiffTransformerEncoder(encoder_layer, nb_layers)
        #self.DAFF = DAFF(64,64,64)
        self.pooling = GlobalSum1D()
        self.pooling2 = GlobalMax1D()

        self.gcembedding = tnn.Sequential(
            'x, adj', [
                (tnn.DenseGINConv(mlp(in_size, d_model, d_model)), 'x, adj -> x'),
                nn.ReLU(inplace=True),
                (tnn.DenseGINConv(mlp(d_model, d_model, d_model)), 'x, adj -> x'),
                nn.ReLU(inplace=True)
            ])

        self.classifier = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(d_model*2, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
        )

        self.bn1 = nn.BatchNorm1d(in_size)

    def forward(self, x, adj, masks, pe, x_lap_pos_enc=None, degree=None):
        #x = self.bn1(x)
        l,m,n=numpy.shape(x)
        x=x.view(l*m,n)
        x=self.bn1(x)
        x=x.view(l,m,n)
        x = x.permute(1, 0, 2)
        output = self.embedding(x)
        outx = F.dropout(output, self.dropout, self.training)
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            #$output = output.permute(1, 0, 2) + x_lap_pos_enc
            output = output + x_lap_pos_enc
        #degree_int = adj.sum(-1).long().unsqueeze(-1)
        #degree_encoding = self.degree_encoding(degree_int)
        #output = output + degree_encoding.transpose(0, 1)
        output, attn = self.encoder(output, pe, degree=degree, src_key_padding_mask=masks, JK=False)
        x = x.permute(1, 0, 2)
        x = self.gcembedding(x, attn)
        #output = self.DAFF(output,edge_index,batch)
        output = output.permute(1, 0, 2)
        ## normal and JK
        #output = self.pooling(output, masks)
        ## readout (sum|max)
        output = torch.cat([self.pooling(output, masks),self.pooling2(output, masks)], dim=-1)
        output2 = self.classifier(output)
        return output2


class GlobalAvg1D(nn.Module):
    def __init__(self):
        super(GlobalAvg1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=1)
        mask = (~mask).float().unsqueeze(-1)
        x = x * mask
        return x.sum(dim=1) / mask.sum(dim=1)


class GlobalSum1D(nn.Module):
    def __init__(self):
        super(GlobalSum1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=1)
        mask = (~mask).float().unsqueeze(-1)
        x = x * mask
        return x.sum(dim=1)


class GlobalMax1D(nn.Module):
    def __init__(self):
        super(GlobalMax1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=1)
        mask = (~mask).float().unsqueeze(-1)
        x = x * mask
        return x.max(dim=1)[0]
