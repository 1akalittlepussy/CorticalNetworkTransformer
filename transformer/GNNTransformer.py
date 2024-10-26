# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
#import torch_geometric.nn.Linear as Linear_pyg
from transformer.GT_layers import DiffTransformerEncoderLayer
import torch_geometric.nn as tnn
from torch_geometric.nn import GATConv, SAGEConv, GINConv
from torch.nn.modules.utils import _triple, _pair, _single
from torch_geometric.nn.pool.topk_pool import topk,filter_adj

class DiffTransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, pe, adj, degree=None, mask=None, src_key_padding_mask=None, JK=False):
        output = src
        xs = []
        for mod in self.layers:
            output = mod(output, pe=pe,adj=adj, degree=degree, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)
            xs.append(output)
        if self.norm is not None:
            output = self.norm(output)
        if JK:
            output = torch.cat(xs, -1)
        return output


class Affine(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, dim, 1]),requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]),requires_grad=True)

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x

class GNNTransformer(nn.Module):
    def __init__(self, in_size, nb_class, nb_heads, d_model,
                 dim_feedforward=2048, dropout=0.1, nb_layers=4,
                 lap_pos_enc=False, lap_pos_enc_dim=0,batch_norm=False):
        super(GNNTransformer, self).__init__()

        #def mlp(in_features, hid, out_features):
        #    return nn.Sequential(
        #        nn.Linear(in_features, hid),
        #        nn.ReLU(inplace=True),
        #        nn.Linear(hid, out_features),
        #    )

        #self.embedding = tnn.Sequential(
        #    'x, adj', [
        #    (tnn.DenseGCNConv(in_size,in_size), 'x, adj -> x'),
        #    nn.ReLU(inplace=True),
        #    nn.BatchNorm1d(in_size),
        #    (tnn.DenseGCNConv(in_size,d_model), 'x, adj -> x'),
        #    nn.ReLU(inplace=True),
        #    nn.BatchNorm1d(d_model)
        #    ])

        def mlp(in_features, hid, out_features):
            return nn.Sequential(
                nn.Linear(in_features, hid),
                nn.ReLU(inplace=True),
                nn.Linear(hid, out_features),
            )

        self.embedding = tnn.Sequential(
            'x, adj', [
                (tnn.DenseGINConv(mlp(in_size, d_model, d_model)), 'x, adj -> x'),
                nn.ReLU(inplace=True),
                #nn.BatchNorm1d(d_model),
                (tnn.DenseGINConv(mlp(d_model, d_model, d_model)), 'x, adj -> x'),
                nn.ReLU(inplace=True),
                #nn.BatchNorm1d(d_model)
            ])

        #deg=torch.from_numpy(np.array(pna_degrees))
        #deg= adj.sum(-1).long().unsqueeze(-1)
        #gin_nn = nn.Sequential(tnn.Linear(in_size, in_size))
        #self.embedding= tnn.GINEConv(gin_nn)

        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        if lap_pos_enc and lap_pos_enc_dim > 0:
            # We embed the pos. encoding in a higher dim space
            # as Bresson et al. and add it to the features.
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)
        #encoder_layer = nn.TransformerEncoderLayer(
        #    d_model, nb_heads, dim_feedforward, dropout)
        #self.encoder = nn.TransformerEncoder(encoder_layer, nb_layers)
        encoder_layer = DiffTransformerEncoderLayer(
            d_model, nb_heads, dim_feedforward, dropout, batch_norm=batch_norm)
        self.encoder = DiffTransformerEncoder(encoder_layer, nb_layers)
        self.pooling = GlobalSum1D()
        self.pooling2 = GlobalMax1D()

        #self.classifier = nn.Sequential(
        #    nn.Linear(d_model, d_model),
        #    nn.ReLU(True),
        #    nn.Linear(d_model, nb_class)
        #)

        #self.classifier = nn.Sequential(
        #    nn.Dropout(dropout, inplace=True),
        #    nn.Linear(d_model*nb_layers, d_model),
        #    nn.ReLU(True),
        #    nn.Linear(d_model, nb_class)
        #)

        #using readout (sum|max)
        self.classifier = nn.Sequential(
             nn.Dropout(dropout, inplace=True),
             nn.Linear(d_model*2, d_model),
             nn.ReLU(True),
             nn.Linear(d_model, nb_class)
            )
        self.bn1 = nn.BatchNorm1d(in_size)
        self.norm1 = nn.BatchNorm1d(in_size)
        self.GELU = nn.GELU()
        self.pre_affine = Affine(in_size)
        self.post_affine = Affine(in_size)

    def forward(self, x, adj, masks, pe, x_lap_pos_enc=None, degree=None):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        x = self.bn1(x)
        input = x
        input_1 = self.embedding(input,adj)
        input_1 = self.pre_affine(input_1)
        #Affine(input_1)
        input_1 =self.norm1(input_1)
        input_1 = self.GELU(input_1)
        input_1 = self.post_affine(input_1)
        output = input_1.permute(1, 0, 2)
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            output = output + x_lap_pos_enc
        #output = self.encoder(output, src_key_padding_mask=masks)
        output = self.encoder(output,pe,adj, degree=degree, src_key_padding_mask=masks, JK=False)
        output = output.permute(1, 0, 2)
        # we make sure to correctly take the masks into account when pooling
        #output = self.pooling(output, masks)
        # we only do mean pooling for now.
        output = torch.cat([self.pooling(output, masks),self.pooling2(output, masks)], dim=-1)
        return self.classifier(output)


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