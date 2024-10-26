# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch_geometric.utils as utils
from torch_geometric.data import Data
import numpy as np
import os


def my_inc(self, key, value, *args, **kwargs):
    if key == 'subgraph_edge_index':
        return self.num_subgraph_nodes
    if key == 'subgraph_node_idx':
        return self.num_nodes
    if key == 'subgraph_indicator':
        return self.num_nodes
    elif 'index' in key:
        return self.num_nodes
    else:
        return 0

class GraphDataset(object):
    def __init__(self, dataset, degree=False, k_hop=2, se="gnn", use_subgraph_edge_attr=False,n_tags=None,return_complete_index=True):
        self.dataset = dataset
        self.n_features = dataset[0].x.shape[-1]
        self.degree = degree
        self.compute_degree()
        self.abs_pe_list = None
        self.return_complete_index = return_complete_index
        self.use_subgraph_edge_attr = use_subgraph_edge_attr
        self.k_hop = k_hop
        self.se = se
        self.n_tags = n_tags

    def compute_degree(self):
        if not self.degree:
            self.degree_list = None
            return
        self.degree_list = []
        for g in self.dataset:
            deg = 1. / torch.sqrt(1. + utils.degree(g.edge_index[0], g.num_nodes))
            self.degree_list.append(deg)

    def input_size(self):
        if self.n_tags is None:
            return self.n_features
        return self.n_tags

    def extract_subgraphs(self):
        print("Extracting {}-hop subgraphs...".format(self.k_hop))
        # indicate which node in a graph it is; for each graph, the
        # indices will range from (0, num_nodes). PyTorch will then
        # increment this according to the batch size
        self.subgraph_node_index = []

        self.subgraph_edge_index = []

        self.subgraph_indicator_index = []

        if self.use_subgraph_edge_attr:
            self.subgraph_edge_attr = []

        for i in range(len(self.dataset)):
            if self.cache_path is not None:
                filepath = "{}_{}.pt".format(self.cache_path, i)
                if os.path.exists(filepath):
                    continue
            graph = self.dataset[i]
            node_indices = []
            edge_indices = []
            edge_attributes = []
            indicators = []
            edge_index_start = 0

            for node_idx in range(graph.num_nodes):
                sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
                    node_idx,
                    self.k_hop,
                    graph.edge_index,
                    relabel_nodes=True,
                    num_nodes=graph.num_nodes
                    )
                node_indices.append(sub_nodes)
                edge_indices.append(sub_edge_index + edge_index_start)
                indicators.append(torch.zeros(sub_nodes.shape[0]).fill_(node_idx))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    edge_attributes.append(graph.edge_attr[edge_mask]) # CHECK THIS DIDN"T BREAK ANYTHING
                edge_index_start += len(sub_nodes)

            self.subgraph_node_index.append(torch.cat(node_indices))
            self.subgraph_edge_index.append(torch.cat(edge_indices, dim=1))
            self.subgraph_indicator_index.append(torch.cat(indicators))
            if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                self.subgraph_edge_attr.append(torch.cat(edge_attributes))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]

        if self.n_features == 1:
            data.x = data.x.squeeze(-1)
        if not isinstance(data.y, list):
            data.y = data.y
        n = data.num_nodes
        s = torch.arange(n)
        if self.return_complete_index:
            data.complete_edge_index = torch.vstack((s.repeat_interleave(n), s.repeat(n)))
        data.degree = None
        if self.degree:
            data.degree = self.degree_list[index]
        data.abs_pe = None
        if self.abs_pe_list is not None and len(self.abs_pe_list) == len(self.dataset):
            data.abs_pe = self.abs_pe_list[index]

            data.num_subgraph_nodes = None
            data.subgraph_node_idx = None
            data.subgraph_edge_index = None
            data.subgraph_indicator = None

        return data