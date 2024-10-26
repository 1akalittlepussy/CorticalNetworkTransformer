# from torch.utils.data import Datasets
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch_geometric.utils as utils
import numpy as np

class GraphDataset(object):
    def __init__(self, dataset, n_tags=None, degree=False):
        """a pytorch geometric dataset as input
        """
        self.dataset = dataset
        self.n_features = dataset[0].x.shape[-1]
        self.pe_list = None
        self.lap_pe_list = None
        self.degree_list = None
        if degree:
            self.compute_degree()
        self.n_tags = n_tags
        self.one_hot()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        if self.x_onehot is not None and len(self.x_onehot) == len(self.dataset):
            data.x_onehot = self.x_onehot[index]
        if self.pe_list is not None and len(self.pe_list) == len(self.dataset):
            data.pe = self.pe_list[index]
        if self.lap_pe_list is not None and len(self.lap_pe_list) == len(self.dataset):
            data.lap_pe = self.lap_pe_list[index]
        if self.degree_list is not None and len(self.degree_list) == len(self.dataset):
            data.degree = self.degree_list[index]
        return data

    def compute_degree(self):
        self.degree_list = []
        for g in self.dataset:
            deg = 1. / torch.sqrt(1. + utils.degree(g.edge_index[0], g.num_nodes))
            self.degree_list.append(deg)

    def input_size(self):
        if self.n_tags is None:
            return self.n_features
        return self.n_tags

    def one_hot(self):
        self.x_onehot = None
        if self.n_tags is not None and self.n_tags > 1:
            self.x_onehot = []
            for g in self.dataset:
                onehot = F.one_hot(g.x.view(-1).long(), self.n_tags)
                self.x_onehot.append(onehot)

    def collate_fn(self):
        def collate(batch):
            batch = list(batch)
            max_len = max(len(g.x) for g in batch)

            if self.n_tags is None:
                padded_x = torch.zeros((len(batch), max_len, self.n_features))
            else:
                # discrete node attributes
                padded_x = torch.zeros((len(batch), max_len, self.n_tags))
            mask = torch.zeros((len(batch), max_len), dtype=bool)
            adjs = torch.zeros((len(batch),max_len, max_len), dtype=torch.float32)
            labels = []
            edge_indice = []
            # TODO: check if position encoding matrix is sparse
            # if it's the case, use a huge sparse matrix
            # else use a dense tensor
            pos_enc = None
            use_pe = hasattr(batch[0], 'pe') and batch[0].pe is not None

            if use_pe:
                if not batch[0].pe.is_sparse:
                    pos_enc = torch.zeros((len(batch), max_len, max_len))
                else:
                    print("Not implemented yet!")

            #edge = None
            #use_edge = hasattr(batch[0], 'edge_index') and batch[0].edge_index is not None
            #if use_edge:
            #    edge_len = batch[0].edge_index.shape[-1]
            #    edge = torch.zeros((len(batch),2, edge_len))

            # process lap PE
            lap_pos_enc = None
            use_lap_pe = hasattr(batch[0], 'lap_pe') and batch[0].lap_pe is not None
            if use_lap_pe:
                lap_pe_dim = batch[0].lap_pe.shape[-1]
                lap_pos_enc = torch.zeros((len(batch), max_len, lap_pe_dim))

            degree = None
            use_degree = hasattr(batch[0], 'degree') and batch[0].degree is not None
            if use_degree:
                degree = torch.zeros((len(batch), max_len))

            spatial_pe = torch.zeros((len(batch), max_len, max_len))

            for i, g in enumerate(batch):
                labels.append(g.y)
                size = torch.Size([max_len, max_len])
                g.edge_attr = edge_attr = torch.ones(g.edge_index.size(1), dtype=torch.float)
                edge_indice.append(g.edge_index)
                adj = torch.sparse_coo_tensor(g.edge_index, edge_attr, size)
                adj = adj.to_dense()
                #adj_np=adj.numpy()
                #M,path = floyd_warshall_py(adj.numpy())
                #M= torch.from_numpy(M)
                g_len = len(g.x)
                #e_len = len(g.edge_index)
                adjs[i, :g_len, :g_len] =adj      

                if self.n_tags is None:
                    padded_x[i, :g_len, :] = g.x
                else:
                    padded_x[i, :g_len, :] = g.x_onehot
                mask[i, g_len:] = True
                if use_pe:
                    pos_enc[i, :g_len, :g_len] = g.pe
                if use_lap_pe:
                    lap_pos_enc[i, :g_len, :g.lap_pe.shape[-1]] = g.lap_pe
                if use_degree:
                    degree[i, :g_len] = g.degree
                #if use_edge:
                #    edge[i, :2, :g.edge_index[-1]]=g.edge_index
                #spatial_pe[i, :g_len, :g_len]= M

            return padded_x, mask, pos_enc, lap_pos_enc, degree, adjs, default_collate(labels), spatial_pe
        return collate

def floyd_warshall_py(adjacency_matrix):
    n = adjacency_matrix.shape[0]

    adj_mat_copy = adjacency_matrix.astype(np.int64, order='C', copy=True)
    M = np.copy(adj_mat_copy)
    path = -1 * np.ones((n, n), dtype=np.int64)

    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510

    # Floyd's algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                cost_ikkj = M[i][k] + M[k][j]
                if M[i][j] > cost_ikkj:
                    M[i][j] = cost_ikkj
                    path[i][j] = k

    # set unreachable path to 510
    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510:
                path[i][j] = -1
                M[i][j] = -1

    return M, path

def floyd_warshall(adjacency_matrix):

    nrows, ncols = adjacency_matrix.shape
    n=nrows
    #adj_mat_copy = adjacency_matrix.astype(np.int64, order='C', casting='safe', copy=True)
    #adj_mat_copy = adjacency_matrix.astype(long, order='C', casting='safe', copy=True)
    #assert adj_mat_copy.flags['C_CONTIGUOUS']
    #cdef numpy.ndarray[long, ndim=2, mode='c'] M = adj_mat_copy
    #cdef numpy.ndarray[long, ndim=2, mode='c'] path = -1 * numpy.ones([n, n], dtype=numpy.int64)

    M = adjacency_matrix.copy()
    path = -1 * np.ones((n, n), dtype=np.int64)
    M_ptr = np.ascontiguousarray(M).ravel().view(np.int64)

    #int i, j, k
    #cdef long M_ij, M_ik, cost_ikkj
    #cdef long* M_ptr = &M[0,0]
    #cdef long* M_i_ptr
    #cdef long* M_k_ptr
    #M_ptr=0
    #M=numpy.ndarray[long, ndim=2, mode='c']

    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510

    # floyed algo
    for k in range(n):
        M_k_ptr = M_ptr + n*k
        for i in range(n):
            M_i_ptr = M_ptr + n*i
            M_ik = M_i_ptr[k]
            for j in range(n):
                cost_ikkj = M_ik + M_k_ptr[j]
                M_ij = M_i_ptr[j]
                if M_ij > cost_ikkj:
                    M_i_ptr[j] = cost_ikkj
                    path[i][j] = k

    # set unreachable path to 510
    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510:
                path[i][j] = 510
                M[i][j] = 510

    return M, path
