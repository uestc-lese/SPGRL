import numpy as np
import scipy.sparse as sp
import torch

def load_data(dataset, label_rate, topk):
    '''
    print('loading data...')
    graph_path = './data/' + dataset + '/' + dataset + '.edge'
    train_index_path = './data/' + dataset + '/train' + str(label_rate) + '.txt'
    test_index_path = './data/' + dataset + '/test' + str(label_rate) + '.txt'
    label_path = './data/' + dataset + '/' + dataset + '.label'
    fgraph_path = './data/' + dataset + '/knn/c' + str(topk) + '.txt'
    feature_path = './data/' + dataset + '/' + dataset + '.feature'
    '''
    print('loading data...')
    graph_path = '/home/liuzhenyu/FRY/graph_clip/data/' + dataset + '/' + dataset + '.edge'
    train_index_path = '/home/liuzhenyu/FRY/graph_clip/data/' + dataset + '/train' + str(label_rate) + '.txt'
    test_index_path = '/home/liuzhenyu/FRY/graph_clip/data/' + dataset + '/test' + str(label_rate) + '.txt'
    label_path = '/home/liuzhenyu/FRY/graph_clip/data/' + dataset + '/' + dataset + '.label'
    fgraph_path = '/home/liuzhenyu/FRY/graph_clip/data/' + dataset + '/knn/c' + str(topk) + '.txt'
    feature_path = '/home/liuzhenyu/FRY/graph_clip/data/' + dataset + '/' + dataset + '.feature'
    
    ## process features
    feature = np.loadtxt(feature_path, dtype = float)
    n = feature.shape[0]
    features = sp.csr_matrix(feature, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    ## process train/test index
    test_index = np.loadtxt(test_index_path, dtype = int)
    test_index = test_index.tolist()
    train_index = np.loadtxt(train_index_path, dtype = int)
    train_index = train_index.tolist()
    train_index = torch.LongTensor(train_index)
    test_index = torch.LongTensor(test_index)
    ## process labels
    labels = np.loadtxt(label_path, dtype = int)
    labels = torch.LongTensor(np.array(labels))
    ## process adj matrixs
    sadj, dsadj = process_graph(graph_path, n)
    fadj, dfadj = process_graph(fgraph_path, n)
    print('data loaded!')
    return features, train_index, test_index, labels, sadj, fadj, dsadj, dfadj

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def process_graph(adj_path, n):
    edges = np.genfromtxt(adj_path, dtype=np.int32)
    edges = np.array(list(edges), dtype=np.int32).reshape(edges.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    nadj = normalize(adj+sp.eye(adj.shape[0]))
    dnadj = nadj.todense()
    dnadj = torch.Tensor(dnadj)
    return sparse_mx_to_torch_sparse_tensor(nadj), dnadj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
