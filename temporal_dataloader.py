import numpy as np
import scipy.sparse as sp
import torch
import dgl
import os
import pandas as pd
from temporal_data_preprocess import (
    normalize_adj, temporal_weighted_adjacency, load_temporal_data, load_npz_to_sparse_graph, binarize_labels
)
from data_preprocess import (
    normalize_adj,
    binarize_labels

)

from dataloader import get_train_val_test_split


def load_cpf_data(dataset, dataset_path, seed, labelrate_train, labelrate_val):
    npz_path = os.path.join(dataset_path, f"{dataset}.npz")
    csv_path = os.path.join(dataset_path, f"{dataset}.csv")
    node_feat_path = os.path.join(dataset_path, f"{dataset}_node.npy")
    
    if os.path.exists(npz_path):
        data = load_npz_to_sparse_graph(npz_path)
    else:
        data = load_temporal_data(csv_path, node_feat_path)
        data = data.standardize()

        np.savez(
            npz_path,
            adj_data=data.adj_matrix.data,
            adj_indices=data.adj_matrix.indices,
            adj_indptr=data.adj_matrix.indptr,
            adj_shape=data.adj_matrix.shape,
            attr_data=data.attr_matrix.data,
            attr_indices=data.attr_matrix.indices,
            attr_indptr=data.attr_matrix.indptr,
            attr_shape=data.attr_matrix.shape,
            labels=np.zeros(data.num_nodes(), dtype=np.int8),
            node_names=data.node_names,
            class_names=np.array(["Temporal_Class"])
        )
        data = load_npz_to_sparse_graph(npz_path)

    data = data.standardize()
    adj, features, labels = data.unpack()
    labels = binarize_labels(labels)
    
    random_state = np.random.RandomState(seed)
    idx_train, idx_val, idx_test = get_train_val_test_split(
        random_state, labels, labelrate_train, labelrate_val
    )
    
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels.argmax(axis=1))
    adj = normalize_adj(adj)
    adj_sp = adj.tocoo()
    g = dgl.graph((adj_sp.row, adj_sp.col))
    g.ndata["feat"] = features
    
    return g, labels, idx_train, idx_val, idx_test

def temporal_to_npz(edge_file, node_feat_file, output_file, decay=0.9):
    """Convert temporal data to NPZ format with time weighting"""
    df = pd.read_csv(edge_file, skiprows=1, header=None)
    node_feats = np.load(node_feat_file)

    adj, node_map = temporal_weighted_adjacency(edge_file, decay)

    num_nodes = adj.shape[0]
    if node_feats.shape[0] < num_nodes:
        zero_pad = np.zeros((num_nodes - node_feats.shape[0], node_feats.shape[1]))
        node_feats = np.vstack([node_feats, zero_pad])

    sorted_indices = [v for k, v in sorted(node_map.items())]
    attr_matrix = node_feats[sorted_indices]
    
    attr_sparse = sp.csr_matrix(attr_matrix)
    
    np.savez(
        output_file,
        adj_data=adj.data,
        adj_indices=adj.indices,
        adj_indptr=adj.indptr,
        adj_shape=adj.shape,
        attr_data=attr_sparse.data,
        attr_indices=attr_sparse.indices,
        attr_indptr=attr_sparse.indptr,
        attr_shape=attr_matrix.shape,
        labels=np.zeros(adj.shape[0], dtype=np.int8),
        node_names=np.array([str(k) for k, v in sorted(node_map.items())]),
        class_names=np.array(["Temporal_Class"])
    )

if __name__ == "__main__":
    temporal_to_npz(
        edge_file= "data/wikipedia/ml_wikipedia.csv",
        node_feat_file= "data/wikipedia/ml_wikipedia_node.npy",
        output_file= "data/wikipedia/ml_wikipedia.npz",
        decay=0.9
    )