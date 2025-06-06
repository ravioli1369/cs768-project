import numpy as np
import scipy.sparse as sp
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from dataloader import SparseGraph


def temporal_weighted_adjacency(edge_file, decay_factor=0.9):
    """Create time-decay weighted adjacency matrix from temporal edge list"""
    df = pd.read_csv(edge_file, skiprows=1, header=None)
    src_nodes = df[0].values
    dst_nodes = df[1].values
    timestamps = df[2].values
    
    min_time = timestamps.min()
    max_time = timestamps.max()
    norm_times = (timestamps - min_time) / (max_time - min_time + 1e-8)
    
    weights = np.power(decay_factor, 1 - norm_times)
    
    unique_nodes = np.unique(np.concatenate([src_nodes, dst_nodes]))
    node_id_map = {node: i for i, node in enumerate(unique_nodes)}
    
    rows = [node_id_map[node] for node in src_nodes]
    cols = [node_id_map[node] for node in dst_nodes]
    adj = sp.csr_matrix((weights, (rows, cols)), 
                       shape=(len(unique_nodes), len(unique_nodes)))
    
    return adj, node_id_map

def load_temporal_data(edge_file, node_feat_file, decay=0.9):
    """Load temporal data and convert to sparse graph format"""

    adj_matrix, node_map = temporal_weighted_adjacency(edge_file, decay)
    node_feats = np.load(node_feat_file)

    sorted_nodes = sorted(node_map.items(), key=lambda x: x[1])
    aligned_feats = node_feats[[int(k) for k, v in sorted_nodes]].reshape(len(sorted_nodes), -1)
    
    
    return SparseGraph(
        adj_matrix=adj_matrix,
        attr_matrix=sp.csr_matrix(aligned_feats),
        labels=None,
        node_names=np.array([str(k) for k, v in sorted_nodes]),
        class_names=None
    )

def load_npz_to_sparse_graph(file_name):
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix(
            (
                loader["adj_data"].ravel(),
                loader["adj_indices"].ravel(),
                loader["adj_indptr"].ravel()
            ),
            shape=tuple(loader["adj_shape"])
        )
        

        if "attr_data" in loader and "attr_indices" in loader and "attr_indptr" in loader:
            try:
                attr_matrix = sp.csr_matrix(
                    (
                        loader["attr_data"].ravel(),
                        loader["attr_indices"].ravel(),
                        loader["attr_indptr"].ravel()
                    ),
                    shape=tuple(loader["attr_shape"])
                )
            except ValueError:
                if sp.issparse(loader["attr_data"]):
                    attr_matrix = loader["attr_data"]
                else:
                    attr_matrix = sp.csr_matrix(loader["attr_data"])
        else:
            # No sparse components found, use attr_matrix directly
            attr_matrix = sp.csr_matrix(loader["attr_matrix"])
        
        data = SparseGraph(
            adj_matrix=adj_matrix,
            attr_matrix=attr_matrix,
            labels=loader["labels"],
            node_names=loader["node_names"],
            class_names=loader["class_names"]
        )
    return data

def is_binary_bag_of_words(features):
    features_coo = features.tocoo()
    return all(
        single_entry == 1.0
        for _, _, single_entry in zip(
            features_coo.row, features_coo.col, features_coo.data
        )
    )


def to_binary_bag_of_words(features):
    """Converts TF/IDF features to binary bag-of-words features."""
    features_copy = features.tocsr()
    features_copy.data[:] = 1.0
    return features_copy


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    adj = normalize(adj + sp.eye(adj.shape[0]))
    return adj


def eliminate_self_loops_adj(A):
    """Remove self-loops from the adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def largest_connected_components(sparse_graph, n_components=1):
    """
    Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][
                         :n_components
                         ]  # reverse order to sort descending
    nodes_to_keep = [
        idx
        for (idx, component) in enumerate(component_indices)
        if component in components_to_keep
    ]
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)


def create_subgraph(
        sparse_graph, _sentinel=None, nodes_to_remove=None, nodes_to_keep=None
):
    """
    Create a graph with the specified subset of nodes.

    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named arguments to this function.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    _sentinel : None
        Internal, to prevent passing positional arguments. Do not use.
    nodes_to_remove : array-like of int
        Indices of nodes that have to removed.
    nodes_to_keep : array-like of int
        Indices of nodes that have to be kept.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph with specified nodes removed.

    """
    if _sentinel is not None:
        raise ValueError(
            "Only call `create_subgraph` with named arguments',"
            " (nodes_to_remove=...) or (nodes_to_keep=...)"
        )
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError(
            "Only one of nodes_to_remove or nodes_to_keep must be provided."
        )
    elif nodes_to_remove is not None:
        nodes_to_keep = [
            i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove
        ]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    sparse_graph.adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is not None:
        sparse_graph.attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.labels is not None:
        sparse_graph.labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is not None:
        sparse_graph.node_names = sparse_graph.node_names[nodes_to_keep]
    return sparse_graph


def binarize_labels(labels, sparse_output=False, return_classes=False):
    """
    Convert labels vector to a binary label matrix.

    In the default single-label case, labels look like
    labels = [y1, y2, y3, ...].
    Also supports the multi-label format.
    In this case, labels should look something like
    labels = [[y11, y12], [y21, y22, y23], [y31], ...].

    Parameters
    ----------
    labels : array-like, shape [num_samples]
        Array of node labels in categorical single- or multi-label format.
    sparse_output : bool, default False
        Whether return the label_matrix in CSR format.
    return_classes : bool, default False
        Whether return the classes corresponding to the columns of the label matrix.

    Returns
    -------
    label_matrix : np.ndarray or sp.csr_matrix, shape [num_samples, num_classes]
        Binary matrix of class labels.
        num_classes = number of unique values in "labels" array.
        label_matrix[i, k] = 1 <=> node i belongs to class k.
    classes : np.array, shape [num_classes], optional
        Classes that correspond to each column of the label_matrix.

    """
    if hasattr(labels[0], "__iter__"):
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    label_matrix = binarizer.fit_transform(labels).astype(np.float32)
    return (label_matrix, binarizer.classes_) if return_classes else label_matrix


def remove_underrepresented_classes(
        g, train_examples_per_class, val_examples_per_class
):
    """
    Remove nodes from graph that correspond to a class of which there are less than
    num_classes * train_examples_per_class + num_classes * val_examples_per_class nodes.

    Those classes would otherwise break the training procedure.
    """
    min_examples_per_class = train_examples_per_class + val_examples_per_class
    examples_counter = Counter(g.labels)
    keep_classes = set(
        class_
        for class_, count in examples_counter.items()
        if count > min_examples_per_class
    )
    keep_indices = [i for i in range(len(g.labels)) if g.labels[i] in keep_classes]

    return create_subgraph(g, nodes_to_keep=keep_indices)
