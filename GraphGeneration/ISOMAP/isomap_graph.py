import logging

import numpy as np
import networkx as nx
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_isomap_graph(representation: np.ndarray,
                  k: int,
                  n_neighbors: int,
                  dim: int) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph using Isomap-reduced representation.

    Parameters
    ----------
    representation : np.ndarray
        The input data representation (high-dimensional).
    k : int
        Number of nearest neighbors for the graph.
    n_neighbors : int
        Number of neighbors to consider for each point.
    dim : int
        Desired number of dimensions for isomap reduction.
    random_state : int
        Seed for reproducibility (since isomap is stochastic).

    Returns
    -------
    nx.DiGraph
        A directed graph representing the k-nearest neighbors for each instance in the new reduced representation.

    Raises
    ------
    ValueError
        If `k` or `dim` are not positive integers, or if `dim` is greater than the number of features in `representation`.
    """

    # Validates inputs
    if k <= 0 or dim <= 0:
        logging.error("Non positive value for k or dimension")
        raise ValueError("Both k-value and dimension must be positive integers.")

    if dim > representation.shape[1]:
        logging.error("Inconsistent dimension")
        raise ValueError(f"dim ({dim}) cannot be greater than the number of features in representation ({representation.shape[1]}).")
    
    isomap = Isomap(n_components=dim, n_neighbors=n_neighbors)

    isomap.fit(representation)

    representation_isomap = isomap.transform(representation)

    # Constructs k-nearest neighbors graph (returns a sparse matrix)
    adj_matrix_isomap = kneighbors_graph(representation_isomap, k, mode='connectivity')
    # Converts adjacency matrix to a directed NetworkX graph
    isomap_graph = nx.from_scipy_sparse_array(adj_matrix_isomap, create_using=nx.DiGraph)

    return isomap_graph

