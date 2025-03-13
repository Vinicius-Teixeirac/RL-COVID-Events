import logging

import numpy as np
import networkx as nx
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import kneighbors_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_lle_graph(representation: np.ndarray,
                  k: int,
                  n_neighbors: int,
                  method: str,
                  dim: int,
                  rnd_state: int) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph using Locally Linear Embedding-reduced representation.

    Parameters
    ----------
    representation : np.ndarray
        The input data representation (high-dimensional).
    k : int
        Number of nearest neighbors for the graph.
    n_neighbors : int
        Number of neighbors to consider for each point.
    method: str
        The algorithm that implements LLE.
    dim : int
        Desired number of dimensions for LLE reduction.
    random_state : int
        Seed for reproducibility (since LLE is stochastic).

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
    
    lle = LocallyLinearEmbedding(n_components=dim, n_neighbors=n_neighbors, method=method, random_state=rnd_state)

    lle.fit(representation)

    representation_lle = lle.transform(representation)

    # Constructs k-nearest neighbors graph (returns a sparse matrix)
    adj_matrix_lle = kneighbors_graph(representation_lle, k, mode='connectivity')
    # Converts adjacency matrix to a directed NetworkX graph
    lle_graph = nx.from_scipy_sparse_array(adj_matrix_lle, create_using=nx.DiGraph)

    return lle_graph

