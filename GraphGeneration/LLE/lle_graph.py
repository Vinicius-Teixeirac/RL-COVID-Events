import logging

import numpy as np
import networkx as nx
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import kneighbors_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_lle_graph(events: np.ndarray,
                  k_lle: int,
                  n_neighbors: int,
                  method: str,
                  n_components: int,
                  random_state: int) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph using Locally Linear Embedding-reduced representation.

    Parameters
    ----------
    events : np.ndarray
        The input data events (high-dimensional).
    k_lle : int
        Number of nearest neighbors for the LLE-reducted graph.
    n_neighbors : int
        Number of neighbors of each point for LLE dimensionality reduction.
    method: str
        The algorithm that implements LLE.
    n_components : int
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
        If `k_lle` or `n_components` are not positive integers, or if `n_components` is greater than the number of features in input data.
    """

    # Validates inputs
    if k_lle <= 0 or n_components <= 0:
        logging.error("Non positive value for k or dimension")
        raise ValueError("Both k_lle and n_components must be positive integers.")

    if n_components > events.shape[1]:
        logging.error("Inconsistent dimension")
        raise ValueError(f"n_components ({n_components}) cannot be greater than the number of features in input data ({events.shape[1]}).")
    
    # Defines the LLE settings
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, method=method, n_components=n_components, random_state=random_state)

    # Fits it to data
    lle.fit(events)

    # Transforms the data to get LLE-reducted representation
    representation_lle = lle.transform(events)

    # Constructs k-nearest neighbors graph (returns a sparse matrix)
    adj_matrix_lle = kneighbors_graph(representation_lle, k_lle, mode='connectivity')
    # Converts adjacency matrix to a directed NetworkX graph
    lle_graph = nx.from_scipy_sparse_array(adj_matrix_lle, create_using=nx.DiGraph)

    return lle_graph

