import logging

import numpy as np
import networkx as nx
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_isomap_graph(events: np.ndarray,
                  k_isomap: int,
                  n_neighbors: int,
                  n_components: int) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph using Isomap-reduced representation.

    Parameters
    ----------
    events : np.ndarray
        The input data events (high-dimensional).
    k_isomap : int
        Number of nearest neighbors for the Isoamp-reduced graph.
    n_neighbors : int
        Number of neighbors of each point for Isomap dimensionality reduction.
    n_components : int
        Desired number of dimensions for isomap reduction.

    Returns
    -------
    nx.DiGraph
        A directed graph representing the k-nearest neighbors for each instance in the new reduced representation.

    Raises
    ------
    ValueError
        If `k_isomap` or `n_components` are not positive integers, or if `n_components` is greater than the number of features in input data.
    """

    # Validates inputs
    if k_isomap <= 0 or n_components <= 0:
        logging.error("Non positive value for k_isomap or n_components")
        raise ValueError("Both k_isomap and n_components must be positive integers.")

    if n_components > events.shape[1]:
        logging.error("Inconsistent dimension")
        raise ValueError(f"n_components ({n_components}) cannot be greater than the number of features in input data ({events.shape[1]}).")
    
    # Defines the Isomap settings
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components, n_jobs=-1)

    # Fits it to data
    isomap.fit(events)

    # Transforms the data to get Isomap-reducted representation
    representation_isomap = isomap.transform(events)

    # Constructs k-nearest neighbors graph (returns a sparse matrix)
    adj_matrix_isomap = kneighbors_graph(representation_isomap, k_isomap, mode='connectivity')
    # Converts adjacency matrix to a directed NetworkX graph
    isomap_graph = nx.from_scipy_sparse_array(adj_matrix_isomap, create_using=nx.DiGraph)

    return isomap_graph