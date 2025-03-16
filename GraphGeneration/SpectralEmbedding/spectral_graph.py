import logging

import numpy as np
import networkx as nx
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import kneighbors_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_spectral_graph(events: np.ndarray,
                  k_spectral: int,
                  n_neighbors: int,
                  n_components: int,
                  random_state: int) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph using Laplacian Eigenmaps-reduced representation.

    Parameters
    ----------
    events : np.ndarray
        The input data representation (high-dimensional).
    k_spectral : int
        Number of nearest neighbors for the graph.
    n_neighbors : int
        Number of neighbors to consider for each point.
    n_components : int
        Desired number of dimensions for Laplacian Eigenmaps reduction.
    random_state : int
        Seed for reproducibility (since Laplacian Eigenmaps is stochastic).

    Returns
    -------
    nx.DiGraph
        A directed graph representing the k-nearest neighbors for each instance in the new reduced representation.

    Raises
    ------
    ValueError
        If `k_spectral` or `n_components` are not positive integers, or if `n_components` is greater than the number of features in input data.
    """

    # Validates inputs
    if k_spectral <= 0 or n_components <= 0:
        logging.error("Non positive value for k or dimension")
        raise ValueError("Both k_spectral and n_components must be positive integers.")

    if n_components > events.shape[1]:
        logging.error("Inconsistent dimension")
        raise ValueError(f"n_components ({n_components}) cannot be greater than the number of features in representation ({events.shape[1]}).")
    
    # Defines the Laplacian Eigenmaps settings
    spectral = SpectralEmbedding(n_neighbors=n_neighbors, n_components=n_components, random_state=random_state)

    # Transforms the data to get Laplacian Eigenmap-reducted representation
    representation_spectral = spectral.fit_transform(events)

    # Constructs k-nearest neighbors graph (returns a sparse matrix)
    adj_matrix_spectral = kneighbors_graph(representation_spectral, k_spectral, mode='connectivity')
    # Converts adjacency matrix to a directed NetworkX graph
    spectral_graph = nx.from_scipy_sparse_array(adj_matrix_spectral, create_using=nx.DiGraph)

    return spectral_graph

