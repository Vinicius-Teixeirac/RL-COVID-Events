import logging

import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_pca_graph(events: np.ndarray,
                  k_pca: int,
                  whiten: bool,
                  n_components: int) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph using PCA-reduced representation.

    Parameters
    ----------
    events : np.ndarray
        The input data events (high-dimensional).
    k_pca : int
        Number of nearest neighbors for the PCA-reducted graph.
    whiten: bool
        When True, vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure 
        uncorrelated outputs with unit component-wise variances.
    n_components : int
        Desired number of dimensions for PCA reduction.

    Returns
    -------
    nx.DiGraph
        A directed graph representing the k-nearest neighbors for each instance in the new reduced representation.

    Raises
    ------
    ValueError
        If `k_pca` or `n_components` are not positive integers, or if `n_components` is greater than the number of features in input data.
    """
    # Validates inputs
    if k_pca <= 0 or n_components <= 0:
        logging.error("Non positive value for k or dimension")
        raise ValueError("Both k-value and dimension must be positive integers.")

    if n_components > events.shape[1]:
        logging.error("Inconsistent dimension")
        raise ValueError(f"n_components ({n_components}) cannot be greater than the number of features in representation ({events.shape[1]}).")

    # Defines the PCA settings
    pca = PCA(whiten=whiten, n_components=n_components)

    # Fits it to the data
    pca.fit(events)

    # Transforms the data to get PCA-reducted representation
    representation_pca = pca.transform(events)

    # Constructs k-nearest neighbors graph (returns a sparse matrix)
    adj_matrix_pca = kneighbors_graph(representation_pca, k_pca, mode='connectivity')
    # Converts adjacency matrix to a directed NetworkX graph
    pca_graph = nx.from_scipy_sparse_array(adj_matrix_pca, create_using=nx.DiGraph)

    return pca_graph