import logging

import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_pca_graph(representation: np.ndarray,
                  k: int,
                  whiten: bool,
                  dim: int) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph using PCA-reduced representation.

    Parameters
    ----------
    representation : np.ndarray
        The input data representation (high-dimensional).
    k : int
        Number of nearest neighbors for the graph.
    dim : int
        Desired number of dimensions for PCA reduction.

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

    # Defines the PCA settings
    pca = PCA(n_components=dim, whiten=whiten)
    # Fits it to representation
    pca.fit(representation)
    # Gets the new transformed space
    representation_pca = pca.transform(representation)

    # Constructs k-nearest neighbors graph (returns a sparse matrix)
    adj_matrix_pca = kneighbors_graph(representation_pca, k, mode='connectivity')
    # Converts adjacency matrix to a directed NetworkX graph
    pca_graph = nx.from_scipy_sparse_array(adj_matrix_pca, create_using=nx.DiGraph)

    return pca_graph

def get_pca_reduction(representation: np.ndarray, dimension: int = 50) -> np.ndarray:
    """
    Reduces the dimension of the initial representation, which may suppress irrelevant variance (noise) and lower
    t-SNE computation time.

     Parameters
    ----------
    representation : np.ndarray
        The input data representation (high-dimensional).
    dim : int
        The lower dimension to be given to t-SNE.

    Returns
    -------
    np.ndarray
        The initial representation after PCA be applied.

    Raises
    ------
    ValueError
        If `dim` is not positive integers, or if `dim` is greater than the number of features in `representation`.
    """
    if dimension <= 0:
        logging.error("Non positive value for dimension")
        raise ValueError("Dimension must be positive integers.")

    if dimension > representation.shape[1]:
        logging.error("Inconsistent dimension")
        raise ValueError(f"dimension ({dimension}) cannot be greater than the number of features in representation ({representation.shape[1]}).")

    return PCA(n_components=dimension).fit_transform(representation)