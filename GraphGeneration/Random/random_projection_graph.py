import logging

import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.random_projection import GaussianRandomProjection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_random_projection_graph(representation: np.ndarray,
                                k: int,
                                dim: int,
                                rnd_state: int) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph using a random projection-reduced representation.

    Parameters
    ----------
    representation : np.ndarray
        The input data representation (high-dimensional).
    k : int
        Number of nearest neighbors for the graph.
    dim : int
        Desired number of dimensions for random projection reduction.
    rnd_state : int
        Seed for reproducibility.

    Returns
    -------
    nx.DiGraph
        A directed graph representing the k-nearest neighbors for each instance in the reduced representation.

    Raises
    ------
    ValueError
        If `k` or `dim` are not positive integers, or if `dim` is greater than the number of features in `representation`.
    """
    
    # Validate inputs
    if k <= 0 or dim <= 0:
        logging.error("Non positive value for k or dimension")
        raise ValueError("Both k-value and dimension must be positive integers.")

    if dim > representation.shape[1]:
        logging.error("Inconsistent dimension")
        raise ValueError(f"dim ({dim}) cannot be greater than the number of features in representation ({representation.shape[1]}).")

    # Apply Gaussian Random Projection for dimensionality reduction
    rp = GaussianRandomProjection(n_components=dim, random_state=rnd_state)
    representation_rp = rp.fit_transform(representation)
    
    # Construct k-nearest neighbors graph (returns a sparse matrix)
    adj_matrix_rp = kneighbors_graph(representation_rp, k, mode='connectivity')
    
    # Convert the sparse adjacency matrix to a directed NetworkX graph
    rp_graph = nx.from_scipy_sparse_array(adj_matrix_rp, create_using=nx.DiGraph)
    
    return rp_graph
