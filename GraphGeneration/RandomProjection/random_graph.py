import logging

import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.random_projection import GaussianRandomProjection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_random_projection_graph(events: np.ndarray,
                                k_random: int,
                                n_components: int,
                                random_state: int) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph using a random projection-reduced representation.

    Parameters
    ----------
    events : np.ndarray
        The input data representation (high-dimensional).
    k_random : int
        Number of nearest neighbors for the Random projection-reducted graph..
    n_components : int
        Desired number of dimensions for random projection reduction.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    nx.DiGraph
        A directed graph representing the k-nearest neighbors for each instance in the reduced representation.

    Raises
    ------
    ValueError
        If `k_random` or `n_components` are not positive integers, or if `n_components` is greater than the number of features in input data.
    """
    
    # Validate inputs
    if k_random <= 0 or n_components <= 0:
        logging.error("Non positive value for k or dimension")
        raise ValueError("Both k_random and n_components must be positive integers.")

    if n_components > events.shape[1]:
        logging.error("Inconsistent dimension")
        raise ValueError(f"n_components ({n_components}) cannot be greater than the number of features in representation ({events.shape[1]}).")

    # Defines the Gaussian Random Projection settings
    rp = GaussianRandomProjection(n_components=n_components, random_state=random_state)

    # Fits it to the data
    rp.fit(events)

    # Transforms the data to get Random Projection-reducted representation
    representation_rp = rp.transform(events)
    
    # Construct k-nearest neighbors graph (returns a sparse matrix)
    adj_matrix_rp = kneighbors_graph(representation_rp, k_random, mode='connectivity')
    
    # Convert the sparse adjacency matrix to a directed NetworkX graph
    rp_graph = nx.from_scipy_sparse_array(adj_matrix_rp, create_using=nx.DiGraph)
    
    return rp_graph
