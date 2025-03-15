import logging
from typing import Union

import numpy as np
import scipy
from umap import UMAP
import networkx as nx
from sklearn.neighbors import kneighbors_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_umap_graph(events: np.ndarray,
                   k_umap: int,
                   n_neighbors: int,
                   min_dist: float,
                   initialization: str,
                   metric : str,
                   n_components: int,
                   random_state: int,
                   precomputed_knn: Union[np.ndarray, scipy.sparse.spmatrix] = None) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph using UMAP-reduced representation.

    Parameters
    ----------
    representation : np.ndarray
        The input data representation (high-dimensional).
    k : int
        Number of nearest neighbors for the graph.
    n_neighbors: int
        This parameter controls how UMAP balances local versus global structure in the data.
    min_dist: float
        The min_dist parameter controls how tightly UMAP is allowed to pack points together.
    initialization: str
        How to initialize the low dimensional embedding.
    dim : int
        Desired number of dimensions for UMAP reduction.
    rnd_state: int
        Seed for reproducibility (since UMAP is stochastic)
    
    Returns
    -------
    nx.DiGraph
        A directed graph representing the k-nearest neighbors for each instance in the new reduced representation.

    Raises
    ------
    ValueError
        If `k`, `n_neighbors` `dim` are not positive integers, or if `dim` is greater than the number of features in `representation`.
    """
    # Validates inputs
    if k_umap <= 0 or n_neighbors<=1 or n_components <= 0:
        logging.error("Non positive value for k or dimension")
        raise ValueError("k-value, n_neighbors and dimension must be positive integers.")

    if n_components > events.shape[1]:
        logging.error("Inconsistent dimension")
        raise ValueError(f"n_components ({n_components}) cannot be greater than the number of features in representation ({events.shape[1]}).")

    if precomputed_knn is not None:
        umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, init=initialization, metric=metric, n_components = n_components,
                    random_state=random_state, precomputed_knn=precomputed_knn)
    else: 
        umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, init=initialization, metric=metric, n_components = n_components,
                    random_state=random_state)
        
    # Fits it to the representation 
    umap.fit(events)

    # Gets the new transformed space
    representation_UMAP = umap.transform(events)
    
    # Constructs k-nearest neighbors graph (returns a sparse matrix)
    adj_matrix_umap = kneighbors_graph(representation_UMAP, k_umap, mode='connectivity')
    # Converts adjacency matrix to a directed NetworkX graph
    umap_graph =  nx.from_scipy_sparse_array(adj_matrix_umap, create_using=nx.DiGraph)

    return umap_graph