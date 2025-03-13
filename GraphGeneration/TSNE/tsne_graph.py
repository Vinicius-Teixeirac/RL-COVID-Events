import logging

import numpy as np
import networkx as nx
# from sklearn.manifold import TSNE
from openTSNE import TSNE
from sklearn.neighbors import kneighbors_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_tsne_graph(representation: np.ndarray,
                   k: int, 
                   ppxty: int,
                   initialization: str,
                   dim: int,
                   rnd_state: int,
                   n_jobs: int = 10) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph using t-SNE-reduced representation.

    Parameters
    ----------
    representation : np.ndarray
        The input data representation (high-dimensional).
    k : int
        Number of nearest neighbors for the graph.
    ppxty: int
        Perplexity. The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms.
    initialization: str
        The initial point positions to be used in the embedding space.
    dim : int
        Desired number of dimensions for t-SNE reduction.
    rnd_state: int
        Seed for reproducibility (since t-SNE is stochastic).
    n_jobs:
        Number of cpu cores to be used in an execution.
    
    Returns
    -------
    nx.DiGraph
        A directed graph representing the k-nearest neighbors for each instance in the new reduced representation.

    Raises
    ------
    ValueError
        If `k` or `dim` are not positive integers, or if `dim` is greater than the number of features in `representation`, and if
        `perplexity` was bad choosen.
    """
    # Validates inputs
    if k <= 0 or dim <= 0:
        logging.error("Non positive value for k or dimension")
        raise ValueError("Both k-value and dimension must be positive integers.")

    if dim > representation.shape[1]:
        logging.error("Inconsistent dimension")
        raise ValueError(f"dim ({dim}) cannot be greater than the number of features in representation ({representation.shape[1]}).")
    
    # Validate perplexity (must be < number of points)
    if ppxty >= representation.shape[0]:
        logging.error("Inconsistent perplexity")
        raise ValueError(f"Perplexity ({ppxty}) must be smaller than the number of samples ({representation.shape[0]}).")

    # Defines the t-SNE settings 
    tsne = TSNE(n_components=dim, perplexity=ppxty, random_state=rnd_state, metric='euclidean', initialization=initialization, n_jobs=n_jobs) 
    # Gets the new transformed space
    representation_tSNE = tsne.fit(representation) 

    # Constructs k-nearest neighbors graph (returns a sparse matrix)
    adj_matrix_tsne = kneighbors_graph(representation_tSNE, k, mode='connectivity')
    # Converts adjacency matrix to a directed NetworkX graph
    tsne_graph = nx.from_scipy_sparse_array(adj_matrix_tsne, create_using=nx.DiGraph)

    return tsne_graph