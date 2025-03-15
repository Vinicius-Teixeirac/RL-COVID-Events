import logging

import numpy as np
import networkx as nx
# from sklearn.manifold import TSNE
from openTSNE import PCA, TSNE
from sklearn.neighbors import kneighbors_graph

from GraphGeneration import get_tsne_graph


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_tsne_graph(events: np.ndarray,
                   k_tsne: int, 
                   perplexity: int,
                   initialization: str,
                   metric: str,
                   apply_pca: bool,
                   n_components: int,
                   random_state: int,
                   n_jobs: int = 10) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph using t-SNE-reduced representation.

    Parameters
    ----------
    events : np.ndarray
        The input data representation (high-dimensional).
    k_tsne : int
        Number of nearest neighbors for the graph.
    perplexity : int
        Perplexity. The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms.
    initialization : str
        The initial point positions to be used in the embedding space.
    metric : str

    apply_pca : bool

    n_components : int
        Desired number of dimensions for t-SNE reduction.
    random_state : int
        Seed for reproducibility (since t-SNE is stochastic).
    n_jobs : int
        Number of cpu cores to be used in an execution.
    
    Returns
    -------
    nx.DiGraph
        A directed graph representing the k-nearest neighbors for each instance in the new reduced representation.

    Raises
    ------
    ValueError
        If `k_tsne` or `n_components` are not positive integers, or if `n_components` is greater than the number of features in input data,
        and if 'perplexity` was bad choosen.
    """
    # Validates inputs
    if k_tsne <= 0 or n_components <= 0:
        logging.error("Non positive value for k or dimension")
        raise ValueError("Both k_tsne and n_components must be positive integers.")

    if n_components > events.shape[1]:
        logging.error("Inconsistent dimension")
        raise ValueError(f"n_components ({n_components}) cannot be greater than the number of features in representation ({events.shape[1]}).")
    
    # Validate perplexity (must be < number of points)
    if perplexity >= events.shape[0]:
        logging.error("Inconsistent perplexity")
        raise ValueError(f"Perplexity ({perplexity}) must be smaller than the number of samples ({events.shape[0]}).")
    
    if apply_pca:
        events = PCA(n_components=50).fit_transform(events)

    # Defines the t-SNE settings 
    tsne = TSNE(perplexity=perplexity, initialization=initialization,
                n_components=n_components, random_state=random_state,
                n_jobs=n_jobs, metric=metric) 
    
    # Gets the new transformed space
    representation_tSNE = tsne.fit(events) 

    # Constructs k-nearest neighbors graph (returns a sparse matrix)
    adj_matrix_tsne = kneighbors_graph(representation_tSNE, k_tsne, mode='connectivity')
    # Converts adjacency matrix to a directed NetworkX graph
    tsne_graph = nx.from_scipy_sparse_array(adj_matrix_tsne, create_using=nx.DiGraph)

    return tsne_graph