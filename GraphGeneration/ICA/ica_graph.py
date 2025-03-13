import logging
from typing import Union

import numpy as np
import networkx as nx
from sklearn.decomposition import FastICA
from sklearn.neighbors import kneighbors_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_ica_graph(representation: np.ndarray,
                  k: int,
                  whiten:Union[str, bool],
                  functional: str,
                  dim: int,
                  rnd_state: int) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph using ICA-reduced representation.

    Parameters
    ----------
    representation : np.ndarray
        The input data representation (high-dimensional).
    k : int
        Number of nearest neighbors for the graph.
    whiten : str or bool
        Whether to apply whitening in the ICA transformation.
    functional: str
        The functional form of the G function used in the approximation to neg-entropy. Could be either ‘logcosh’, ‘exp’, or ‘cube’. 
    dim : int
        Desired number of dimensions for ICA reduction.
    random_state :
        Seed for reproducibility (since ICA is stochastic).     

    Returns
    -------
    nx.DiGraph
        A directed graph representing the k-nearest neighbors for each instance in the new reduced representation.

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

    # Defines the ICA settings
    ica = FastICA(n_components=dim, whiten=whiten, fun=functional, random_state=rnd_state)
    
    # Fits it to representation
    ica.fit(representation)

    # Transforms the data
    representation_ica = ica.transform(representation)

    # Constructs k-nearest neighbors graph (returns a sparse matrix)
    adj_matrix_ica = kneighbors_graph(representation_ica, k, mode='connectivity')
    # Converts adjacency matrix to a directed NetworkX graph
    ica_graph = nx.from_scipy_sparse_array(adj_matrix_ica, create_using=nx.DiGraph)

    return ica_graph
