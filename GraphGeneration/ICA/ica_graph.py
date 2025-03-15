import logging
from typing import Union

import numpy as np
import networkx as nx
from sklearn.decomposition import FastICA
from sklearn.neighbors import kneighbors_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_ica_graph(events: np.ndarray,
                  k_ica: int,
                  whiten:Union[str, bool],
                  fun: str,
                  n_components: int,
                  random_state: int) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph using ICA-reduced representation.

    Parameters
    ----------
    events : np.ndarray
        The input data events (high-dimensional).
    k_ica : int
        Number of nearest neighbors for the ICA-reduced graph.
    whiten : str or bool
        Whether to apply whitening in the ICA transformation.
    fun: str
        The functional form of the G function used in the approximation to neg-entropy. Could be either ‘logcosh’, ‘exp’, or ‘cube’. 
    n_components : int
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
        If `k` or `n_components` are not positive integers, or if `n_components` is greater than the number of features in input data.
    """
    # Validate inputs
    if k_ica <= 0 or n_components <= 0:
        logging.error("Non positive value for k or dimension")
        raise ValueError("Both k_ica and n_components must be positive integers.")

    if n_components > events.shape[1]:
        logging.error("Inconsistent dimension")
        raise ValueError(f"n_components ({n_components}) cannot be greater than the number of features in input data ({events.shape[1]}).")

    # Defines the ICA settings
    ica = FastICA(whiten=whiten, fun=fun, n_components=n_components, random_state=random_state)
    
    # Fits it to data
    ica.fit(events)

    # Transforms the data to get ICA-reducted representation
    representation_ica = ica.transform(events)

    # Constructs k-nearest neighbors graph (returns a sparse matrix)
    adj_matrix_ica = kneighbors_graph(representation_ica, k_ica, mode='connectivity')
    # Converts adjacency matrix to a directed NetworkX graph
    ica_graph = nx.from_scipy_sparse_array(adj_matrix_ica, create_using=nx.DiGraph)

    return ica_graph
