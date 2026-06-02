import logging
import warnings

import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_consistency_graph(semantic_features: np.ndarray,
                          geospatial_features: np.ndarray,
                          temporal_features: np.ndarray,
                          k_s: int,
                          k_g: int,
                          k_t: int,
                          threshold: int = 2) -> nx.DiGraph:
    """
    Generates a directed consistency graph from three set of features (semantic, geospatial, temporal).

    Instances are considered neighbors if they appear as neighbors in at least 'threshold' 
    of the three k-nearest neighbor graphs.

    Parameters
    ----------
    semantic_features : np.ndarray of shape (n_samples, n_semantic_features)
        The semantic features of the dataset.
    geospatial_features : np.ndarray of shape (n_samples, n_geospatial_features)
        The geospatial features of the dataset.
    temporal_features : np.ndarray of shape (n_samples, n_temporal_features)
        The temporal features of the dataset.
    k_s : int
        Number of neighbors for the semantic k-NN graph.
    k_g : int
        Number of neighbors for the geospatial k-NN graph.
    k_t : int
        Number of neighbors for the temporal k-NN graph.
    threshold : int, optional
        The minimum number of graphs in which two instances must be neighbors 
        to be considered neighbors in the final consistency graph. Defaults to 2.

    Returns
    -------
    nx.DiGraph
        A directed graph representing the consistency graph.

    Raises
    ------
    ValueError
        If the input arrays do not have the same number of samples, or if any k-value is non-positive.

    Warns
    -----
    UserWarning
        If `threshold` is greater than 3, as there are only three components.
    """
    # Validates shapes 
    if semantic_features.shape[0] != geospatial_features.shape[0] or \
       semantic_features.shape[0] != temporal_features.shape[0]:
        logging.error("Inconsistent features shapes")
        raise ValueError("All sets of features must have the same number of samples (rows).")
    
    # Validates k's
    if k_s <= 0 or k_g <= 0 or k_t <= 0:
        logging.error("Inconsistent values for k")
        raise ValueError("k-values must be positive integers.")

    # Checks threshold
    if threshold > 3:
        warnings.warn(
            f"Threshold ({threshold}) is greater than the number of available graphs (3). "
            "This may result in an empty graph.", 
            UserWarning
        )
    
    # Builds k-NN adjacency matrices
    adj_matrix_semantic = kneighbors_graph(semantic_features, k_s, mode='connectivity', metric="cosine").toarray()
    adj_matrix_geospatial = kneighbors_graph(np.radians(geospatial_features), k_g, mode='connectivity', metric="haversine").toarray()
    adj_matrix_temporal = kneighbors_graph(temporal_features, k_t, mode='connectivity', metric="euclidean").toarray()

    # Sums adjacency matrices and threshold
    adj_matrix_consistency = adj_matrix_semantic + adj_matrix_geospatial + adj_matrix_temporal
    adj_matrix_consistency = np.where(adj_matrix_consistency < threshold, 0, 1)

    return nx.DiGraph(adj_matrix_consistency)