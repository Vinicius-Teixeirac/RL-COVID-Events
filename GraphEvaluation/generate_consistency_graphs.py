import argparse
import warnings
from pathlib import Path
import logging

import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph

from utils import load_representation, save_edges

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_consistency_graph(semantic_representation: np.ndarray,
                          geospatial_representation: np.ndarray,
                          temporal_representation: np.ndarray,
                          k_s: int,
                          k_g: int,
                          k_t: int,
                          threshold: int = 2) -> nx.DiGraph:
    """
    Generates a directed consistency graph from three representations (semantic, geospatial, temporal).

    Instances are considered neighbors if they appear as neighbors in at least 'threshold' 
    of the three k-nearest neighbor graphs.

    Parameters
    ----------
    semantic_representation : np.ndarray of shape (n_samples, n_semantic_features)
        The semantic representation of the dataset.
    geospatial_representation : np.ndarray of shape (n_samples, n_geospatial_features)
        The geospatial representation of the dataset.
    temporal_representation : np.ndarray of shape (n_samples, n_temporal_features)
        The temporal representation of the dataset.
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
    if semantic_representation.shape[0] != geospatial_representation.shape[0] or \
       semantic_representation.shape[0] != temporal_representation.shape[0]:
        logging.error("Inconsistent representation shapes")
        raise ValueError("All representations must have the same number of samples (rows).")
    
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
    adj_matrix_semantic = kneighbors_graph(semantic_representation, k_s, mode='connectivity', metric="cosine").toarray()
    adj_matrix_geospatial = kneighbors_graph(geospatial_representation, k_g, mode='connectivity', metric="haversine").toarray()
    adj_matrix_temporal = kneighbors_graph(temporal_representation, k_t, mode='connectivity', metric="euclidean").toarray()

    # Sums adjacency matrices and threshold
    adj_matrix_consistency = adj_matrix_semantic + adj_matrix_geospatial + adj_matrix_temporal
    adj_matrix_consistency = np.where(adj_matrix_consistency < threshold, 0, 1)

    return nx.DiGraph(adj_matrix_consistency)



if __name__ == "__main__": 
    # the arguments that'll be used to create consistency matrices
    parser = argparse.ArgumentParser(description ='Generate the reference graphs')
    parser.add_argument('--hyperparameters',  type=int, nargs=3, 
						help='Number of neighbors for semantic, geographical and temporal neighbors graph: k_s, k_g, k_t')
    parser.add_argument('--dataset_path', type=str, help='The path for the current dataset file')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/Consistency", help="Directory to save the output.")
    args = parser.parse_args()

    # getting the dataset specifications
    dataset_path = args.dataset_path
    dataset_name = Path(dataset_path).stem
    output_dir = args.output_dir

    # acquiring from arguments hyperparameters
    k_s, k_g, k_t = args.hyperparameters
    
    # loading the representations to obtaining the consistency graph from them
    representations = load_representation(dataset_name)
    semantic, geospatial, temporal = representations['semantic'], representations['geospatial'], representations['temporal']
    consistency_graph = get_consistency_graph(semantic, geospatial, temporal, k_s, k_g, k_t) 

    # defining the file's name and saving the representation   
    output_file = Path(output_dir) / dataset_name / f"consistency_edges_{k_s}_{k_g}_{k_t}.pkl"
    save_edges(consistency_graph, output_file)
