import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph

from utils import load_representation, save_edges


def get_consistency_graph(semantic_representation: np.ndarray, geospatial_representation: np.ndarray,
                              temporal_representation: np.ndarray, k_s: int, k_g: int, k_t: int) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph summing up the three different components of each instance.

    Parameters
    ----------
    semantic_representation : np.ndarray
        The input data semantic representation.
    geospatial_representation : np.ndarray
        The input data geospatial_representation.
    temporal_representation : np.ndarray
        The input data temporal representation.    
    k_s : int
        Number of nearest neighbors to consider while creating the semantic graph of nearest neighbors.
    k_g : int
        Number of nearest neighbors to consider while creating the geospatial graph of nearest neighbors.
    k_t : int
        Number of nearest neighbors to consider while creating the temporal graph of nearest neighbors.
    
    Returns
    -------
    nx.DiGraph
        A directed graph representing the consistency graph.
    """
    
    # obtains the consistency graph components: the semantic, geospatial and temporal neighbors of each instance
    adj_matrix_semantic = kneighbors_graph(semantic_representation, k_s, mode='connectivity',metric="cosine")
    adj_matrix_geospatial = kneighbors_graph(geospatial_representation, k_g, mode='connectivity',metric="haversine")
    adj_matrix_temporal = kneighbors_graph(temporal_representation, k_t, mode='connectivity',metric="euclidean")
    # sums up the k-neighbors graphs' adjacency matrices to define the consistency graph's adjacency matrix
    adj_matrix_consistency = adj_matrix_semantic.toarray() + adj_matrix_geospatial.toarray() + adj_matrix_temporal.toarray()
    # The instances are neighbors in consistency graph if they are neighbors in two or three componentes
    adj_matrix_consistency = np.where(adj_matrix_consistency < 2, 0, 1) 
    # obtains the consistency graph (as a nx.digraph) from its adjacency
    consistency_graph =  nx.DiGraph(adj_matrix_consistency) 
    # returns the consistency graph as a networkx object
    return consistency_graph


if __name__ == "__main__": 
    # parsing the arguments that'll be used to create consistency matrices
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
    output_file = f"{output_dir}/consistency_edges_{dataset_name}_{k_s}_{k_g}_{k_t}.pkl"
    save_edges(consistency_graph, output_file)
