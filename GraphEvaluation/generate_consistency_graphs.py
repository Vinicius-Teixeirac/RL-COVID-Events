import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph

from utils import load_representation, save_edges

def get_consistency_graph(semantic_representation: np.ndarray, geospatial_representation: np.ndarray,
                              temporal_representation: np.ndarray, k_s: int, k_g: int, k_t: int) -> np.ndarray:
    # get each k-neighbors from representations
    adj_matrix_semantic = kneighbors_graph(semantic_representation, k_s, mode='connectivity',metric="cosine")
    adj_matrix_geospatial = kneighbors_graph(geospatial_representation, k_g, mode='connectivity',metric="haversine")
    adj_matrix_temporal = kneighbors_graph(temporal_representation, k_t, mode='connectivity',metric="euclidean")
    
    # sum up the k-neighbors graphs and define the consistency neighborhood
    # The instances are neighbors in consistency if they are neighbors in two or three representations
    adj_matrix_final = adj_matrix_semantic.toarray() + adj_matrix_geospatial.toarray() + adj_matrix_temporal.toarray()
    adj_matrix_final = np.where(adj_matrix_final < 2, 0, 1) # we can do something as consider the reciprocity. Like if A + A^t = 2, then theres a edge
    # get the matrix
    consistency_graph =  nx.DiGraph(adj_matrix_final) 
    return consistency_graph


if __name__ == "__main__":
    output_dir = "./GeneratedGraphs/Consistency" 
    Path(output_dir).mkdir(parents=True, exist_ok=True) # just making sure the directory exists ... can be made in .sh but it seems caution to do it here
    
    parser = argparse.ArgumentParser(description ='Generate the reference graphs')
    parser.add_argument('--hyperparameters',  type=int, nargs=3, 
						help='Number of neighbors for semantic, geographical and temporal neighbors graph: k_s, k_g, k_t')
    parser.add_argument('--dataset_path', type=str, help='The name of the current dataset file') 
    args = parser.parse_args()

    k_s, k_g, k_t = args.hyperparameters
    dataset_path = args.dataset_path
 
    representations = load_representation(args.dataset_path, 'else')
    semantic, geospatial, temporal = representations['semantic'], representations['geospatial'], representations['temporal']
    consistency_graph = get_consistency_graph(semantic, geospatial, temporal, k_s, k_g, k_t) # this graph is also directed...

    output_file = f"{output_dir}/consistency_edges_{Path(args.dataset_path).stem}_{k_s}_{k_g}_{k_t}.pkl"
    save_edges(consistency_graph, output_file)
