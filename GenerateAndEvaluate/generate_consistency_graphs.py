import os
import pickle
import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph

def get_consistency_adjacency(semantic_representation: np.array, geospatial_representation: np.array,
                              temporal_representation: np.array, k_s: int, k_g: int, k_t: int) -> np.array:
    # get each k-neighbors from representations
    adj_matrix_semantic = kneighbors_graph(semantic_representation, k_s, mode='connectivity',metric="cosine")
    adj_matrix_geospatial = kneighbors_graph(geospatial_representation, k_g, mode='connectivity',metric="haversine")
    adj_matrix_temporal = kneighbors_graph(temporal_representation, k_t, mode='connectivity',metric="euclidean")
    
    # sum up the k-neighbors graphs and define the consistency neighborhood
    # The instances are neighbors in consistency if they are neighbors in two or three representations
    adj_matrix_final = adj_matrix_semantic.toarray() + adj_matrix_geospatial.toarray() + adj_matrix_temporal.toarray()
    adj_matrix_final = np.where(adj_matrix_final <= 2, 0, 1) # we can do something as consider the reciprocity. Like if A + A^t = 2, then theres a edge
    
    return adj_matrix_final


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
 
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    representations_path = f"./DatasetRepresentations/{dataset_name}_representations.pkl"

    with open(representations_path, 'rb') as f:
        representations = pickle.load(f)
    
    semantic, geospatial, temporal = representations['semantic'], representations['geospatial'], representations['temporal']

    consistency_adjacency = get_consistency_adjacency(semantic, geospatial, temporal, k_s, k_g, k_t) # this graph is also directed...

    consistency_graph =  nx.DiGraph(consistency_adjacency) # here, i should put nx.DiGraph()

    consistency_edges = set(consistency_graph.edges())
    output_file = f"{output_dir}/consistency_edges_{dataset_name}_{k_s}_{k_g}_{k_t}.pkl"
    
    with open(output_file, 'wb') as f:
        pickle.dump(consistency_edges, f)
    
    print(f"Edges saved successfully to {output_file}", flush=True)
