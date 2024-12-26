import os
import pickle
import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph

from DataPreparation.generate_representations import get_representations


def get_consistency_adjacency(R_s: np.array, R_g: np.array, R_t: np.array, k_s: int, k_g: int, k_t: int) -> np.array:
    # these kneighbors_graphs are directed
    A_s = kneighbors_graph(R_s, k_s, mode='connectivity',metric="cosine")
    A_g = kneighbors_graph(R_g, k_g, mode='connectivity',metric="haversine")
    A_t = kneighbors_graph(R_t, k_t, mode='connectivity',metric="euclidean")
    
    A_f = A_s.toarray() + A_g.toarray() + A_t.toarray()
    A_f = np.where(A_f < 2, 0, 1) # we can do something as consider the reciprocity. Like if A + A^t = 2, then theres a edge
    
    return A_f


if __name__ == "__main__":
    output_dir = "./GeneratedGraphs/Consistency"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    parser = argparse.ArgumentParser(description ='Generate the reference graphs')
    parser.add_argument('--hyperparameters',  type=int, nargs=3, 
						help='Number of neighbors for semantic, geographical and temporal neighbors graph: k_s, k_g, k_t')
    parser.add_argument('--dataset_path', type=str, help='The name of the current dataset file') 
    args = parser.parse_args()

    k_s, k_g, k_t = args.hyperparameters
    dataset_path = args.dataset_path
 
    dataset_name = os.path.basename(dataset_path)
    R_s_norm, R_g, R_t, _ = get_representations(dataset_path)

    A_f = get_consistency_adjacency(R_s_norm, R_g, R_t, k_s, k_g, k_t) # this graph is also directed...

    G_consistency =  nx.DiGraph(A_f) # here, i should put nx.DiGraph()

    edges = set(G_consistency.edges())
    output_file = f"{output_dir}/edges_consistency_{dataset_name[:-4]}_{k_s}_{k_g}_{k_t}.pkl"
    
    with open(output_file, 'wb') as f:
        pickle.dump(edges, f)
    
    print(f"Edges saved successfully to {output_file}", flush=True)
