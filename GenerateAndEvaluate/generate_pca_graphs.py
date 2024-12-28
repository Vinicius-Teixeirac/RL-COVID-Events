import os
import pickle
import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

def get_PCA_adjacency(representation: np.array, k: int, dim: int) -> np.array:
    pca = PCA(n_components=dim)
    pca.fit(representation)
    representation_PCA = pca.transform(representation)

    adj_matrix_pca = kneighbors_graph(representation_PCA, k, mode='connectivity').toarray()
    
    return adj_matrix_pca


if __name__ == "__main__":
    output_dir = "./GeneratedGraphs/PCA"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    parser = argparse.ArgumentParser(description ='Generate the PCA graphs')
    parser.add_argument('--hyperparameters',  type=int, nargs=2, 
                            help='Number of neighbors in PCA graph, desired dimensionality: k_pca, desired_dimensionality')
    parser.add_argument('--dataset_path', type=str, help='The name of the current dataset file') 

    args = parser.parse_args()
    k_pca, desired_dimensionality = args.hyperparameters

    dataset_path = args.dataset_path
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    representations_path = f"./DatasetRepresentations/{dataset_name}_representations.pkl"

    with open(representations_path, 'rb') as f:
        representations = pickle.load(f)
        
    final_representation = representations['final']

    pca_adjacency = get_PCA_adjacency(final_representation, k_pca, desired_dimensionality)
    
    pca_graph =  nx.DiGraph(pca_adjacency)

    pca_edges = set(pca_graph.edges())
    output_file = f"{output_dir}/pca_edges_{dataset_name}_{k_pca}.pkl"
    
    with open(output_file, 'wb') as f:
        pickle.dump(pca_edges, f)
    
    print(f"Edges saved successfully to {output_file}", flush=True)
