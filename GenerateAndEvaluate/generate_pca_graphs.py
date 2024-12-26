import os
import pickle
import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

def get_pca_graph(representation: np.array, k: int, dim: int) -> np.array:
    pca = PCA(n_components=dim)
    pca.fit(representation)
    representation_PCA = pca.transform(representation)

    A_PCA = kneighbors_graph(representation_PCA, k, mode='connectivity').toarray()
    
    return A_PCA


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
    dataset_name = os.path.basename(dataset_path)

    R_f_scaled = np.load(f'./DatasetRepresentations/{dataset_name[:-4]}_representation.npy')

    A_PCA = get_pca_graph(R_f_scaled, k_pca, desired_dimensionality)

    G_PCA =  nx.DiGraph(A_PCA)

    edges = set(G_PCA.edges())
    output_file = f"{output_dir}/edges_pca_{dataset_name[:-4]}_{k_pca}.pkl"
    
    with open(output_file, 'wb') as f:
        pickle.dump(edges, f)
    
    print(f"Edges saved successfully to {output_file}", flush=True)
