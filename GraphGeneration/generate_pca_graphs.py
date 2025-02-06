import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

from utils import load_representation, save_edges

def get_PCA_graph(representation: np.ndarray, k: int, dim: int) -> np.ndarray:
    # defines PCA
    pca = PCA(n_components=dim)
    # fits it to representation
    pca.fit(representation)
    # gets the new transformed space
    representation_PCA = pca.transform(representation)
    # defines the k-neighbors graph adjacency
    adj_matrix_pca = kneighbors_graph(representation_PCA, k, mode='connectivity').toarray()
    # gets the matrix
    pca_graph = nx.DiGraph(adj_matrix_pca)
    return pca_graph


if __name__ == "__main__":
    output_dir = "./GeneratedGraphs/PCA"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    parser = argparse.ArgumentParser(description ='Generate the PCA graphs')
    parser.add_argument('--hyperparameters',  type=int, nargs=2, 
                            help='Number of neighbors in PCA graph, desired dimensionality: k_pca, desired_dimensionality')
    parser.add_argument('--dataset_path', type=str, help='The name of the current dataset file') 

    args = parser.parse_args()
    k_pca, desired_dimensionality = args.hyperparameters
    
    final_representation = load_representation(args.dataset_path)
    pca_graph = get_PCA_graph(final_representation, k_pca, desired_dimensionality)

    output_file = f"{output_dir}/pca_edges_{Path(args.dataset_path).stem}_{k_pca}.pkl"
    save_edges(pca_graph, output_file)

