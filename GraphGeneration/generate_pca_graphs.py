import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

from utils import load_representation, save_edges


def get_PCA_graph(representation: np.ndarray, k: int, dim: int) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph using PCA-reduced representation.

    Parameters
    ----------
    representation : np.ndarray
        The input data representation (high-dimensional).
    k : int
        Number of nearest neighbors for the graph.
    dim : int
        Desired number of dimensions for PCA reduction.

    Returns
    -------
    nx.DiGraph
        A directed graph representing the k-nearest neighbors for each instance in the new reduced representation.
    """
    # defines the PCA settings
    pca = PCA(n_components=dim)
    # fits it to representation
    pca.fit(representation)
    # gets the new transformed space
    representation_PCA = pca.transform(representation)
    # defines the k-neighbors graph adjacency matrix
    adj_matrix_pca = kneighbors_graph(representation_PCA, k, mode='connectivity').toarray()
    # obtains the graph (as a nx.digraph) from its adjacency
    pca_graph = nx.DiGraph(adj_matrix_pca)
    # returns the k-neighbors graph as a networkx object
    return pca_graph


if __name__ == "__main__":    
    # parsing the arguments that'll be used on the PCA method
    parser = argparse.ArgumentParser(description ='Generate the PCA graphs')
    parser.add_argument('--hyperparameters',  type=int, nargs=2, 
                            help='Number of neighbors in PCA graph, desired dimensionality: k_pca, dim')
    parser.add_argument('--dataset_path', type=str, help='The path for the current dataset file')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/PCA", help="Directory to save the output.")
    args = parser.parse_args()

    # getting the dataset specifications
    dataset_path = args.dataset_path
    dataset_name = Path(dataset_path).stem
    output_dir = args.output_dir

    # acquiring from arguments the method's hyperparameters
    k_pca, dim = args.hyperparameters
    
    # loading the representation and obtaining the PCA graph from it
    final_representation = load_representation(dataset_name, 'final')
    pca_graph = get_PCA_graph(final_representation, k_pca, dim)

    # defining the file's name and saving the representation
    output_file = f"{output_dir}/pca_edges_{dataset_name}_{k_pca}.pkl"
    save_edges(pca_graph, output_file)

