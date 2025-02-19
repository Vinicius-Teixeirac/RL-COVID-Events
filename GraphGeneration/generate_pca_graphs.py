import logging
import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

from utils import load_representation, save_edges

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_PCA_graph(representation: np.ndarray,
                  k: int,
                  whiten: bool,
                  dim: int) -> nx.DiGraph:
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

    Raises
    ------
    ValueError
        If `k` or `dim` are not positive integers, or if `dim` is greater than the number of features in `representation`.
    """
    # Validates inputs
    if k <= 0 or dim <= 0:
        logging.error("Non positive value for k or dimension")
        raise ValueError("Both k-value and dimension must be positive integers.")

    if dim > representation.shape[1]:
        logging.error("Inconsistent dimension")
        raise ValueError(f"dim ({dim}) cannot be greater than the number of features in representation ({representation.shape[1]}).")

    # defines the PCA settings
    pca = PCA(n_components=dim, whiten=whiten)
    # fits it to representation
    pca.fit(representation)
    # gets the new transformed space
    representation_pca = pca.transform(representation)

    # Constructs k-nearest neighbors graph (returns a sparse matrix)
    adj_matrix_pca = kneighbors_graph(representation_pca, k, mode='connectivity')
    # Converts adjacency matrix to a directed NetworkX graph
    pca_graph = nx.from_scipy_sparse_array(adj_matrix_pca, create_using=nx.DiGraph)

    return pca_graph


if __name__ == "__main__":    
    # parsing the arguments that'll be used on the PCA method
    parser = argparse.ArgumentParser(description ='Generate the PCA graphs')
    parser.add_argument('--hyperparameters',  type=int, nargs=2, help='Number of neighbors in PCA graph, desired dimensionality')
    parser.add_argument('--whiten', type=int, help='PCA whiten hyperparameter (0 or 1)')
    parser.add_argument('--dataset_path', type=str, help='The path for the current dataset file')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/PCA", help="Directory to save the output.")
    args = parser.parse_args()

    # getting the dataset specifications
    dataset_path = args.dataset_path
    dataset_name = Path(dataset_path).stem
    output_dir = args.output_dir

    # acquiring from arguments the method's hyperparameters
    k_pca, dim = args.hyperparameters
    whiten = bool(args.whiten)  
    
    # loading the representation and obtaining the PCA graph from it
    final_representation = load_representation(dataset_name, 'final')
    pca_graph = get_PCA_graph(final_representation, k_pca, whiten, dim)

    # defining the file's name and saving the representation
    whiten_suffix = "_whitened" if whiten else ""
    output_file = Path(output_dir) / dataset_name / f"pca_edges_{k_pca}{whiten_suffix}.pkl"
    save_edges(pca_graph, output_file)

