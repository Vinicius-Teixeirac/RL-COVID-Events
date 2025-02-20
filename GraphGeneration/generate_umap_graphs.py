import logging
import argparse
from pathlib import Path


import numpy as np
from umap import UMAP
import networkx as nx
from sklearn.neighbors import kneighbors_graph

from utils import load_representation, save_edges

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_UMAP_graph(representation: np.ndarray,
                   k: int,
                   n_neighbors: int,
                   min_dist: float,
                   initialization: str,
                   dim: int,
                   random_state: int) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph using UMAP-reduced representation.

    Parameters
    ----------
    representation : np.ndarray
        The input data representation (high-dimensional).
    k : int
        Number of nearest neighbors for the graph.
    n_neighbors: int
        This parameter controls how UMAP balances local versus global structure in the data.
    min_dist: float
        The min_dist parameter controls how tightly UMAP is allowed to pack points together.
    initialization: str
        How to initialize the low dimensional embedding.
    dim : int
        Desired number of dimensions for UMAP reduction.
    rnd_state: int
        Seed for reproducibility (since UMAP is stochastic)
    
    Returns
    -------
    nx.DiGraph
        A directed graph representing the k-nearest neighbors for each instance in the new reduced representation.

    Raises
    ------
    ValueError
        If `k`, `n_neighbors` `dim` are not positive integers, or if `dim` is greater than the number of features in `representation`.
    """
    # Validates inputs
    if k <= 0 or n_neighbors<=1 or dim <= 0:
        logging.error("Non positive value for k or dimension")
        raise ValueError("k-value, n_neighbors and dimension must be positive integers.")

    if dim > representation.shape[1]:
        logging.error("Inconsistent dimension")
        raise ValueError(f"dim ({dim}) cannot be greater than the number of features in representation ({representation.shape[1]}).")

    # defines the UMAP settings
    umap = UMAP(n_neighbors=n_neighbors, n_components=dim, min_dist=min_dist, metric='euclidean',
                   random_state=random_state, init=initialization)
    # fits it to the representation 
    umap.fit(representation)
    # gets the new transformed space
    representation_UMAP = umap.transform(representation)
    
    # Constructs k-nearest neighbors graph (returns a sparse matrix)
    adj_matrix_umap = kneighbors_graph(representation_UMAP, k, mode='connectivity')
    # Converts adjacency matrix to a directed NetworkX graph
    umap_graph =  nx.from_scipy_sparse_array(adj_matrix_umap)

    return umap_graph


if __name__ == "__main__":
    # parsing the arguments that'll be used on the UMAP method
    parser = argparse.ArgumentParser(description='Generate the UMAP graphs')
    parser.add_argument('--int_hyperparameters', type=int, nargs=4,
                        help='''Four intergers arguments: Number of neighbors in UMAP graph, UMAP n_neighbors hyperparameter, 
                        desired dimensionality, and random state''')
    parser.add_argument('--float_hyperparameters', type=float, help='One float hyperparameter: UMAP min_dist hyperparameter')
    parser.add_argument('--initialization', type=str, help='The initial point positions to be used in the embedding space.')
    parser.add_argument('--dataset_path', type=str, help='The path for the current dataset file')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/UMAP", help="Directory to save the output.")
    args = parser.parse_args()

    # getting the dataset specifications
    dataset_path = args.dataset_path
    dataset_name = Path(dataset_path).stem
    output_dir = args.output_dir

    # acquiring from arguments the method's hyperparameters
    k_umap, n_neighbors, dim, rnd_state = args.int_hyperparameters
    min_dist = args.float_hyperparameters
    init = args.initialization

    # since this UMAP parameters accounts the point itself as nearest neighbor
    n_neighbors += 1

    # loading the representation and obtaining the UMAP graph from it
    final_representation = load_representation(dataset_name, 'final')
    umap_graph = get_UMAP_graph(final_representation, k_umap, n_neighbors, min_dist, init, dim, rnd_state)

    # defining the file's name and saving the representation 
    output_file = Path(output_dir) / dataset_name / f"umap_edges_{k_umap}_{n_neighbors}_{min_dist}_{init}.pkl"
    save_edges(umap_graph, output_file)

