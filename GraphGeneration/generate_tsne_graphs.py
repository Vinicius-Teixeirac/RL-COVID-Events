import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph

from utils import load_representation, save_edges


def get_tSNE_graph(representation: np.ndarray,
                   k: int, 
                   ppxty: int,
                   dim: int,
                   rnd_state: int) -> nx.DiGraph:
    """
    Generates a k-nearest neighbors graph using t-SNE-reduced representation.

    Parameters
    ----------
    representation : np.ndarray
        The input data representation (high-dimensional).
    k : int
        Number of nearest neighbors for the graph.
    ppxty: int
        Perplexity. The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms.
    dim : int
        Desired number of dimensions for t-SNE reduction.
    rnd_state: int
        Seed for reproducibility (since t-SNE is stochastic)
    
    Returns
    -------
    nx.DiGraph
        A directed graph representing the k-nearest neighbors for each instance in the new reduced representation.

    Raises
    ------
    ValueError
        If `k` or `dim` are not positive integers, or if `dim` is greater than the number of features in `representation`, and if
        `perplexity` was bad choosen.
    """
    # Validates inputs
    if k <= 0 or dim <= 0:
        raise ValueError("Both k-value and dimension must be positive integers.")

    if dim > representation.shape[1]:
        raise ValueError(f"dim ({dim}) cannot be greater than the number of features in representation ({representation.shape[1]}).")
    
    # Validate perplexity (must be < number of points)
    if ppxty >= representation.shape[1]:
        raise ValueError(f"Perplexity ({ppxty}) must be smaller than the number of samples ({representation.shape[1]}).")

    # defines the t-SNE settings 
    tsne = TSNE(n_components=dim, perplexity=ppxty, random_state=rnd_state, metric='euclidean')
    # gets the new transformed space
    representation_tSNE = tsne.fit_transform(representation) 

    # Constructs k-nearest neighbors graph (returns a sparse matrix)
    adj_matrix_tsne = kneighbors_graph(representation_tSNE, k, mode='connectivity')
    # Converts adjacency matrix to a directed NetworkX graph
    tsne_graph = nx.from_scipy_sparse_matrix(adj_matrix_tsne, create_using=nx.DiGraph)

    return tsne_graph


if __name__ == "__main__":
    # parsing the arguments that'll be used on the t-SNE method
    parser = argparse.ArgumentParser(description ='Generate the t-SNE graphs')
    parser.add_argument('--hyperparameters',  type=int, nargs=4, 
                            help='''Four intergers arguments: Number of neighbors in t-SNE graph, perplexity, desired dimensionality, 
                            and random state''')
    parser.add_argument('--dataset_path', type=str, help='The path for the current dataset file')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/TSNE", help="Directory to save the output.")
    args = parser.parse_args()

    # getting the dataset specifications
    dataset_path = args.dataset_path
    dataset_name = Path(dataset_path).stem
    output_dir = args.output_dir

    # acquiring from arguments the method's hyperparameters
    k_tsne, ppxty, dim, rnd_state = args.hyperparameters

    # loading the representation and obtaining the t-SNE graph from it
    final_representation = load_representation(dataset_name, 'final')
    tsne_graph = get_tSNE_graph(final_representation, k_tsne, ppxty, dim, rnd_state)

    # defining the file's name and saving the representation
    output_file = Path(output_dir) / dataset_name / f"tsne_edges_{k_tsne}_{ppxty}.pkl"
    save_edges(tsne_graph, output_file)

