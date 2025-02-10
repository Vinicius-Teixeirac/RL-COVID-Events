import argparse
from pathlib import Path

import numpy as np
from umap import UMAP
import networkx as nx
from sklearn.neighbors import kneighbors_graph

from utils import load_representation, save_edges


def get_UMAP_graph(representation: np.ndarray, k: int, n_neighbors: int, min_dist: float, dim: int, random_state: int) -> nx.DiGraph:
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
    dim : int
        Desired number of dimensions for UMAP reduction.
    rnd_state: int
        Seed for reproducibility (since UMAP is stochastic)
    
    Returns
    -------
    nx.DiGraph
        A directed graph representing the k-nearest neighbors for each instance in the new reduced representation.
    """

    # defines the UMAP settings
    umap = UMAP(n_neighbors=n_neighbors, n_components=dim, min_dist=min_dist, metric='euclidean',
                   random_state=random_state)
    # fits it to the representation 
    umap.fit(representation)
    # gets the new transformed space
    representation_UMAP = umap.transform(representation)
    # defines the k-neighbors graph adjacency matrix
    adj_matrix_umap = kneighbors_graph(representation_UMAP, k, mode='connectivity').toarray()
    # obtains the graph (as a nx.digraph) from its adjacency
    umap_graph =  nx.DiGraph(adj_matrix_umap)
    # returns the k-neighbors graph as a networkx object
    return umap_graph


if __name__ == "__main__":
    # parsing the arguments that'll be used on the UMAP method
    parser = argparse.ArgumentParser(description='Generate the UMAP graphs')
    parser.add_argument('--int_hyperparameters', type=int, nargs=4,
                        help='''Four intergers arguments: Number of neighbors in UMAP graph, UMAP n_neighbors hyperparameter, 
                        desired dimensionality, and random state''')
    parser.add_argument('--float_hyperparameters', type=float, nargs=1,
                        help='One float hyperparameter: UMAP min_dist hyperparameter')
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

    # loading the representation and obtaining the UMAP graph from it
    final_representation = load_representation(dataset_name, 'final')
    umap_graph = get_UMAP_graph(final_representation, k_umap, n_neighbors, min_dist, dim, rnd_state)

    # defining the file's name and saving the representation 
    output_file = f"{output_dir}/umap_edges_{dataset_name}_{k_umap}_{n_neighbors}_{min_dist}.pkl"
    save_edges(umap_graph, output_file)

