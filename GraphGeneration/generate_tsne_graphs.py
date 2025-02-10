import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph

from utils import load_representation, save_edges


def get_tSNE_graph(representation: np.ndarray,  k: int, ppxty: int, dim: int, rnd_state: int) -> nx.DiGraph:
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
    """
    # defines the t-SNE settings 
    tsne = TSNE(n_components=dim, perplexity=ppxty, random_state=rnd_state, metric='euclidean')
    # fits it to the representation
    tsne.fit(representation)
    # gets the new transformed space
    representation_tSNE = tsne.transform(representation) 
    # defines the k-neighbors graph adjacency matrix
    adj_matrix_tsne = kneighbors_graph(representation_tSNE, k, mode='connectivity').toarray()
    # obtains the graph (as a nx.digraph) from its adjacency
    tsne_graph =  nx.DiGraph(adj_matrix_tsne)
    # returns the k-neighbors graph as a networkx object
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
    output_file = f"{output_dir}/tsne_edges_{dataset_name}_{k_tsne}_{ppxty}.pkl"
    save_edges(tsne_graph, output_file)

