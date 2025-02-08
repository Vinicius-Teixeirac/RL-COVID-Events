import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph

from utils import load_representation, save_edges

def get_tSNE_graph(representation: np.ndarray,  k: int, perplexity: int, dim: int, random_state: int) -> np.ndarray:
    # defines the t-SNE settings 
    tsne = TSNE(n_components=dim, perplexity=perplexity, random_state=random_state, metric='euclidean')
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
    # creating the t-SNE generated graphs' directory to save results
    output_dir = "./GeneratedGraphs/TSNE"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # parsing the arguments that'll be used on the t-SNE method
    parser = argparse.ArgumentParser(description ='Generate the t-SNE graphs')
    parser.add_argument('--hyperparameters',  type=int, nargs=4, 
                            help='''Number of neighbors in t-SNE graph, perplexity, desired dimensionality, and random state: 
                                    k_tsne, ppxty, desired_dimensionality, rnd_state''')
    parser.add_argument('--dataset_path', type=str, help='The name of the current dataset file') 
    args = parser.parse_args()

    # acquiring from arguments the method's hyperparameters
    k_tsne, ppxty, desired_dimensionality, rnd_state = args.hyperparameters

    # loading the representation and obtaining the t-SNE graph from it
    final_representation = load_representation(args.dataset_path, 'final')
    tsne_graph = get_tSNE_graph(final_representation, k_tsne, ppxty, desired_dimensionality, rnd_state)

    # defining the file's name and saving the representation
    output_file = f"{output_dir}/tsne_edges_{Path(args.dataset_path).stem}_{k_tsne}_{ppxty}.pkl"
    save_edges(tsne_graph, output_file)

