import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph

from utils import load_representation, save_edges

def get_tSNE_matrix(representation: np.ndarray,  k: int, perplexity: int, dim: int, random_state: int) -> np.ndarray:
    # defines t-SNE
    tsne = TSNE(n_components=dim, perplexity=perplexity, random_state=random_state, metric='euclidean')
    # fits it to the representation
    tsne.fit(representation)
    # gets the new transformed space
    representation_tSNE = tsne.transform(representation) 
    # defines the k-neighbors graph
    adj_matrix_tsne = kneighbors_graph(representation_tSNE, k, mode='connectivity').toarray()
    # gets the matrix
    tsne_graph =  nx.DiGraph(adj_matrix_tsne)
    return tsne_graph


if __name__ == "__main__":
    output_dir = "./GeneratedGraphs/TSNE"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description ='Generate the t-SNE graphs')
    parser.add_argument('--hyperparameters',  type=int, nargs=4, 
                            help='''Number of neighbors in t-SNE graph, perplexity, desired dimensionality, and random state: 
                                    k_tsne, ppxty, desired_dimensionality, rnd_state''')
    parser.add_argument('--dataset_path', type=str, help='The name of the current dataset file') 

    args = parser.parse_args()
    k_tsne, ppxty, desired_dimensionality, rnd_state = args.hyperparameters

    final_representation = load_representation(args.dataset_path, 'final')
    tsne_graph = get_tSNE_matrix(final_representation, k_tsne, ppxty, desired_dimensionality, rnd_state)

    output_file = f"{output_dir}/tsne_edges_{Path(args.dataset_path).stem}_{k_tsne}_{ppxty}.pkl"
    save_edges(tsne_graph, output_file)

