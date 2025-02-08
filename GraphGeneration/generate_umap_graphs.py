import argparse
from pathlib import Path

import numpy as np
from umap import UMAP
import networkx as nx
from sklearn.neighbors import kneighbors_graph

from utils import load_representation, save_edges

def get_UMAP_graph(representation: np.ndarray, k:int, umap_neighbors: int, dim: int, min_dist: float, random_state: int) -> np.ndarray:
    # defines the UMAP settings
    umap = UMAP(n_neighbors=umap_neighbors, n_components=dim, min_dist=min_dist, metric='euclidean',
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
    # creating the UMAP generated graphs' directory to save results
    output_dir = "./GeneratedGraphs/UMAP"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # parsing the arguments that'll be used on the UMAP method
    parser = argparse.ArgumentParser(description='Generate the UMAP graphs')
    parser.add_argument('--int_hyperparameters', type=int, nargs=4,
                        help='Four integer hyperparameters: k_umap, umap_n_neighbors, desired_dimensionality, rnd_state')
    parser.add_argument('--float_hyperparameters', type=float, nargs=1,
                        help='One float hyperparameter: m_dist')
    parser.add_argument('--dataset_path', type=str, help='The name of the current dataset file') 
    args = parser.parse_args()

    # acquiring from arguments the method's hyperparameters
    k_umap, umap_n_neighbors, desired_dimensionality, rnd_state = args.hyperparameters_int
    m_dist = args.hyperparameters_float[0]

    # loading the representation and obtaining the UMAP graph from it
    final_representation = load_representation(args.dataset_path, 'final')
    umap_graph = get_UMAP_graph(final_representation, k_umap, umap_n_neighbors, desired_dimensionality, m_dist, rnd_state)

    # defining the file's name and saving the representation 
    output_file = f"{output_dir}/umap_edges_{Path(args.dataset_path).stem}_{k_umap}_{umap_n_neighbors}_{m_dist}.pkl"
    save_edges(umap_graph, output_file)

