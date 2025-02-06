import argparse
from pathlib import Path

import numpy as np
from umap import UMAP
import networkx as nx
from sklearn.neighbors import kneighbors_graph

from utils import load_representation, save_edges

def get_UMAP_graph(representation: np.ndarray, k:int, umap_neighbors: int, dim: int, min_dist: float, random_state: int) -> np.ndarray:
    # defines UMAP 
    umap = UMAP(n_neighbors=umap_neighbors, n_components=dim, min_dist=min_dist, metric='euclidean',
                   random_state=random_state)
    # fits it to the representation 
    umap.fit(representation)
    # gets the new transformed space
    representation_UMAP = umap.transform(representation)
    # defines the k-neighbors graph adjacency
    adj_matrix_umap = kneighbors_graph(representation_UMAP, k, mode='connectivity').toarray()
    # gets the matrix
    umap_graph =  nx.DiGraph(adj_matrix_umap)
    return umap_graph


if __name__ == "__main__":
    output_dir = "./GeneratedGraphs/UMAP"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
     
    parser = argparse.ArgumentParser(description='Generate the UMAP graphs')
    parser.add_argument('--int_hyperparameters', type=int, nargs=4,
                        help='Four integer hyperparameters: k_umap, umap_n_neighbors, desired_dimensionality, rnd_state')
    parser.add_argument('--float_hyperparameters', type=float, nargs=1,
                        help='One float hyperparameter: m_dist')
    parser.add_argument('--dataset_path', type=str, help='The name of the current dataset file') 

    args = parser.parse_args()
    k_umap, umap_n_neighbors, desired_dimensionality, rnd_state = args.hyperparameters_int
    m_dist = args.hyperparameters_float[0]

    final_representation = load_representation(args.dataset_path, 'final')
    umap_graph = get_UMAP_graph(final_representation, k_umap, umap_n_neighbors, desired_dimensionality, m_dist, rnd_state)

    output_file = f"{output_dir}/umap_edges_{Path(args.dataset_path).stem}_{k_umap}_{umap_n_neighbors}_{m_dist}.pkl"
    save_edges(umap_graph, output_file)

