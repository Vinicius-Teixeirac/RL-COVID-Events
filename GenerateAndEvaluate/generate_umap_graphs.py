import os
import pickle
import argparse
from pathlib import Path

import numpy as np
from umap import UMAP
import networkx as nx
from sklearn.neighbors import kneighbors_graph

def get_UMAP_adjacency(representation: np.array, k:int, umap_neighbors: int, dim: int, min_dist: float, random_state: int) -> np.array:
    # defines UMAP 
    umap = UMAP(n_neighbors=umap_neighbors, n_components=dim, min_dist=min_dist, metric='euclidean',
                   random_state=random_state)
    # fits it to the representation 
    umap.fit(representation)
    # get the new transformed space
    representation_UMAP = umap.transform(representation)
    # defines the k-neighbors graph
    adj_matrix_umap = kneighbors_graph(representation_UMAP, k, mode='connectivity').toarray()
    # returns its adjacency
    return adj_matrix_umap


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

    dataset_path = args.dataset_path
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    representations_path = f"./DatasetRepresentations/{dataset_name}_representations.pkl"

    with open(representations_path, 'rb') as f:
        representations = pickle.load(f)
        
    final_representation = representations['final']

    umap_adjacency = get_UMAP_adjacency(final_representation, k_umap, umap_n_neighbors, desired_dimensionality, m_dist, rnd_state)

    umap_graph =  nx.DiGraph(umap_adjacency)

    umap_edges = set(umap_graph.edges())
    output_file = f'{output_dir}/umap_edges_{dataset_name}_{k_umap}_{umap_n_neighbors}_{m_dist}.pkl'
    
    with open(output_file, 'wb') as f:
        pickle.dump(umap_edges, f)
    
    print(f"Edges saved successfully to {output_file}", flush=True)

