import os
import pickle
import argparse
from pathlib import Path

import numpy as np
from umap import UMAP
import networkx as nx
from sklearn.neighbors import kneighbors_graph

def get_UMAP_graph(representation: np.array, k:int, umap_neighbors: int, dim: int, min_dist: float, random_state: int) -> np.array:
    reducer = UMAP(n_neighbors=umap_neighbors, n_components=dim, min_dist=min_dist, metric='cosine',
                   random_state=random_state)

    reducer.fit(representation)
    
    representation_UMAP = reducer.transform(representation)

    A_UMAP = kneighbors_graph(representation_UMAP, k, mode='connectivity').toarray()
    
    return A_UMAP


if __name__ == "__main__":
    output_dir = "./GeneratedGraphs/UMAP"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
     
    parser = argparse.ArgumentParser(description='Generate the UMAP graphs')

    parser.add_argument('--int_hyperparameters', type=int, nargs=4,
                        help='Four integer hyperparameters: k_UMAP, UMAP_n_neighbors, desired_dimensionality, rnd_state')

    parser.add_argument('--float_hyperparameters', type=float, nargs=1,
                        help='One float hyperparameter: m_dist')

    parser.add_argument('--dataset_path', type=str, help='The name of the current dataset file') 


    args = parser.parse_args()

    k_UMAP, UMAP_n_neighbors, desired_dimensionality, rnd_state = args.hyperparameters_int

    m_dist = args.hyperparameters_float[0]

    dataset_path = args.dataset_path
    dataset_name = os.path.basename(dataset_path)

    R_f_scaled = np.load(f'./DatasetRepresentations/{dataset_name[:-4]}_representation.npy')

    A_UMAP = get_UMAP_graph(R_f_scaled, k_UMAP, UMAP_n_neighbors, desired_dimensionality, m_dist, rnd_state)

    G_UMAP =  nx.DiGraph(A_UMAP)

    edges = set(G_UMAP.edges())
    output_file = f'{output_dir}/edges_umap_{dataset_name[:-4]}_neighbors={UMAP_n_neighbors}_k={k_UMAP}_min_dist={m_dist}.pkl'
    
    with open(output_file, 'wb') as f:
        pickle.dump(edges, f)
    
    print(f"Edges saved successfully to {output_file}", flush=True)

