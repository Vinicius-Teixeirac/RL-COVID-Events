import os
import pickle
import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph

def get_tSNE_graph(representation: np.array,  k: int, perplexity: int, dim: int, random_state: int) -> np.array:
    tsne = TSNE(n_components=dim, perplexity=perplexity, random_state=random_state, metric='cosine') # which metric will we actually use?
    tsne.fit(representation)
    representation_tSNE = tsne.transform(representation) 

    A_TSNE = kneighbors_graph(representation_tSNE, k, mode='connectivity').toarray()
    
    return A_TSNE


if __name__ == "__main__":
    output_dir = "./GeneratedGraphs/Tsne"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description ='Generate the t-SNE graphs')
    parser.add_argument('--hyperparameters',  type=int, nargs=4, 
                            help='''Number of neighbors in t-SNE graph, perplexity, desired dimensionality, and random state: 
                                    k_tsne, ppxty, desired_dimensionality, rnd_state''')
    parser.add_argument('--dataset_path', type=str, help='The name of the current dataset file') 

    args = parser.parse_args()
    k_tsne, ppxty, desired_dimensionality, rnd_state = args.hyperparameters

    dataset_path = args.dataset_path
    dataset_name = os.path.basename(dataset_path)

    R_f_scaled = np.load(f'./DatasetRepresentations/{dataset_name[:-4]}_representation.npy')

    A_TSNE = get_tSNE_graph(R_f_scaled, k_tsne, ppxty, desired_dimensionality, rnd_state)

    G_TSNE =  nx.DiGraph(A_TSNE)

    edges = set(G_TSNE.edges())
    output_file = f"{output_dir}/edges_tsne_{dataset_name[:-4]}_{k_tsne}_{ppxty}.pkl"
    
    with open(output_file, 'wb') as f:
        pickle.dump(edges, f)
    
    print(f"Edges saved successfully to {output_file}", flush=True)

