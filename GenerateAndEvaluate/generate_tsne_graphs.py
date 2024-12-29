import os
import pickle
import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph

def get_tSNE_adjacency(representation: np.array,  k: int, perplexity: int, dim: int, random_state: int) -> np.array:
    # defines t-SNE
    tsne = TSNE(n_components=dim, perplexity=perplexity, random_state=random_state, metric='euclidean')
    # fits it to the representation
    tsne.fit(representation)
    # get the new transformed space
    representation_tSNE = tsne.transform(representation) 
    # defines the k-neighbors graph
    adj_matrix_tsne = kneighbors_graph(representation_tSNE, k, mode='connectivity').toarray()
    # returns its adjacency   
    return adj_matrix_tsne


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

    dataset_path = args.dataset_path
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    representations_path = f"./DatasetRepresentations/{dataset_name}_representations.pkl"

    with open(representations_path, 'rb') as f:
        representations = pickle.load(f)
        
    final_representation = representations['final']

    tsne_adjacency = get_tSNE_adjacency(final_representation, k_tsne, ppxty, desired_dimensionality, rnd_state)

    tsne_graph =  nx.DiGraph(tsne_adjacency)

    tsne_edges = set(tsne_graph.edges())
    output_file = f"{output_dir}/tsne_edges_{dataset_name}_{k_tsne}_{ppxty}.pkl"
    
    with open(output_file, 'wb') as f:
        pickle.dump(tsne_edges, f)
    
    print(f"Edges saved successfully to {output_file}", flush=True)

