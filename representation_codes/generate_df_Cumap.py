from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import os

""" Getting the hiperparameters for the experiments

"""

parser = argparse.ArgumentParser(description ='Get the specifications')

parser.add_argument('hyperparameters', metavar='N', type=int, nargs=5,
                    help='four integer hyperparameters: k_UMAP, UMAP_n_neighbors, min_dist, desired_dimensionality, rnd_state')


parser.add_argument('dataset_path', type=str, 
                    help='The name of the dataset file') 

args = parser.parse_args()
k_UMAP, UMAP_n_neighbors, m_dist, desired_dimensionality, rnd_state = args.hyperparameters

dataset_path = args.dataset_path
dataset_name = os.path.basename(dataset_path)

output_dir = "./results/Cumap"
Path(output_dir).mkdir(parents=True, exist_ok=True)

dataset = pd.read_pickle(f"{dataset_path}")


""" Applying UMAP aligned (with Consistency)

"""

kclique = nx.community.k_clique_communities(G_consistencia, 3)
labels_dict = {}
c_id = 0
for c in kclique:
  for node in c:
    labels_dict[node] = c_id
  c_id += 1

y = [-1]*G_consistencia.number_of_nodes()

for node in G_consistencia.nodes():
  if node in labels_dict: y[node] = labels_dict[node]

aligned_reducer = umap.UMAP(n_neighbors=UMAP_n_neighbors, n_components=desired_dimensionality, min_dist=m_dist, metric='cosine', random_state=rnd_state)

R_f_consistency_UMAP = aligned_reducer.fit_transform(R_f_scaled,y=y)

A_consistency_UMAP = kneighbors_graph(R_f_consistency_UMAP, k_UMAP, mode='connectivity').toarray()

G_consistency_UMAP =  nx.Graph(A_consistency_UMAP)

df_consistency_UMAP = pd.DataFrame([str(edge) for edge in G_consistency_UMAP.edges()], columns=['Edges'])

df_consistency_UMAP.to_parquet(f'{output_dir}/df_consistency_UMAP_{dataset_name}_neighbors={UMAP_n_neighbors}_k={k_UMAP}_min_dist={m_dist}.parquet')

print('success', flush=True)