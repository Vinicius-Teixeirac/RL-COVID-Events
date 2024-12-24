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

parser.add_argument('hyperparameters', metavar='N', type=int, nargs=8,
                    help='eight integer hyperparameters: k_s, k_g, k_t, k_pca, ppxty, k_TSNE, k_UMAP, UMAP_n_neighbors')


parser.add_argument('dataset_path', type=str, 
                    help='The name of the dataset file') 

args = parser.parse_args()
k_s, k_g, k_t, k_pca, ppxty, k_TSNE, k_UMAP, UMAP_n_neighbors = args.hyperparameters

dataset_path = args.dataset_path
dataset_name = os.path.basename(dataset_path)

output_dir = "./results/"
Path(output_dir).mkdir(parents=True, exist_ok=True)

dataset = pd.read_pickle(f"{dataset_path}")

"""
Getting the representations:
* R_s - Semantic
* R_t - Temporal
* R_g - Geospatial
"""

# R_s
R_s = np.array(dataset['embeddings'].to_list()) 
R_s_norm = normalize(R_s, norm="l2") # normalized to reduce numerical instability in cosine measure

# R_g
R_g = np.array(dataset[['country_lat','country_lng']])

# R_t
dataset['dates'] = pd.to_datetime(dataset['dates'])

minimal_date = min(dataset['dates'])

tdiff = list(dataset['dates'] - minimal_date)

time_delta = [t_delta.days for t_delta in tdiff]

dataset['date_timediff'] = time_delta

R_t = np.array(dataset[['date_timediff']])

""" Concatenating the representations 

"""
scaler = StandardScaler()

R_f = np.concatenate((R_s_norm,R_g,R_t),axis=1)

R_f_scaled = scaler.fit_transform(R_f)

""" Consistency graph as a way to measure the success of the representation

"""

from sklearn.neighbors import kneighbors_graph

A_s = kneighbors_graph(R_s_norm, k_s, mode='connectivity',metric="cosine")
A_g = kneighbors_graph(R_g, k_g, mode='connectivity',metric="haversine")
A_t = kneighbors_graph(R_t, k_t, mode='connectivity',metric="euclidean")

A_f = A_s.toarray() + A_g.toarray() + A_t.toarray()

A_f = np.where(A_f < 2, 0, 1)

G_consistencia =  nx.Graph(A_f)

""" Creating a dataframe to facilitate the comparsions

"""

df_consistencia = pd.DataFrame([str(edge) for edge in G_consistencia.edges()], columns=['Edges'])

df_consistencia.to_csv(f'{output_dir}/df_consistencia_{dataset_name}_{k_s}_{k_g}_{k_t}.csv')

desired_dimensionality = 2

rnd_state = 42

""" Applying PCA

"""

from sklearn.decomposition import PCA

pca = PCA(n_components=desired_dimensionality)
pca.fit(R_f_scaled)
R_f_PCA = pca.transform(R_f_scaled)

A_PCA = kneighbors_graph(R_f_PCA, k_pca, mode='connectivity').toarray()

G_PCA =  nx.Graph(A_PCA)

df_PCA = pd.DataFrame([str(edge) for edge in G_PCA.edges()], columns=['Edges'])

df_PCA.to_csv(f'{output_dir}/df_pca_{dataset_name}_k={k_pca}_.csv')

""" Applying T-SNE

"""

from sklearn.manifold import TSNE

R_f_TSNE = TSNE(n_components=desired_dimensionality, perplexity=ppxty, random_state=rnd_state, metric='cosine').fit_transform(R_f_scaled)

A_TSNE = kneighbors_graph(R_f_TSNE, k_TSNE, mode='connectivity').toarray()

G_TSNE =  nx.Graph(A_TSNE)

df_TSNE = pd.DataFrame([str(edge) for edge in G_TSNE.edges()], columns=['Edges'])

df_TSNE.to_csv(f'{output_dir}/df_TSNE_{dataset_name}_k={k_TSNE}_ppxty={ppxty}_.csv')

""" Applying UMAP

"""

import umap

reducer = umap.UMAP(n_neighbors=UMAP_n_neighbors, n_components=desired_dimensionality, metric='cosine', random_state=rnd_state)

R_f_UMAP = reducer.fit_transform(R_f_scaled)

A_UMAP = kneighbors_graph(R_f_UMAP, k_UMAP, mode='connectivity').toarray()

G_UMAP =  nx.Graph(A_UMAP)

df_UMAP = pd.DataFrame([str(edge) for edge in G_UMAP.edges()], columns=['Edges'])

df_UMAP.to_csv(f'{output_dir}/df_UMAP_{dataset_name}_neighbors={UMAP_n_neighbors}_k={k_UMAP}_.csv')

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

aligned_reducer = umap.UMAP(n_neighbors=UMAP_n_neighbors, n_components=desired_dimensionality, metric='cosine', random_state=rnd_state)

R_f_consistency_UMAP = aligned_reducer.fit_transform(R_f_scaled,y=y)

A_consistency_UMAP = kneighbors_graph(R_f_consistency_UMAP, k_UMAP, mode='connectivity').toarray()

G_consistency_UMAP =  nx.Graph(A_consistency_UMAP)

df_consistency_UMAP = pd.DataFrame([str(edge) for edge in G_consistency_UMAP.edges()], columns=['Edges'])

df_consistency_UMAP.to_csv(f'{output_dir}/df_consistency_UMAP_{dataset_name}_neighbors={UMAP_n_neighbors}_k={k_UMAP}_.csv')

print('success', flush=True)