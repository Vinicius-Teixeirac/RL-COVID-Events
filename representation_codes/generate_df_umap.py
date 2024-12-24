from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import os

""" Getting the hiperparameters for the experiments

"""

import argparse
import os

parser = argparse.ArgumentParser(description='Get the specifications')

parser.add_argument('hyperparameters_int', metavar='N', type=int, nargs=4,
                    help='Four integer hyperparameters: k_UMAP, UMAP_n_neighbors, desired_dimensionality, rnd_state')

parser.add_argument('hyperparameters_float', metavar='F', type=float, nargs=1,
                    help='One float hyperparameter: m_dist')

parser.add_argument('dataset_path', type=str, 
                    help='The name of the dataset file') 

args = parser.parse_args()

# Unpacking the integer hyperparameters
k_UMAP, UMAP_n_neighbors, desired_dimensionality, rnd_state = args.hyperparameters_int

# Extracting the single float value
m_dist = args.hyperparameters_float[0]

# Getting the dataset name from the path
dataset_path = args.dataset_path
dataset_name = os.path.basename(dataset_path)


output_dir = "./results/umap"
Path(output_dir).mkdir(parents=True, exist_ok=True)

dataset = pd.read_pickle(f"{dataset_path}")

R_f_scaled = np.load(f'./representations/{dataset_name[:-4]}_representations.npy')

""" Applying UMAP

"""

from umap import UMAP
from sklearn.neighbors import kneighbors_graph

reducer = UMAP(n_neighbors=UMAP_n_neighbors, n_components=desired_dimensionality, min_dist=m_dist, metric='cosine', random_state=rnd_state , n_jobs=10)

R_f_UMAP = reducer.fit_transform(R_f_scaled)

A_UMAP = kneighbors_graph(R_f_UMAP, k_UMAP, mode='connectivity').toarray()

G_UMAP =  nx.Graph(A_UMAP)

df_UMAP = pd.DataFrame([str(edge) for edge in G_UMAP.edges()], columns=['Edges'])

df_UMAP.to_parquet(f'{output_dir}/df_UMAP_{dataset_name[:-4]}_neighbors={UMAP_n_neighbors}_k={k_UMAP}_min_dist={m_dist}.parquet')

print('success', flush=True)