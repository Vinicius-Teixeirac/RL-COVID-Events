from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
import os

""" Getting the hiperparameters for the experiments

"""

parser = argparse.ArgumentParser(description ='Get the specifications')

parser.add_argument('hyperparameters', metavar='N', type=int, nargs=3,
                    help='three integer hyperparameters: k_s, k_g, k_t')


parser.add_argument('dataset_path', type=str, 
                    help='The name of the dataset file') 

args = parser.parse_args()
k_s, k_g, k_t = args.hyperparameters

dataset_path = args.dataset_path
dataset_name = os.path.basename(dataset_path)

output_dir = "./results/consistencia"
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

df_consistencia.to_parquet(f'{output_dir}/df_consistencia_{dataset_name[:-4]}_{k_s}_{k_g}_{k_t}.parquet')

print('success', flush=True)