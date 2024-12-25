from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import os
from data_steps.generate_representations import get_representations

""" Getting the hiperparameters for the experiments"""

parser = argparse.ArgumentParser(description ='Generate the consistency datasets')
parser.add_argument('--hyperparameters',  type=int, nargs=3, 
                    help='Number of neighbors for semantic, geographical and temporal neighbors graph: k_s, k_g, k_t')
parser.add_argument('--dataset_path', type=str, help='The name of the current dataset file') 
args = parser.parse_args()

k_s, k_g, k_t = args.hyperparameters
dataset_path = args.dataset_path

dataset_name = os.path.basename(dataset_path)

output_dir = "./results/consistencia"
Path(output_dir).mkdir(parents=True, exist_ok=True)

"""Getting the representation components"""

R_s_norm, R_g, R_t, _ = get_representations(dataset_path)

""" Consistency graph as a way to measure the success of the representation"""

from sklearn.neighbors import kneighbors_graph

A_s = kneighbors_graph(R_s_norm, k_s, mode='connectivity',metric="cosine")
A_g = kneighbors_graph(R_g, k_g, mode='connectivity',metric="haversine")
A_t = kneighbors_graph(R_t, k_t, mode='connectivity',metric="euclidean")

# these kneighbors_graphs are directed

A_f = A_s.toarray() + A_g.toarray() + A_t.toarray()

A_f = np.where(A_f < 2, 0, 1)

G_consistencia =  nx.Graph(A_f) 

""" Creating a dataframe to facilitate the comparsions """

df_consistencia = pd.DataFrame([str(edge) for edge in G_consistencia.edges()], columns=['Edges'])

df_consistencia.to_parquet(f'{output_dir}/df_consistencia_{dataset_name[:-4]}_{k_s}_{k_g}_{k_t}.parquet')

print('success', flush=True)