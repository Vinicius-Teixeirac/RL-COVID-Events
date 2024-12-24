from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import os

""" Getting the hiperparameters for the experiments

"""

parser = argparse.ArgumentParser(description ='Get the specifications')

parser.add_argument('hyperparameters', metavar='N', type=int, nargs=4,
                    help='four integer hyperparameters: k_TSNE, ppxty, desired_dimentionality, rnd_state')


parser.add_argument('dataset_path', type=str, 
                    help='The name of the dataset file') 

args = parser.parse_args()
k_TSNE, ppxty, desired_dimensionality, rnd_state = args.hyperparameters

dataset_path = args.dataset_path
dataset_name = os.path.basename(dataset_path)

output_dir = "./results/tsne"
Path(output_dir).mkdir(parents=True, exist_ok=True)

dataset = pd.read_pickle(f"{dataset_path}")

R_f_scaled = np.load(f'./representations/{dataset_name[:-4]}_representations.npy')

""" Applying T-SNE

"""

from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph


R_f_TSNE = TSNE(n_components=desired_dimensionality, perplexity=ppxty, random_state=rnd_state, metric='cosine').fit_transform(R_f_scaled)

A_TSNE = kneighbors_graph(R_f_TSNE, k_TSNE, mode='connectivity').toarray()

G_TSNE =  nx.Graph(A_TSNE)

df_TSNE = pd.DataFrame([str(edge) for edge in G_TSNE.edges()], columns=['Edges'])

df_TSNE.to_parquet(f'{output_dir}/df_TSNE_{dataset_name[:-4]}_k={k_TSNE}_ppxty={ppxty}.parquet')

print('success', flush=True)

