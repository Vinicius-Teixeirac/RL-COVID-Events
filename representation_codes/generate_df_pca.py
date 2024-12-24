from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import os

""" Getting the hiperparameters for the experiments

"""

parser = argparse.ArgumentParser(description ='Get the specifications')

parser.add_argument('hyperparameters', metavar='N', type=int, nargs=2,
                    help='two integer hyperparameters: k_pca, desired_dimensionality')


parser.add_argument('dataset_path', type=str, 
                    help='The name of the dataset file') 

args = parser.parse_args()
k_pca, desired_dimensionality = args.hyperparameters

dataset_path = args.dataset_path
dataset_name = os.path.basename(dataset_path)

output_dir = "./results/pca"
Path(output_dir).mkdir(parents=True, exist_ok=True)

dataset = pd.read_pickle(f"{dataset_path}")

R_f_scaled = np.load(f'./representations/{dataset_name[:-4]}_representations.npy')

""" Applying PCA

"""

from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


pca = PCA(n_components=desired_dimensionality)
pca.fit(R_f_scaled)
R_f_PCA = pca.transform(R_f_scaled)

A_PCA = kneighbors_graph(R_f_PCA, k_pca, mode='connectivity').toarray()

G_PCA =  nx.Graph(A_PCA)

df_PCA = pd.DataFrame([str(edge) for edge in G_PCA.edges()], columns=['Edges'])

df_PCA.to_parquet(f'{output_dir}/df_pca_{dataset_name[:-4]}_k={k_pca}.parquet')

print('success', flush=True)
