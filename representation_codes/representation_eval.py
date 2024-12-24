"""## Avaliação

"""

from pathlib import Path
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description ='Got the values for k')
parser.add_argument('integers', metavar ='N',
                    type = int, nargs = 8) 

args = parser.parse_args()
k_s, k_g, k_t, k_pca, ppxty, k_TSNE, k_UMAP, UMAP_n_neighbors = args.integers

dataset_name = "dados.pkl"
output_dir = "./results/"
Path(output_dir).mkdir(parents=True, exist_ok=True)

dataset = pd.read_pickle(dataset_name)

from irmetrics.topk import recall
from irmetrics.topk import precision

df_consistencia = pd.read_csv(f'{output_dir}/df_consistencia_{dataset_name}_{k_s}_{k_g}_{k_t}.csv',index_col=0)
y_true = df_consistencia['Edges'].to_list()

df_PCA = pd.read_csv(f'{output_dir}/df_pca_{dataset_name}_k={k_pca}_.csv',index_col=0)
y_pred_pca = df_PCA['Edges'].to_list()

df_TSNE = pd.read_csv(f'{output_dir}/df_TSNE_{dataset_name}_k={k_TSNE}_ppxty={ppxty}_.csv',index_col=0)
y_pred_tsne = df_TSNE['Edges'].to_list()

df_UMAP = pd.read_csv(f'{output_dir}/df_UMAP_{dataset_name}_neighbors={UMAP_n_neighbors}_k={k_UMAP}_.csv',index_col=0)
y_pred_umap = df_UMAP['Edges'].to_list()

df_consistency_UMAP = pd.read_csv(f'{output_dir}/df_alignedUMAP_{dataset_name}_neighbors={UMAP_n_neighbors}_k={k_UMAP}_.csv',index_col=0)
y_pred_consistency_UMAP = df_consistency_UMAP['Edges'].to_list()

print('PCA precision=',precision(y_true,y_pred_pca),' recall=',recall(y_true,y_pred_pca))
print('TSNE precision=',precision(y_true,y_pred_tsne),' recall=',recall(y_true,y_pred_tsne))
print('UMAP precision=',precision(y_true,y_pred_umap),' recall=',recall(y_true,y_pred_umap))
print('UMAP consistency precision=',precision(y_true,y_pred_consistency_UMAP),' recall=',recall(y_true,y_pred_consistency_UMAP))