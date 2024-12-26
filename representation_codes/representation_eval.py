from pathlib import Path
import argparse
import pickle
import os

def graph_edges_metrics(edges_predicted: set, ground_truth: set) -> tuple: 
    
    true_positives = edges_predicted & ground_truth
    false_positives = edges_predicted - ground_truth
    false_negatives = ground_truth - edges_predicted

    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if len(true_positives) + len(false_positives) > 0 else 0
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if len(true_positives) + len(false_negatives) > 0 else 0

    return f"Precision: {precision:.2f}, Recall: {recall:.2f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ='Generate the consistency datasets')
    parser.add_argument('--hyperparameters',  type=int, nargs=9, 
                            help='Every relevant hyperparameter to compare each ')
    parser.add_argument('--dataset_path', type=str, help='The name of the current dataset file') 
    args = parser.parse_args()

    k_s, k_g, k_t, k_pca, k_tsne, ppxty, k_umap, umap_n_neighbors, min_dist = args.hyperparameters
    dataset_path = args.dataset_path
    
    dataset_name = os.path.basename(dataset_path)

    output_dir = "RL-COVID-EVENTS/Eval_results/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    with open(f"./RL-COVID-EVENTS/Results/Consistency/edges_Consistency_{dataset_name[:-4]}_{k_s}_{k_g}_{k_t}.pkl", 'rb') as file:
        edges_consistency = pickle.load(file)
        
    with open(f"./RL-COVID-EVENTS/Results/PCA/edges_PCA_{dataset_name[:-4]}_{k_pca}.pkl", 'rb') as file:
        edges_pca = pickle.load(file)   
        
    with open(f"./RL-COVID-EVENTS/Results/TSNE/edges_TSNE_{dataset_name[:-4]}_{k_tsne}_{ppxty}.pkl", 'rb') as file:
        edges_consistency = pickle.load(file)

    with open(f"./RL-COVID-EVENTS/Results/UMAP/edges_UMAP_{dataset_name[:-4]}_{k_umap}_{umap_n_neighbors}__{min_dist}.pkl", 'rb') as file:
        edges_consistency = pickle.load(file)
    
    

# Its pivotal to check if the files name is matching

# df_consistencia = pd.read_csv(f'{output_dir}/df_consistencia_{dataset_name}_{k_s}_{k_g}_{k_t}.csv',index_col=0)
# y_true = df_consistencia['Edges'].to_list()

# df_PCA = pd.read_csv(f'{output_dir}/df_pca_{dataset_name}_k={k_pca}_.csv',index_col=0)
# y_pred_pca = df_PCA['Edges'].to_list()

# df_TSNE = pd.read_csv(f'{output_dir}/df_TSNE_{dataset_name}_k={k_TSNE}_ppxty={ppxty}_.csv',index_col=0)
# y_pred_tsne = df_TSNE['Edges'].to_list()

# df_UMAP = pd.read_csv(f'{output_dir}/df_UMAP_{dataset_name}_neighbors={UMAP_n_neighbors}_k={k_UMAP}_.csv',index_col=0)
# y_pred_umap = df_UMAP['Edges'].to_list()

# df_consistency_UMAP = pd.read_csv(f'{output_dir}/df_alignedUMAP_{dataset_name}_neighbors={UMAP_n_neighbors}_k={k_UMAP}_.csv',index_col=0)
# y_pred_consistency_UMAP = df_consistency_UMAP['Edges'].to_list()

# print('PCA precision=',precision(y_true,y_pred_pca),' recall=',recall(y_true,y_pred_pca))
# print('TSNE precision=',precision(y_true,y_pred_tsne),' recall=',recall(y_true,y_pred_tsne))
# print('UMAP precision=',precision(y_true,y_pred_umap),' recall=',recall(y_true,y_pred_umap))
# print('UMAP consistency precision=',precision(y_true,y_pred_consistency_UMAP),' recall=',recall(y_true,y_pred_consistency_UMAP))

    