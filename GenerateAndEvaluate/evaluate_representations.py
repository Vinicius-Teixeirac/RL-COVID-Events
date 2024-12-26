import os
import pickle
import argparse
from pathlib import Path


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

    output_dir = "./EvaluationResults/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    with open(f"./GeneratedGraphs/Consistency/edges_consistency_{dataset_name[:-4]}_{k_s}_{k_g}_{k_t}.pkl", 'rb') as file:
        edges_consistency = pickle.load(file)
        
    with open(f"./GeneratedGraphs/PCA/edges_pca_{dataset_name[:-4]}_{k_pca}.pkl", 'rb') as file:
        edges_pca = pickle.load(file)   
        
    with open(f"./GeneratedGraphs/TSNE/edges_tsne_{dataset_name[:-4]}_{k_tsne}_{ppxty}.pkl", 'rb') as file:
        edges_consistency = pickle.load(file)

    with open(f"./GeneratedGraphs/UMAP/edges_umap_{dataset_name[:-4]}_{k_umap}_{umap_n_neighbors}__{min_dist}.pkl", 'rb') as file:
        edges_consistency = pickle.load(file)
    
# Its pivotal to check if the files name is matching