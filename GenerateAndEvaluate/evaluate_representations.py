import os
import pickle
from pathlib import Path


def graph_edges_metrics(edges_predicted: set, ground_truth: set) -> tuple:
    true_positives = edges_predicted & ground_truth
    false_positives = edges_predicted - ground_truth
    false_negatives = ground_truth - edges_predicted

    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if len(true_positives) + len(false_positives) > 0 else 0
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if len(true_positives) + len(false_negatives) > 0 else 0

    return precision, recall


def compare_graphs(reference_folder: str, comparison_folder: str, method: str, output_dir: str):
    reference_files = sorted(os.listdir(reference_folder))
    comparison_files = sorted(os.listdir(comparison_folder))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []

    for ref_file in reference_files:
        ref_path = os.path.join(reference_folder, ref_file)
        with open(ref_path, 'rb') as f:
            ref_edges = pickle.load(f)

        for comp_file in comparison_files:
            comp_path = os.path.join(comparison_folder, comp_file)
            with open(comp_path, 'rb') as f:
                comp_edges = pickle.load(f)

            precision, recall = graph_edges_metrics(comp_edges, ref_edges)
            results.append({
                "Reference Graph": ref_file,
                "Compared Graph": comp_file,
                "Method": method,
                "Precision": precision,
                "Recall": recall
            })

            print(f"Compared {ref_file} with {comp_file}: Precision={precision:.2f}, Recall={recall:.2f}")

    results_file = os.path.join(output_dir, f"comparison_results_{method}.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Comparison results saved to {results_file}")


if __name__ == "__main__":
    reference_folder = "./GeneratedGraphs/Consistency"
    pca_folder = "./GeneratedGraphs/PCA"
    tsne_folder = "./GeneratedGraphs/TSNE"
    umap_folder = "./GeneratedGraphs/UMAP"
    output_dir = "./EvaluationResults"

    compare_graphs(reference_folder, pca_folder, "PCA", output_dir)
    compare_graphs(reference_folder, tsne_folder, "TSNE", output_dir)
    compare_graphs(reference_folder, umap_folder, "UMAP", output_dir)
