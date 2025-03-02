import argparse
from pathlib import Path
import logging

from GraphEvaluation import compare_graphs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Parsing the evaluation arguments
    parser = argparse.ArgumentParser(description="Evaluate graph representations in parallel.")
    parser.add_argument('--dataset_path', type=str, help='The path for the current dataset file')
    parser.add_argument("--reference_folder", type=str, default="./GeneratedGraphs/Consistency", help="Directory for consistency graphs.")
    parser.add_argument("--comparison_folder", type=str, help="Directory for a specific method's graphs (PCA, TSNE, UMAP, etc.).")
    parser.add_argument("--method", type=str, help="Method name (PCA, TSNE, UMAP, etc.).")
    parser.add_argument("--output_dir", type=str, default="./EvaluationResults", help="Directory to save evaluation results.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs to use (-1 for all cores).")
    args = parser.parse_args()

    # Accessing folder paths
    dataset_path = args.dataset_path
    dataset_name = Path(dataset_path).stem

    # Building the actual folders for reference and comparison
    reference_folder = Path(args.reference_folder) / dataset_name
    comparison_folder = Path(args.comparison_folder) / dataset_name
    output_dir = Path(args.output_dir) / dataset_name
    method_name = args.method
    n_jobs=args.n_jobs

    # Evaluating results
    compare_graphs(reference_folder, comparison_folder, method_name, output_dir, n_jobs)
    logging.info(f"{method_name} evaluation completed successfully.")
