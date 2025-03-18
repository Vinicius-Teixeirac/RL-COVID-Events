import argparse
from pathlib import Path
import logging

from GraphEvaluation import compare_graphs

def argument_parsing():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate graph representations in parallel.")

    parser.add_argument('--dataset_name', type=str, required=True,
                        help='The path for the current dataset file')
    parser.add_argument("--reference_folder", type=str, default="./GeneratedGraphs/Consistency",
                         help="Directory for consistency graphs.")
    parser.add_argument("--comparison_folder", type=str, required=True,
                         help="Directory for a specific method's graphs (PCA, TSNE, UMAP, etc.).")
    parser.add_argument("--method", type=str, required=True,
                        help="Method name (PCA, TSNE, UMAP, etc.).")
    parser.add_argument("--output_dir", type=str, default="./EvaluationResults", 
                        help="Directory to save evaluation results.")
    parser.add_argument("--n_jobs", type=int, default=-1, 
                        help="Number of parallel jobs to use (-1 for all cores).")
    
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Parsing the evaluation arguments
    args = argument_parsing()

    # Building the actual folders for reference and comparison
    reference_folder = Path(args.reference_folder) / args.dataset_name
    comparison_folder = Path(args.comparison_folder) / args.dataset_name
    output_dir = Path(args.output_dir) / args.dataset_name
    method_name = args.method
    n_jobs=args.n_jobs

    # Evaluating results
    compare_graphs(reference_folder, comparison_folder, method_name, output_dir, n_jobs)
    logging.info(f"{method_name} evaluation completed successfully.")