import logging
import argparse
from pathlib import Path

from GraphGeneration import get_pca_graph
from Utils import load_event_features, save_edges

def argument_parsing():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate the Principal Component Analysis graphs')
    
    parser.add_argument('--k_pca', type=int, required=True,
                        help='Number of neighbors in the PCA-reducted k-NN graph')
    parser.add_argument('--whiten', type=int, choices=[0, 1], required=True,
                        help='PCA whiten hyperparameter (0 or 1)')
    parser.add_argument('--n_components', type=int, default=2,
                        help='Number of components for PCA dimensionality reduction (default: 2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='The name of the current dataset')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/PCA",
                        help="Directory to save the output generated graph.")
    
    return parser.parse_args()

if __name__ == "__main__":    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Parsing the arguments that'll be used on the PCA method
    args = argument_parsing()

    # Loading the representation and obtaining the PCA graph from it
    final_representation = load_event_features(args.dataset_name, 'final')
    pca_graph = get_pca_graph(
        events = final_representation,
        k_pca = args.k_pca,
        whiten = bool(args.whiten),
        n_components = args.n_components,
        random_state = args.random_state)

    # Defining the file's name and saving the representation
    whiten_suffix = "_whitened" if args.whiten else ""

    output_file = Path(args.output_dir) / args.dataset_name / f"pca_edges_{args.k_pca}{whiten_suffix}.pkl"
    save_edges(pca_graph, output_file)