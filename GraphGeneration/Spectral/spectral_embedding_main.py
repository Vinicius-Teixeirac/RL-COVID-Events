import logging
import argparse
from pathlib import Path

from GraphGeneration import get_spectral_graph
from Utils import load_event_features, save_edges

def argument_parsing():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate the Laplacian Eigenmaps graphs')
    
    parser.add_argument('--k_spectral', type=int, required=True,
                        help='Number of neighbors in the Laplacian Eigenmaps-reducted k-NN graph')
    parser.add_argument('--n_neighbors', type=int, required=True,
                        help='Number of neighbors of each point for Laplacian Eigenmaps dimensionality reduction')
    parser.add_argument('--n_components', type=int, default=2,
                        help='Number of components for Laplacian Eigenmaps dimensionality reduction (default: 2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='The name of the current dataset')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/Spectral",
                        help="Directory to save the output generated graph.")
    
    return parser.parse_args()

if __name__ == "__main__":    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Parsing the arguments that'll be used on the Laplacian Eigenmaps method
    args = argument_parsing()

    # Getting the dataset specifications
    dataset_name = args.dataset_name
    output_dir = args.output_dir

    # Loading the representation and obtaining the spectral graph from it
    final_representation = load_event_features(dataset_name, 'final')
    spectral_graph = get_spectral_graph(
        events = final_representation,
        k_spectral = args.k_spectral,
        n_neighbors = args.n_neighbors, 
        n_components = args.n_components, 
        random_state = args.random_state)

    output_file = Path(output_dir) / dataset_name / f"spectral_edges_{args.k_spectral}_{args.n_neighbors}.pkl"
    save_edges(spectral_graph, output_file)

