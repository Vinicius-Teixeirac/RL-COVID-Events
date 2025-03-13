import logging
import argparse
from pathlib import Path

from GraphGeneration import get_spectral_graph
from Utils import load_event_features, save_edges

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":    
    # Parsing the arguments that'll be used on the Laplacian Eigenmaps method
    parser = argparse.ArgumentParser(description ='Generate the Laplacian Eigenmaps graphs')
    parser.add_argument('--hyperparameters',  type=int, nargs=4, help='Number of neighbors in Laplacian Eigenmaps graph, n_neighbors for Laplacian Eigenmaps, desired dimensionality, and random state')
    parser.add_argument('--dataset_path', type=str, help='The path for the current dataset file')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/LaplacianEigenmaps", help="Directory to save the output.")
    args = parser.parse_args()

    # Getting the dataset specifications
    dataset_path = args.dataset_path
    dataset_name = Path(dataset_path).stem
    output_dir = args.output_dir

    # Acquiring from arguments the method's hyperparameters
    k_spectral, n_neighbors, dim, rnd_state = args.hyperparameters

    # Loading the representation and obtaining the spectral graph from it
    final_representation = load_event_features(dataset_name, 'final')
    spectral_graph = get_spectral_graph(final_representation, k_spectral, n_neighbors, dim, rnd_state)

    output_file = Path(output_dir) / dataset_name / f"spectral_edges_{k_spectral}_{n_neighbors}.pkl"
    save_edges(spectral_graph, output_file)

