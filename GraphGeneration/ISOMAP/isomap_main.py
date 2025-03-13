import logging
import argparse
from pathlib import Path

from GraphGeneration import get_spectral_graph
from Utils import load_event_features, save_edges

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":    
    # Parsing the arguments that'll be used on the Isomap method
    parser = argparse.ArgumentParser(description ='Generate the Isomap graphs')
    parser.add_argument('--hyperparameters',  type=int, nargs=3, help='Number of neighbors in Isomap graph, n_neighbors for Isomap, desired dimensionality')
    parser.add_argument('--dataset_path', type=str, help='The path for the current dataset file')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/ISOMAP", help="Directory to save the output.")
    args = parser.parse_args()

    # Getting the dataset specifications
    dataset_path = args.dataset_path
    dataset_name = Path(dataset_path).stem
    output_dir = args.output_dir

    # Acquiring from arguments the method's hyperparameters
    k_isomap, n_neighbors, dim = args.hyperparameters

    # Loading the representation and obtaining the isomap graph from it
    final_representation = load_event_features(dataset_name, 'final')
    isomap_graph = get_spectral_graph(final_representation, k_isomap, n_neighbors, dim)

    output_file = Path(output_dir) / dataset_name / f"isomap_edges_{k_isomap}_{n_neighbors}.pkl"
    save_edges(isomap_graph, output_file)

