import logging
import argparse
from pathlib import Path

from GraphGeneration import get_lle_graph
from Utils import load_event_features, save_edges

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":    
    # Parsing the arguments that'll be used on the LLE method
    parser = argparse.ArgumentParser(description ='Generate the LLE graphs')
    parser.add_argument('--hyperparameters',  type=int, nargs=4, help='Number of neighbors in LLE graph, n_neighbors for LLE, desired dimensionality, and random state')
    parser.add_argument('--method', type=str, choices=['standard', 'hessian', 'modified', 'ltsa'], help='LLE method parameter')
    parser.add_argument('--dataset_path', type=str, help='The path for the current dataset file')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/LLE", help="Directory to save the output.")
    args = parser.parse_args()

    # Getting the dataset specifications
    dataset_path = args.dataset_path
    dataset_name = Path(dataset_path).stem
    output_dir = args.output_dir

    # Acquiring from arguments the method's hyperparameters
    k_lle, n_neighbors, dim, rnd_state = args.hyperparameters
    method = args.method

    # Loading the representation and obtaining the lle graph from it
    final_representation = load_event_features(dataset_name, 'final')
    lle_graph = get_lle_graph(final_representation, k_lle, n_neighbors, method, dim, rnd_state)

    output_file = Path(output_dir) / dataset_name / f"lle_edges_{k_lle}_{n_neighbors}_{method}.pkl"
    save_edges(lle_graph, output_file)

