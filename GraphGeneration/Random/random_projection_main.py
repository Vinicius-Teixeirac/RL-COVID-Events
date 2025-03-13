import logging
import argparse
from pathlib import Path

from GraphGeneration import get_random_projection_graph
from Utils import load_event_features, save_edges

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    # Parsing the arguments for the Random Projection method
    parser = argparse.ArgumentParser(description='Generate the Random Projection graphs')
    parser.add_argument('--hyperparameters', type=int, nargs=3, 
                        help='Number of neighbors for the graph, desired dimensionality, and random state')
    parser.add_argument('--dataset_path', type=str, help='The path for the current dataset file')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/RandomProjections",
                        help="Directory to save the output.")
    args = parser.parse_args()

    # Getting dataset specifications
    dataset_path = args.dataset_path
    dataset_name = Path(dataset_path).stem
    output_dir = args.output_dir

    # Extracting hyperparameters: k (neighbors), desired dimension, and random state
    k, dim, rnd_state = args.hyperparameters

    # Load the representation (assuming 'final' representation is desired)
    final_representation = load_event_features(dataset_name, 'final')
    
    # Generate the random projection graph from the representation
    rp_graph = get_random_projection_graph(final_representation, k, dim, rnd_state)

    # Define the output file path and save the graph edges
    output_file = Path(output_dir) / dataset_name / f"rp_edges_{k}.pkl"
    save_edges(rp_graph, output_file)
