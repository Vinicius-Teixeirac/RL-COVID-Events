import logging
import argparse
from pathlib import Path

from GraphGeneration import get_random_projection_graph
from Utils import load_event_features, save_edges

def argument_parsing():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate the Gaussian Random Projection graphs')
    
    parser.add_argument('--k_rp', type=int, required=True,
                        help='Number of neighbors in the Random Projection-reducted k-NN graph')
    parser.add_argument('--n_components', type=int, default=2,
                        help='Number of components for Gaussian Random Projection dimensionality reduction (default: 2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='The name of the current dataset')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/RandomProjection",
                        help="Directory to save the output generated graph.")
    
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Parsing the arguments for the Random Projection method
    args = argument_parsing()

    # Load the representation 
    final_representation = load_event_features(args.dataset_name, 'final')
    
    # Generate the random projection graph from the representation
    rp_graph = get_random_projection_graph(
        events = final_representation, 
        k_random = args.k_rp,
        n_components = args.n_components,
        random_state = args.random_state)

    # Define the output file path and save the graph edges
    output_file = Path(args.output_dir) / args.dataset_name / f"random_proj_edges_{args.k_rp}.pkl"
    save_edges(rp_graph, output_file)
