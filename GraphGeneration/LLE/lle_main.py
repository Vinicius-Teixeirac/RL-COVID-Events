import logging
import argparse
from pathlib import Path

from GraphGeneration import get_lle_graph
from Utils import load_event_features, save_edges

def argument_parsing():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate the Locally Linear Embeddings graphs')
    
    parser.add_argument('--k_lle', type=int, required=True,
                        help='Number of neighbors in the LLE-reducted k-NN graph')
    parser.add_argument('--n_neighbors', type=int, required=True,
                        help='Number of neighbors of each point for LLE dimensionality reduction')
    parser.add_argument('--method', type=str, choices=['standard', 'hessian', 'modified', 'ltsa'], required=True,
                        help='LLE method parameter (check sklearn.manifold.LocallyLinearEmbedding for more details)')
    parser.add_argument('--n_components', type=int, default=2,
                        help='Number of components for LLE dimensionality reduction (default: 2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='The name of the current dataset')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/LLE",
                        help="Directory to save the output generated graph.")
    
    return parser.parse_args()

if __name__ == "__main__":    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Parsing the arguments that'll be used on the LLE method
    args = argument_parsing()

    # Loading the representation and obtaining the lle graph from it
    final_representation = load_event_features(args.dataset_name, 'final')
    lle_graph = get_lle_graph(
        events = final_representation, 
        k_lle = args.k_lle,
        n_neighbors = args.n_neighbors,
        method = args.method,
        n_components = args.n_components,
        random_state = args.random_state)

    output_file = Path(args.output_dir) / args.dataset_name / f"lle_edges_{args.k_lle}_{args.n_neighbors}_{args.method}.pkl"
    save_edges(lle_graph, output_file)

