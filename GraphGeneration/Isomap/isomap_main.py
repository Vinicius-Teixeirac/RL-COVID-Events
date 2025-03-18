import logging
import argparse
from pathlib import Path

from GraphGeneration import get_isomap_graph
from Utils import load_event_features, save_edges

def argument_parsing():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate the Isomap graphs')
    
    parser.add_argument('--k_isomap', type=int, required=True,
                        help='Number of neighbors in the Isomap-reducted k-NN graph')
    parser.add_argument('--n_neighbors', type=int, required=True,
                        help='Number of neighbors of each point for Isomap dimensionality reduction')
    parser.add_argument('--n_components', type=int, default=2,
                        help='Number of components for Isomap dimensionality reduction (default: 2)')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='The name of the current dataset')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/Isomap",
                        help="Directory to save the output generated graph.")
    
    return parser.parse_args()

if __name__ == "__main__":    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        
    # Parsing the arguments that'll be used on the Isomap method
    args = argument_parsing()

    # Loading the representation and obtaining the isomap graph from it
    final_representation = load_event_features(args.dataset_name, 'final')
    isomap_graph = get_isomap_graph(
        events = final_representation, 
        k_isomap = args.k_isomap, 
        n_neighbors = args.n_neighbors,
        n_components = args.n_components)

    output_file = Path(args.output_dir) / args.dataset_name / f"isomap_edges_{args.k_isomap}_{args.n_neighbors}.pkl"
    save_edges(isomap_graph, output_file)