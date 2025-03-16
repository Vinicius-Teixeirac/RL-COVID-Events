import argparse
import logging
from pathlib import Path

from GraphEvaluation import get_consistency_graph
from Utils import load_event_features, save_edges

def argument_parsing():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate the consistency graphs')
    
    parser.add_argument('--k_s', type=int, required=True,
                        help='Number of nearest neighbors in semantic graph')
    parser.add_argument('--k_g', type=int, required=True,
                        help='Number of nearest neighbors in geospatial graph')
    parser.add_argument("--k_t", type=int, required=True,
                        help='Number of nearest neighbors in temporal graph')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='The name of the current dataset')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/Consistency",
                        help="Directory to save the output generated graph.")
    
    return parser.parse_args()


if __name__ == "__main__": 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # The arguments that'll be used to create consistency matrices
    args = argument_parsing()
    
    # Loading the representations to obtaining the consistency graph from them
    representations = load_event_features(args.dataset_name)
    semantic, geospatial, temporal = representations['semantic'], representations['geospatial'], representations['temporal']
    consistency_graph = get_consistency_graph(
        semantic_features = semantic,
        geospatial_features = geospatial,
        temporal_features = temporal,
        k_s = args.k_s,
        k_g = args.k_g,
        k_t = args.k_t) 

    # Defining the file's name and saving the representation   
    output_file = Path(args.output_dir) / args.dataset_name / f"consistency_edges_{args.k_s}_{args.k_g}_{args.k_t}.pkl"
    save_edges(consistency_graph, output_file)
