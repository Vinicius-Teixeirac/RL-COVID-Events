import argparse
import logging
from pathlib import Path

from GraphEvaluation import get_consistency_graph
from Utils import load_event_features, save_edges

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__": 
    # The arguments that'll be used to create consistency matrices
    parser = argparse.ArgumentParser(description ='Generate the reference graphs')
    parser.add_argument('--hyperparameters',  type=int, nargs=3, 
						help='Number of neighbors for semantic, geographical and temporal neighbors graph: k_s, k_g, k_t')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='The name of the current dataset')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/Consistency", help="Directory to save the output.")
    args = parser.parse_args()

    # Getting the dataset specifications
    dataset_name = args.dataset_name
    output_dir = args.output_dir

    # Acquiring from arguments hyperparameters
    k_s, k_g, k_t = args.hyperparameters
    
    # Loading the representations to obtaining the consistency graph from them
    representations = load_event_features(dataset_name)
    semantic, geospatial, temporal = representations['semantic'], representations['geospatial'], representations['temporal']
    consistency_graph = get_consistency_graph(semantic, geospatial, temporal, k_s, k_g, k_t) 

    # Defining the file's name and saving the representation   
    output_file = Path(output_dir) / dataset_name / f"consistency_edges_{k_s}_{k_g}_{k_t}.pkl"
    save_edges(consistency_graph, output_file)
