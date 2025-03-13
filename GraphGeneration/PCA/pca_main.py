import logging
import argparse
from pathlib import Path

from GraphGeneration import get_pca_graph
from Utils import load_event_features, save_edges

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":    
    # Parsing the arguments that'll be used on the PCA method
    parser = argparse.ArgumentParser(description ='Generate the PCA graphs')
    parser.add_argument('--hyperparameters',  type=int, nargs=2, help='Number of neighbors in PCA graph, desired dimensionality')
    parser.add_argument('--whiten', type=int, choices=[0, 1], help='PCA whiten hyperparameter (0 or 1)')
    parser.add_argument('--dataset_path', type=str, help='The path for the current dataset file')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/PCA", help="Directory to save the output.")
    args = parser.parse_args()

    # Getting the dataset specifications
    dataset_path = args.dataset_path
    dataset_name = Path(dataset_path).stem
    output_dir = args.output_dir

    # Acquiring from arguments the method's hyperparameters
    k_pca, dim = args.hyperparameters
    whiten = bool(args.whiten)  
    
    # Loading the representation and obtaining the PCA graph from it
    final_representation = load_event_features(dataset_name, 'final')
    pca_graph = get_pca_graph(final_representation, k_pca, whiten, dim)

    # Defining the file's name and saving the representation
    whiten_suffix = "_whitened" if whiten else ""
    output_file = Path(output_dir) / dataset_name / f"pca_edges_{k_pca}{whiten_suffix}.pkl"
    save_edges(pca_graph, output_file)

