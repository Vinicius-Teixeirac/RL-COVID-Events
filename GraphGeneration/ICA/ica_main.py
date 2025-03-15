import logging
import argparse
from pathlib import Path

from GraphGeneration import get_ica_graph
from Utils import load_event_features, save_edges

def argument_parsing():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate the Independent Component Analysis graphs')
    
    parser.add_argument('--k_ica', type=int, required=True,
                        help='Number of neighbors in the ICA-reducted k-NN graph')
    parser.add_argument('--whiten', type=str, choices=['0', 'unit-variance', 'arbitrary-variance'], required=True,
                        help='ICA whiten hyperparameter (check sklearn.decomposition.FastICA for more details)')
    parser.add_argument('--fun', type=str, choices=['logcosh', 'exp', 'cube'], required=True,
                        help='ICA fun hyperparameter (check sklearn.decomposition.FastICA for more details)')
    parser.add_argument('--n_components', type=int, default=2,
                        help='Number of components for ICA dimensionality reduction (default: 2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='The name of the current dataset')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/ICA",
                        help="Directory to save the output generated graph.")
    
    return parser.parse_args()

if __name__ == "__main__":    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Parsing the arguments that'll be used on the ICA method
    args = argument_parsing()

    # Getting the dataset specifications
    dataset_name = args.dataset_name
    output_dir = args.output_dir

    # Acquiring from arguments the method's hyperparameters
    whiten = False if args.whiten == '0' else args.whiten
    
    # Loading the features and obtaining the ICA graph from it
    final_features = load_event_features(dataset_name, 'final')
    ica_graph = get_ica_graph(
        events = final_features,
        k_ica = args.k_ica,
        whiten = whiten,
        fun = args.fun,
        n_components = args.n_components,
        random_state = args.random_state)

    output_file = Path(output_dir) / dataset_name / f"ica_edges_{args.k_ica}_{whiten}_{args.fun}.pkl"
    save_edges(ica_graph, output_file)
