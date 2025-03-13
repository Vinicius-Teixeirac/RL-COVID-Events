import pickle
import argparse
from pathlib import Path

from GraphGeneration import get_umap_graph
from Utils import load_event_features, save_edges

if __name__ == "__main__":
    # Parsing the arguments that'll be used on the UMAP method
    parser = argparse.ArgumentParser(description='Generate the UMAP graphs')
    parser.add_argument('--int_hyperparameters', type=int, nargs=4,
                        help='''Four intergers arguments: Number of neighbors in UMAP graph, UMAP n_neighbors hyperparameter, 
                        desired dimensionality, and random state''')
    parser.add_argument('--float_hyperparameters', type=float, help='One float hyperparameter: UMAP min_dist hyperparameter')
    parser.add_argument('--initialization', type=str, help='The initial point positions to be used in the embedding space.')
    parser.add_argument('--dataset_path', type=str, help='The path for the current dataset file')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/UMAP", help="Directory to save the output.")
    parser.add_argument('--precomputed_knn_path', type=str, default=None)
    args = parser.parse_args()

    # Getting the dataset specifications
    dataset_path = args.dataset_path
    dataset_name = Path(dataset_path).stem
    output_dir = args.output_dir

    # Acquiring from arguments the method's hyperparameters
    k_umap, n_neighbors, dim, rnd_state = args.int_hyperparameters
    min_dist = args.float_hyperparameters
    init = args.initialization

    # Since this UMAP parameters accounts the point itself as nearest neighbor
    n_neighbors += 1

    if args.precomputed_knn_path is not None:
        with open(args.precomputed_knn_path, "rb") as f:
            precomputed_knn = pickle.load(f)

    # Loading the representation and obtaining the UMAP graph from it
    final_representation = load_event_features(dataset_name, 'final')
    umap_graph = get_umap_graph(final_representation, k_umap, n_neighbors, min_dist, init, dim, rnd_state, precomputed_knn)

    # Defining the file's name and saving the representation 
    output_file = Path(output_dir) / dataset_name / f"umap_edges_{k_umap}_{n_neighbors}_{min_dist}_{init}.pkl"
    save_edges(umap_graph, output_file)

