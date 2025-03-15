import pickle
import argparse
import logging
from pathlib import Path

from GraphGeneration import get_umap_graph
from Utils import load_event_features, save_edges

def argument_parsing():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate the UMAP graphs')
    
    parser.add_argument('--k_umap', type=int, required=True,
                        help='Number of neighbors in the UMAP-reducted k-NN graph')
    parser.add_argument('--n_neighbors', type=int, required=True,
                        help='Number of neighbors of each point for UMAP dimensionality reduction')
    parser.add_argument('--min_dist', type=float, required=True,
                         help='One float hyperparameter: UMAP min_dist hyperparameter')
    parser.add_argument("--initialization", type=str, choices=['pca', 'spectral'], required=True,
                        help="The initial point positions for embedding")
    parser.add_argument("--metric", type=str, choices=['euclidean', 'cosine'], required=True,
                        help="The input space metric to calculate neares neighbors")
    parser.add_argument('--n_components', type=int, default=2,
                        help='Number of components for UMAP dimensionality reduction (default: 2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='The name of the current dataset')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/UMAP",
                        help="Directory to save the output generated graph.")
    parser.add_argument('--precomputed_knn_path', type=str, default=None,
                        help="A precomputed knn matrix to optimize the n_neighbors graph construction setep (check umap-learn docs)")

    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Parsing the arguments that'll be used on the UMAP method
    args = argument_parsing()

    if args.precomputed_knn_path is not None:
        with open(args.precomputed_knn_path, "rb") as f:
            precomputed_knn = pickle.load(f)

    # Loading the representation and obtaining the UMAP graph from it
    final_representation = load_event_features(args.dataset_name, 'final')
    umap_graph = get_umap_graph(
        events = final_representation,
        k_umap = args.k_umap,
        n_neighbors = args.n_neighbors,
        min_dist = args.min_dist,
        initialization = args.initialization,
        n_components = args.n_components,
        random_state = args.random_state,
        precomputed_knn = precomputed_knn)

    # Defining the file's name and saving the representation 
    output_file = Path(args.output_dir) / args.dataset_name / f"umap_edges_{args.k_umap}_{args.n_neighbors}_{args.min_dist}_{args.initialization}_{args.metric}.pkl"
    save_edges(umap_graph, output_file)

