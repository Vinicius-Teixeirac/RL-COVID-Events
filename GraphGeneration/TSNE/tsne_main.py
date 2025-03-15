import logging
import argparse
from pathlib import Path

from GraphGeneration import get_tsne_graph
from Utils import load_event_features, save_edges

def argument_parsing():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate the t-SNE graphs')
    
    parser.add_argument('--k_tsne', type=int, required=True,
                        help='Number of neighbors in the t-SNE-reducted k-NN graph')
    parser.add_argument('--perplexity', type=int, required=True,
                        help='t-SNE perplexity hyperparameter')
    parser.add_argument("--initialization", type=str, choices=['pca', 'spectral'], required=True,
                        help="The initial point positions for embedding")
    parser.add_argument("--metric", type=str, choices=['euclidean', 'cosine'], required=True,
                        help="The input space metric to calculate neares neighbors")
    parser.add_argument("--apply_pca", type=int, choices=[0, 1], required=True,
                        help="Whether to apply PCA reduction before t-SNE (0 or 1)")
    parser.add_argument('--n_components', type=int, default=2,
                        help='Number of components for t-SNE dimensionality reduction (default: 2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='The name of the current dataset')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/TSNE",
                        help="Directory to save the output generated graph.")
    
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Parsing arguments
    args = argument_parsing()

    # Getting dataset details
    dataset_name = args.dataset_name
    output_dir = args.output_dir

    # Loading representation
    final_representation = load_event_features(dataset_name, "final")
    tsne_graph = get_tsne_graph(
        events = final_representation,
        k_tsne = args.k_tsne,
        perplexity = args.perplexity,
        initialization = args.initialization,
        metric = args.metric,
        apply_pca = bool(args.apply_pca),
        n_components = args.n_components,
        random_state = args.random_state
    )

    if args.apply_pca:
        output_subdir = Path(str(output_dir) + "_PCA")
        output_file = output_subdir / dataset_name / f"tsne_pca_edges_{args.k_tsne}_{args.perplexity}_{args.initialization}_{args.metric}.pkl"
    else:
        output_subdir = output_dir
        output_file = output_subdir / dataset_name / f"tsne_edges_{args.k_tsne}_{args.perplexity}_{args.initialization}_{args.metric}.pkl"
    
    # save graph
    save_edges(tsne_graph, output_file)
