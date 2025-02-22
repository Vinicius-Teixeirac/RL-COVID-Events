import logging
import argparse
from pathlib import Path

from GraphGeneration import get_pca_reduction, get_tsne_graph
from utils import load_representation, save_edges

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser(description="Generate the t-SNE graphs")
    parser.add_argument('--hyperparameters',  type=int, nargs=4, help='''Four intergers arguments: 
                        Number of neighbors in t-SNE graph, perplexity, desired dimensionality, and random state''')
    parser.add_argument("--apply_pca", type=int, choices=[0, 1], help="Whether to apply PCA reduction before t-SNE (0 or 1)")
    parser.add_argument("--initialization", type=str, help="The initial point positions for embedding")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/TSNE", help="Directory to save output")

    args = parser.parse_args()

    # Getting dataset details
    dataset_path = args.dataset_path
    dataset_name = Path(dataset_path).stem
    output_dir = Path(args.output_dir)

    # acquiring from arguments the method's hyperparameters
    k_tsne, ppxty, dim, rnd_state = args.hyperparameters
    init = args.initialization
    apply_pca = bool(args.apply_pca)

    # Loading representation
    representation = load_representation(dataset_name, "final")

    # Applying PCA if needed
    if apply_pca:
        representation = get_pca_reduction(representation)
        output_subdir = Path(str(output_dir) + "_PCA")
        output_file = output_subdir / dataset_name / f"tsne+pca_edges_{k_tsne}_{ppxty}_{init}.pkl"
    else:
        output_subdir = output_dir
        output_file = output_subdir / dataset_name / f"tsne_edges_{k_tsne}_{ppxty}_{init}.pkl"

    # Generating t-SNE graph
    tsne_graph = get_tsne_graph(representation, k_tsne, ppxty, init, dim, rnd_state)

    # Defining filename and save graph
    save_edges(tsne_graph, output_file)
