import logging
import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from generate_tsne_graphs import get_tSNE_graph
from utils import load_representation, save_edges

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_pca_reduction(representation: np.ndarray, dimension: int = 50) -> np.ndarray:
    """
    Reduces the dimension of the initial representation, which may suppress irrelevant variance (noise) and lower
    t-SNE computation time.

     Parameters
    ----------
    representation : np.ndarray
        The input data representation (high-dimensional).
    dim : int
        The lower dimension to be given to t-SNE.

    Returns
    -------
    np.ndarray
        The initial representation after PCA be applied.

    Raises
    ------
    ValueError
        If `dim` is not positive integers, or if `dim` is greater than the number of features in `representation`.
    """
    if dimension <= 0:
        logging.error("Non positive value for dimension")
        raise ValueError("Dimension must be positive integers.")

    if dimension > representation.shape[1]:
        logging.error("Inconsistent dimension")
        raise ValueError(f"dimension ({dimension}) cannot be greater than the number of features in representation ({representation.shape[1]}).")

    return PCA(n_components=dimension).fit_transform(representation)

if __name__ == "__main__":
    # parsing the arguments that'll be used on the t-SNE method
    parser = argparse.ArgumentParser(description ='Generate the t-SNE graphs')
    parser.add_argument('--hyperparameters',  type=int, nargs=4, help='''Four intergers arguments: 
                        Number of neighbors in t-SNE graph, perplexity, desired dimensionality, and random state''')
    parser.add_argument('--initialization', type=str, help='The initial point positions to be used in the embedding space.')
    parser.add_argument('--dataset_path', type=str, help='The path for the current dataset file')
    parser.add_argument("--output_dir", type=str, default="./GeneratedGraphs/PCA+TSNE", help="Directory to save the output.")
    args = parser.parse_args()

    # getting the dataset specifications
    dataset_path = args.dataset_path
    dataset_name = Path(dataset_path).stem
    output_dir = args.output_dir

    # acquiring from arguments the method's hyperparameters
    k_tsne, ppxty, dim, rnd_state = args.hyperparameters
    init = args.initialization

    # loading the representation, reducing its amount of features and obtaining the pca + t-SNE graph from it
    final_representation = load_representation(dataset_name, 'final')
    pca_reduction = get_pca_reduction(final_representation)
    pca_tsne_graph = get_tSNE_graph(pca_reduction, k_tsne, ppxty, init, dim, rnd_state)

    # defining the file's name and saving the representation
    output_file = Path(output_dir) / dataset_name / f"pca+tsne_edges_{k_tsne}_{ppxty}_{init}.pkl"
    save_edges(pca_tsne_graph, output_file)
