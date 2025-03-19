import pickle
import argparse
import logging

from umap.umap_ import nearest_neighbors

from Utils import load_event_features

def precompute_knn(dataset_name: str, n_neighbors: int, metric: str,  random_state: int, output_path: str):
    """
    Generates a k-nearest neighbors graph for a given dataset.

    This is a optimization step. Since the experiments will consider many `n_neighbors` for UMAP, instead of calculating the k-nearest
    neighbors every time, one can calculate the k-nearest neighbors once for the largest `n_neighbors` and just prune for smaller values.

    Parameters
    ----------
    dataset_name : str
        The dataset to compute k-nn.
    n_neighbors : int
        The number of neighbors to consider while computing k-nn. One may make sure
        that this is the maximum n_neighbors that will appear in experiments.
    metric : str
        The metrics in which UMAP will be calculated
    random_state : int
        Seed for reproductibility.
    output_path : str
        Path to save the knn graph

    Returns
    -------
    None
        The function computes and saves the k-nearest neighbors graph.

    """

    events = load_event_features(dataset_name, 'final')

    knn_graph = nearest_neighbors(
        events,
        n_neighbors=n_neighbors,
        metric=metric,
        metric_kwds=None,
        angular=False,
        random_state=random_state
    )
    # Save precomputed k-NN graph
    with open(output_path, "wb") as f:
        pickle.dump(knn_graph, f)
    print(f"Precomputed k-NN graph saved to {output_path}")


def argument_parsing():
    parser = argparse.ArgumentParser(description="Compute and save precomputed k-NN graph for Manifold methods")

    parser.add_argument('--dataset_name', type=str, required=True,
                        help="The dataset to compute the k-nearest neighbors")
    parser.add_argument('--max_neighbors', type=int, required=True,
                        help="The maximum n_neighbors that will appear in experiments")
    parser.add_argument('--metric', type=str, choices=['euclidean', 'cosine'], required=True,
                        help="The metrics in which UMAP will be calculated")
    parser.add_argument('--random_state', type=int, default=42,
                        help="seed for reproductibility")
    parser.add_argument('--output_path', type=str, required=True, 
                        help="The path to save the knn graph") # aqui não estou seguindo o padrão de projeto porque deixei o output sendo criando fora do .py, mas vou me preocupar com isso dps

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Parsing the arguments used for knn 
    args = argument_parsing()

    precompute_knn(args.dataset_name, args.max_neighbors, args.metric, args.random_state, args.output_path)