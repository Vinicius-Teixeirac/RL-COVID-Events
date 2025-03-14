import pickle
import argparse

from umap.umap_ import nearest_neighbors

from Utils import load_event_features


# vou ter que melhorar muita coisa, especialmente o logging. Colocar TODAS AS INFORMAÇÕES (n_neighbors, random state e output path no logging)

def precompute_knn(dataset_name: str, n_neighbors: int, random_state: int, output_path: str):
    # Load representation (assumes your Utils.load_event_features handles this)
    representation = load_event_features(dataset_name, 'final')
    # Compute k-NN using the maximum n_neighbors
    knn_graph = nearest_neighbors(
        representation,
        n_neighbors=n_neighbors,
        metric="euclidean",
        metric_kwds=None,
        angular=False,
        random_state=random_state
    )
    # Save precomputed k-NN graph
    with open(output_path, "wb") as f:
        pickle.dump(knn_graph, f)
    print(f"Precomputed k-NN graph saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute and save precomputed k-NN graph")
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--max_neighbors', type=int, required=True)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    max_neighbors = args.max_neighbors + 1
    random_state = args.random_state
    output_path = args.output_path
    precompute_knn(dataset_name, max_neighbors, random_state, output_path)

