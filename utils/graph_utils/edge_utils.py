import os
import pickle

import networkx as nx

def save_edges(graph: nx.DiGraph, output_path: str) -> None:
    edges = set(graph.edges())
    with open(output_path, 'wb') as f:
        pickle.dump(edges, f)
    print(f"Edges saved successfully to {output_path}", flush=True)

def load_edges(folder: str, file: str) -> set:
    path = os.path.join(folder, file)
    with open(path, 'rb') as f:
        edges = pickle.load(f)
    return edges