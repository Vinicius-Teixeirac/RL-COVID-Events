import os
import pickle
from pathlib import Path

import networkx as nx

def save_edges(graph: nx.DiGraph, output_path: str) -> None:
    """
    Saves the edges of a directed graph to a pickle file.

    Parameters
    ----------
    graph : nx.DiGraph
        The directed graph from which edges will be saved.
    output_path : str
        The file path where the edges will be saved.

    Returns
    -------
    None
        The function saves the edges as a pickle file.

    Raises
    ------
    ValueError
        If the graph has no edges to save.
    """
    edges = set(graph.edges())
    if not edges:
        raise ValueError("The graph has no edges to save.")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(edges, f)
    print(f"Edges saved successfully to {output_path}", flush=True)


def load_edges(folder: str, file: str) -> set:
    """
    Loads edges from a pickle file.

    Parameters
    ----------
    folder : str
        The folder containing the pickle file.
    file : str
        The specific file to load.

    Returns
    -------
    set
        A set of edges loaded from the file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    EOFError
        If the file is empty or corrupted.
    """
    path = os.path.join(folder, file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file '{path}' does not exist.")
    
    with open(path, 'rb') as f:
        try:
            edges = pickle.load(f)
        except EOFError:
            raise EOFError(f"The file '{path}' is empty or corrupted.")
    
    return edges
