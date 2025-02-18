import pickle
from pathlib import Path
import logging

import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
        logging.error("Attempted to save an empty graph.")
        raise ValueError("The graph has no edges to save.")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(edges, f)

    logging.info(f"Edges saved successfully to {output_path}")

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
    path = Path(folder) / file
    if not path.exists():
        logging.error(f"File not found: {path}")
        raise FileNotFoundError(f"The file '{path}' does not exist.")
    
    try:
        with path.open("rb") as f:
            edges = pickle.load(f)
        return edges
    except EOFError:
        logging.error(f"Corrupted or empty file: {path}")
        raise EOFError(f"The file '{path}' is empty or corrupted.")
