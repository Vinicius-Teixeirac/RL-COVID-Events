import pickle
import logging
from typing import Union
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_representations(representations: dict[str, np.ndarray], dataset_name: str, output_dir: str) -> None:
    """
    Saves multiple representations to a pickle file.

    Parameters
    ----------
    representations : dict[str, np.ndarray]
        Dictionary containing different representations (e.g., semantic, geospatial, temporal, and final).
    dataset_name : str
        Name of the dataset associated with the representations.
    output_dir : str
        Directory where the representations will be saved.

    Returns
    -------
    None
    """
    if not representations:
        logging.error("The representations dictionary is empty. Nothing to save.")
        raise ValueError("The representations dictionary is empty. Nothing to save.")

    output_path = Path(output_dir) / f"{dataset_name}_representations.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, 'wb') as f:
            pickle.dump(representations, f)
        logging.info(f"Representations successfully saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save representations to {output_path}: {e}")
        raise 


def load_representation(dataset_name: str, target: str = None, path: str = './DatasetRepresentations') -> Union[dict[str, np.ndarray], np.ndarray]:
    """
    Loads representations from a pickle file.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset for which the representations are saved.
    target : str, optional
        Specific representation key to load (e.g., 'semantic', 'geospatial', 'temporal', or 'final').
    path : str, optional
        Path to the directory containing the saved representations. Default is './DatasetRepresentations'.

    Returns
    -------
    Union[dict[str, np.ndarray], np.ndarray]
        If `target` is provided, returns the corresponding numpy array.
        Otherwise, returns a dictionary with all representations.
    """
    representations_path = Path(path) / f"{dataset_name}_representations.pkl"

    if not representations_path.exists():
        logging.error(f"Representation file not found at {representations_path}")
        raise FileNotFoundError(f"Representation file not found at {representations_path}")

    with open(representations_path, 'rb') as f:
        representations = pickle.load(f)

    logging.info(f"Successfully loaded representations for dataset: {dataset_name}")

    if target:
        if target not in representations:
            available_keys = ', '.join(representations.keys())
            logging.error(f"Requested representation '{target}' not found. Available keys: {available_keys}")
            raise ValueError(f"Target representation '{target}' not found. Available keys: {available_keys}")
        return representations[target]

    return representations
