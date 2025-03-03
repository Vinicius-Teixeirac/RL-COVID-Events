import pickle
import logging
from typing import Union
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_event_features(event_features: dict[str, np.ndarray], dataset_name: str, output_dir: str) -> None:
    """
    Saves multiple event features to a pickle file.

    Parameters
    ----------
    event_features : dict[str, np.ndarray]
        Dictionary containing different event features (e.g., semantic, geospatial, temporal, and final).
    dataset_name : str
        Name of the dataset associated with the event features.
    output_dir : str
        Directory where the event features will be saved.

    Returns
    -------
    None
    """
    if not event_features:
        logging.error("The event_features dictionary is empty. Nothing to save.")
        raise ValueError("The event_features dictionary is empty. Nothing to save.")

    output_path = Path(output_dir) / f"{dataset_name}_event_features.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, 'wb') as f:
            pickle.dump(event_features, f)
        logging.info(f"Event features successfully saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save event features to {output_path}: {e}")
        raise


def load_event_features(dataset_name: str, target: str = None, path: str = './DatasetEventFeatures') -> Union[dict[str, np.ndarray], np.ndarray]:
    """
    Loads event features from a pickle file.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset for which the event features are saved.
    target : str, optional
        Specific event feature key to load (e.g., 'semantic', 'geospatial', 'temporal', or 'final').
    path : str, optional
        Path to the directory containing the saved event features. Default is './DatasetEventFeatures'.

    Returns
    -------
    Union[dict[str, np.ndarray], np.ndarray]
        If `target` is provided, returns the corresponding numpy array.
        Otherwise, returns a dictionary with all event_features.
    """
    features_path = Path(path) / f"{dataset_name}_event_features.pkl"

    if not features_path.exists():
        logging.error(f"Event features file not found at {features_path}")
        raise FileNotFoundError(f"Event features file not found at {features_path}")

    with open(features_path, 'rb') as f:
        event_features = pickle.load(f)

    logging.info(f"Successfully loaded event features for dataset: {dataset_name}")

    if target is not None:
        if target not in event_features:
            available_keys = ', '.join(event_features.keys())
            logging.error(f"Requested event feature '{target}' not found. Available keys: {available_keys}")
            raise ValueError(f"Target event feature '{target}' not found. Available keys: {available_keys}")
        return event_features[target]

    return event_features
