import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler

from Utils import save_event_features

def get_event_features(dataset_path: str) -> dict[str, np.ndarray]:
    """
    Extracts event features from a given event dataset. An event has semantical (the news content), geospatial (the lat-long coordinates)
    and temporal (a timestamp defined by the date an event happened) features.

    Parameters
    ----------
    dataset_path : str
        The path to the dataset from which to extract event features.
    
    Returns
    -------
    dict[str, np.ndarray]
        A dictionary containing semantic, geospatial, and temporal event features,
        as well as a final scaled feature set for dimensionality reduction.

    Raises
    ------
    FileNotFoundError
        If the file given by `dataset_path` is not found.
    """

    if not Path(dataset_path).is_file():
        logging.error(f"File {dataset_path} not found.")
        raise FileNotFoundError(f"File {dataset_path} does not exist.")

    dataset = pd.read_pickle(dataset_path)

    # Semantic features
    semantic = np.array(dataset['embeddings'].to_list())
    semantic_l2 = normalize(semantic, norm="l2")  # L2 normalization for stability in distance measures

    # Geospatial features
    geospatial = np.array(dataset[['country_lat', 'country_lng']])

    # Temporal features
    dataset['dates'] = pd.to_datetime(dataset['dates'])
    minimal_date = min(dataset['dates'])
    time_delta = [(d - minimal_date).total_seconds() / 3600 for d in dataset['dates']]
    temporal = np.array(time_delta, ndmin=2).T

    # Transforming the geospatial features into cartesian coordinates (so they can be scaled)
    lat_rad = np.deg2rad(dataset['country_lat'])
    lng_rad = np.deg2rad(dataset['country_lng'])
    x, y, z = np.cos(lat_rad) * np.cos(lng_rad), np.cos(lat_rad) * np.sin(lng_rad), np.sin(lat_rad)
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    z = np.array(z).reshape(-1, 1)

    combined = np.concatenate((semantic_l2, x, y, z, temporal), axis=1)

    scaler = StandardScaler()
    final = scaler.fit_transform(combined)
    
    # Store features in a dictionary for modular use
    event_features = {
        'semantic': semantic_l2,
        'geospatial': geospatial,
        'temporal': temporal,
        'final': final
    }

    return event_features

def argument_parsing():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generates scaled event features from a dataset.")

    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset (Pickle file).")
    parser.add_argument("--output_dir", type=str, default="./DatasetEventFeatures", help="Directory to save the output.")

    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Parsing the arguments used for feature extraction
    args = argument_parsing()
    
    # Getting the dataset specifications
    dataset_path = args.dataset_path
    dataset_name = Path(dataset_path).stem
    output_dir = args.output_dir
    
    # Extracting and saving event features
    event_features = get_event_features(dataset_path)
    save_event_features(event_features, dataset_name, output_dir)  