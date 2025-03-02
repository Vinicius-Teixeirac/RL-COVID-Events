import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler

from utils import save_event_features  # Renamed for consistency

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_event_features(dataset_path: str) -> dict[str, np.ndarray]:
    """
    Extracts event features from a dataset.

    Parameters
    ----------
    dataset_path : str
        The path to the dataset from which to extract event features.
    
    Returns
    -------
    dict[str, np.ndarray]
        A dictionary containing semantic, geospatial, and temporal event features,
        as well as a final scaled feature set for dimensionality reduction.
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
    time_delta = [(d - minimal_date).days for d in dataset['dates']]
    dataset['date_timediff'] = time_delta
    temporal = np.array(dataset[['date_timediff']])

    # Concatenate features
    combined = np.concatenate((semantic_l2, geospatial, temporal), axis=1)
    
    # Scale them to get the final feature set
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

if __name__ == "__main__":
    # Parsing the arguments used for feature extraction
    parser = argparse.ArgumentParser(description="Generates scaled event features from a dataset.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset (Pickle file).")
    parser.add_argument("--output_dir", type=str, default="./DatasetEventFeatures", help="Directory to save the output.")
    args = parser.parse_args()
    
    # Getting the dataset specifications
    dataset_path = args.dataset_path
    dataset_name = Path(dataset_path).stem
    output_dir = args.output_dir
    
    # Extracting and saving event features
    event_features = get_event_features(dataset_path)
    save_event_features(event_features, dataset_name, output_dir)  # Renamed for clarity
