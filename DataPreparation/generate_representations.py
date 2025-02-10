import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler

from utils import save_representations

def get_representations(dataset_path: str)-> dict[str, np.ndarray]:
    """
    Generates the representations for each dataset.

    Parameters
    ----------
    dataset_path : str
        The path to the dataset to extract representations from
    
    Returns
    -------
    dict[str, np.array]
        A dict with the semantic, geospatial and temporal representations, also with the final representation for the posterior 
        reduction method
    """
    dataset = pd.read_pickle(dataset_path)

    # sematic - Semantic Representation
    sematic = np.array(dataset['embeddings'].to_list())
    sematic_l2 = normalize(sematic, norm="l2") # l2 normed for stability in distance measures

    # geospatial - Geospatial Representation
    geospatial = np.array(dataset[['country_lat', 'country_lng']])

    # temporal - Temporal Representation
    dataset['dates'] = pd.to_datetime(dataset['dates'])
    minimal_date = min(dataset['dates'])
    time_delta = [(d - minimal_date).days for d in dataset['dates']]
    dataset['date_timediff'] = time_delta
    temporal = np.array(dataset[['date_timediff']])

    # Concatenates representations
    combined = np.concatenate((sematic_l2, geospatial, temporal), axis=1)
    
    # scales then to get the final representation
    scaler = StandardScaler()
    final = scaler.fit_transform(combined)
    
    # saves them in a dictonary to a more modular approach
    representations = {
        'semantic': sematic_l2,
        'geospatial': geospatial,
        'temporal': temporal,
        'final': final
    }

    return representations

if __name__ == "__main__":
    # parsing the arguments that'll be used on the representation extraction
    parser = argparse.ArgumentParser(description="Generates scaled representations from a dataset.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset (Pickle file).")
    parser.add_argument("--output_dir", type=str, default="./DatasetRepresentations", help="Directory to save the output.")
    args = parser.parse_args()
    
    # getting the dataset specifications
    dataset_path = args.dataset_path
    dataset_name = Path(dataset_path).stem
    output_dir = args.output_dir
    
    # Extracting and saving representations
    representations = get_representations(dataset_path)
    save_representations(representations, dataset_name, output_dir)
