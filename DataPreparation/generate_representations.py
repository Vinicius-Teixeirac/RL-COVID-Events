import os
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler

def get_representations(dataset_path):
    dataset = pd.read_pickle(dataset_path)

    # sematic - Semantic Representation
    sematic = np.array(dataset['embeddings'].to_list())
    sematic_l2 = normalize(sematic, norm="l2")

    # geospatial - Geospatial Representation
    geospatial = np.array(dataset[['country_lat', 'country_lng']])

    # temporal - Temporal Representation
    dataset['dates'] = pd.to_datetime(dataset['dates'])
    minimal_date = min(dataset['dates'])
    time_delta = [(d - minimal_date).days for d in dataset['dates']]
    dataset['date_timediff'] = time_delta
    temporal = np.array(dataset[['date_timediff']])

    # Concatenate representations
    combined = np.concatenate((sematic_l2, geospatial, temporal), axis=1)
    
    # scale then to get the final representation
    scaler = StandardScaler()
    final = scaler.fit_transform(combined)
    
    # save them in a dictonary to a more modular approach (i think)
    representations = {
        'semantic': sematic_l2,
        'geospatial': geospatial,
        'temporal': temporal,
        'final': final
    }

    return representations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scaled representations from a dataset.")
    parser.add_argument("--dataset", type=str, help="Path to the dataset (Pickle file).")
    parser.add_argument("--output_dir", type=str, default="./DatasetRepresentations", help="Directory to save the output.")
    args = parser.parse_args()
    
    dataset_path = args.dataset
    output_dir = args.output_dir

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    representations = get_representations(dataset_path)
        
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = f"{output_dir}/{dataset_name}_representations.pkl"

    with open(output_file, 'wb') as f:
        pickle.dump(representations, f)

    print(f"Representations saved to {output_file}")