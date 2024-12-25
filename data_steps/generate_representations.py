import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize, StandardScaler
import os
import argparse


def get_representations(dataset_name):
    dataset = pd.read_pickle(dataset_name)

    # R_s - Semantic Representation
    R_s = np.array(dataset['embeddings'].to_list())
    R_s_norm = normalize(R_s, norm="l2")

    # R_g - Geospatial Representation
    R_g = np.array(dataset[['country_lat', 'country_lng']])

    # R_t - Temporal Representation
    dataset['dates'] = pd.to_datetime(dataset['dates'])
    
    minimal_date = min(dataset['dates'])

    time_delta = [(d - minimal_date).days for d in dataset['dates']]
    
    dataset['date_timediff'] = time_delta
    
    R_t = np.array(dataset[['date_timediff']])

    # Concatenate and scale
    R_f = np.concatenate((R_s_norm, R_g, R_t), axis=1)
    scaler = StandardScaler()
    R_f_scaled = scaler.fit_transform(R_f)

    return R_s_norm, R_g, R_t, R_f_scaled


# main so i can run it independently 
"""
in windows i've did 
foreach ($file in Get-ChildItem -Path "data_steps\resultant_datasets\*") {
     echo "Processing file: $file"
     python3 data_steps\get_representations.py $file  
}
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scaled representations from a dataset.")
    parser.add_argument("--dataset", type=str, help="Path to the dataset (Pickle file).")
    parser.add_argument("--output_dir", type=str, default="./datasets'_representations", help="Directory to save the output.")
    args = parser.parse_args()
    
    dataset_path = args.dataset
    output_dir = args.output_dir

    dataset_name = os.path.basename(dataset_path)[:-4]
    _, _, _, R_f_scaled = get_representations(dataset_path)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{output_dir}/{dataset_name}_representations.npy", R_f_scaled)
