import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize, StandardScaler
import os

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

    return R_f_scaled

if __name__ == "__main__":
    import sys
    dataset = sys.argv[1]
    dataset_name = os.path.basename(dataset)[:-4]
    R_f_scaled = get_representations(dataset)
    output_dir = "./representations"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{output_dir}/{dataset_name}_representations.npy", R_f_scaled)
