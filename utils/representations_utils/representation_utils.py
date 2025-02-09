import Path
import pickle
from typing import Union

import numpy as np

def save_representations(representations: dict[str, np.ndarray], output_dir: str, dataset_name: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = f"{output_dir}/{dataset_name}_representations.pkl"

    with open(output_file, 'wb') as f:
        pickle.dump(representations, f)
    
    print(f"Representations saved to {output_file}")


def load_representation(dataset_path: str, target: str = None) -> Union[dict, np.ndarray]:
    dataset_name = Path(dataset_path).stem
    representations_path = f"./DatasetRepresentations/{dataset_name}_representations.pkl"

    with open(representations_path, 'rb') as f:
        representations = pickle.load(f)

    if target:
        if target not in representations:
            raise ValueError(f"Target representation '{target}' not found. Available keys: {list(representations.keys())}")
        return representations[target]
    return representations

