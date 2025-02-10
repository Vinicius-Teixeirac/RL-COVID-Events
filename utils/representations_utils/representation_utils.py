import Path
import pickle
from typing import Union

import numpy as np

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
        The function saves the representations as a pickle file in the specified directory.

    Raises
    ------
    ValueError
        If the `representations` dictionary is empty.
    """
    if not representations:
        raise ValueError("The representations dictionary is empty. Nothing to save.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = f"{output_dir}/{dataset_name}_representations.pkl"

    with open(output_file, 'wb') as f:
        pickle.dump(representations, f)
    
    print(f"Representations saved to {output_file}")



def load_representation(dataset_name: str, target: str = None, path: str = './DatasetRepresentations') -> Union[dict[str, np.ndarray], np.ndarray]:
    """
    Loads representations from a pickle file.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset for which the representations are saved.
    target : str, optional
        Specific representation key to load (e.g., 'semantic', 'geospatial', 'temporal', or 'final').
        If not provided, all representations are returned as a dictionary.
    path : str, optional
        Path to the directory containing the saved representations. Default is './DatasetRepresentations'.

    Returns
    -------
    Union[dict[str, np.ndarray], np.ndarray]
        If `target` is provided, returns the corresponding numpy array.
        Otherwise, returns a dictionary with all representations.

    Raises
    ------
    ValueError
        If the target representation is not found in the saved representations.
    FileNotFoundError
        If the representation file does not exist.
    """
    representations_path = f"{path}/{dataset_name}_representations.pkl"

    if not Path(representations_path).exists():
        raise FileNotFoundError(f"Representation file not found at {representations_path}")

    with open(representations_path, 'rb') as f:
        representations = pickle.load(f)

    if target:
        if target not in representations:
            available_keys = ', '.join(representations.keys())
            raise ValueError(f"Target representation '{target}' not found. Available keys: {available_keys}")
        return representations[target]
    
    return representations
