import os
from pathlib import Path
import logging
from joblib import Parallel, delayed

import pandas as pd

from Utils import load_edges

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def graph_edges_metrics(edges_predicted: set, ground_truth: set) -> tuple[float, float]:
    """
    Calculates precision and recall for the predicted graph edges against the ground truth.

    Parameters
    ----------
    edges_predicted : set
        The set of edges predicted by a nearest neighbors graph method.
    ground_truth : set
        The set of edges from the reference consistency graph.

    Returns
    -------
    tuple[float, float]
        A tuple containing the precision and recall scores, in that order.
    """
    true_positives = edges_predicted & ground_truth
    false_positives = edges_predicted - ground_truth
    false_negatives = ground_truth - edges_predicted

    # Computes precision: proportion of predicted edges that are correct
    precision = (len(true_positives) / (len(true_positives) + len(false_positives))
                 if (len(true_positives) + len(false_positives)) > 0 else 0)

    # Computes recall: proportion of ground truth edges that were correctly predicted
    recall = (len(true_positives) / (len(true_positives) + len(false_negatives))
              if (len(true_positives) + len(false_negatives)) > 0 else 0)
    return precision, recall

def batch_iter(lst: list, batch_size: int):
    """Yields successive batches of size `batch_size` from list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def compare_graphs( reference_folder: str,
                    comparison_folder: str,
                    method: str,
                    output_dir: str,
                    n_jobs: int = -1,
                    ref_batch_size: int = 15,
                    comp_batch_size: int = 15) -> None:
    """
    Compares graphs from a reference folder with graphs from a comparison folder in batches,
    calculating precision and recall for each pair. Results are saved as a Parquet file.
    
    Parameters:
    ----------
      reference_folder:  str 
            Path to the ground truth graphs.

      comparison_folder: str 
            Path to the method-generated graphs.

      method: str 
            The name of the method (e.g., PCA, TSNE, UMAP).

      output_dir: str 
            Directory to save evaluation results.

      n_jobs: int, optional 
            Number of parallel jobs to use (-1 uses all available cores).

      ref_batch_size: int, optional 
            Number of reference files to load per batch.

      comp_batch_size: int, optional 
            Number of comparison files to load per batch.
    """
    # Validate directories
    ref_folder_path = Path(reference_folder)
    comp_folder_path = Path(comparison_folder)
    if not ref_folder_path.is_dir():
        raise FileNotFoundError(f"Reference folder '{reference_folder}' does not exist.")
    if not comp_folder_path.is_dir():
        raise FileNotFoundError(f"Comparison folder '{comparison_folder}' does not exist.")

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get file lists
    reference_files = sorted(os.listdir(reference_folder))
    comparison_files = sorted(os.listdir(comparison_folder))
    
    if not reference_files or not comparison_files:
        logging.warning("No reference or comparison files found. Exiting.")
        return

    overall_results = []

    # Batch over reference files
    for ref_batch in batch_iter(reference_files, ref_batch_size):
        # Preload the current batch of reference files
        ref_edges_dict = {}
        for ref_file in ref_batch:
            ref_path = ref_folder_path / ref_file
            if not ref_path.exists():
                logging.warning(f"Reference file {ref_path} does not exist. Skipping.")
                continue
            ref_edges_dict[ref_file] = load_edges(str(reference_folder), ref_file)
        
        if not ref_edges_dict:
            continue

        # Batch over comparison files for each reference batch
        for comp_batch in batch_iter(comparison_files, comp_batch_size):
            comp_edges_dict = {}
            for comp_file in comp_batch:
                comp_path = comp_folder_path / comp_file
                if not comp_path.exists():
                    logging.warning(f"Comparison file {comp_path} does not exist. Skipping.")
                    continue
                comp_edges_dict[comp_file] = load_edges(str(comparison_folder), comp_file)
            
            if not comp_edges_dict:
                continue

            # Build tasks for every combination within the current batch pair
            tasks = [(ref_file, comp_file) for ref_file in ref_edges_dict for comp_file in comp_edges_dict]

            def compare_using_cache(task):
                ref_file, comp_file = task
                ref_edges = ref_edges_dict[ref_file]
                comp_edges = comp_edges_dict[comp_file]
                precision, recall = graph_edges_metrics(comp_edges, ref_edges)
                logging.info(
                    f"Compared {ref_file} with {comp_file}: Precision={precision:.2f}, Recall={recall:.2f}"
                )
                return {
                    "Reference Graph": ref_file,
                    "Compared Graph": comp_file,
                    "Precision": precision,
                    "Recall": recall
                }

            # Run comparisons in parallel for the current batch
            batch_results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(compare_using_cache)(task) for task in tasks
            )
            overall_results.extend(batch_results)
            # Free up memory used by the comparison batch
            comp_edges_dict.clear()
        
        # Clear the reference batch from memory once all its comparisons are done
        ref_edges_dict.clear()

    # Save overall results if available
    if overall_results:
        structured_results = pd.DataFrame(overall_results)
        results_file = output_path / f"comparison_results_{method}.parquet"
        structured_results.to_parquet(results_file, index=False)
        logging.info(f"Comparison results saved to {results_file}")
    else:
        logging.warning("No valid comparisons were performed; no results to save.")
