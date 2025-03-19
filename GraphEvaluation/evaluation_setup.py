import logging
from pathlib import Path
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

def compare_reference_to_comparisons(ref_folder: str, comp_folder: str, ref_file: str, comp_files: list) -> list:
    """
    Loads the reference file once and compares it against a list of comparison files.
    Returns a list of dictionaries with the comparison results.
    """
    results = []
    ref_path = Path(ref_folder) / ref_file
    if not ref_path.exists():
        logging.warning(f"Reference file {ref_path} does not exist. Skipping.")
        return results

    # Load the reference edges once
    ref_edges = load_edges(ref_folder, ref_file)

    for comp_file in comp_files:
        comp_path = Path(comp_folder) / comp_file
        if not comp_path.exists():
            logging.warning(f"Comparison file {comp_path} does not exist. Skipping.")
            continue

        comp_edges = load_edges(comp_folder, comp_file)
        precision, recall = graph_edges_metrics(comp_edges, ref_edges)
        logging.info(f"Compared {ref_file} with {comp_file}: Precision={precision:.2f}, Recall={recall:.2f}")

        results.append({
            "Reference Graph": ref_file,
            "Compared Graph": comp_file,
            "Precision": precision,
            "Recall": recall
        })
    return results


def compare_graphs(reference_folder: str, comparison_folder: str, method: str, output_dir: str, n_jobs: int = -1) -> None:
    """
    Compares graphs by grouping all comparisons for each reference file, thereby:
      1. Reducing redundant I/O (loading the same reference file only once).
      2. Minimizing scheduling overhead by creating fewer, larger tasks.
    
    Saves the results as a Parquet file.
    """
    ref_dir = Path(reference_folder)
    comp_dir = Path(comparison_folder)

    if not ref_dir.is_dir():
        raise FileNotFoundError(f"Reference folder '{reference_folder}' does not exist.")
    if not comp_dir.is_dir():
        raise FileNotFoundError(f"Comparison folder '{comparison_folder}' does not exist.")

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # List files using pathlib
    reference_files = sorted([p.name for p in ref_dir.iterdir() if p.is_file()])
    comp_files = sorted([p.name for p in comp_dir.iterdir() if p.is_file()])

    if not reference_files or not comp_files:
        logging.warning("No reference or comparison files found. Exiting.")
        return

    # Create one task per reference file
    all_results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(compare_reference_to_comparisons)(str(ref_dir), str(comp_dir), ref_file, comp_files)
        for ref_file in reference_files
    )

    # Flatten the list of lists into a single list of results
    results = [result for sublist in all_results for result in sublist]

    if results:
        df = pd.DataFrame(results)
        results_file = Path(output_dir) / f"comparison_results_{method}.parquet"
        df.to_parquet(results_file, index=False)
        logging.info(f"Comparison results saved to {results_file}")
    else:
        logging.warning("No valid comparisons were performed; no results to save.")