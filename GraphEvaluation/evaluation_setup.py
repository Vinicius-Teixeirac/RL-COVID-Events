import logging
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from Utils import load_edges

def graph_edges_metrics(edges_predicted: set, ground_truth: set) -> tuple[float, float]:
    """
    Calculates precision and recall for the predicted graph edges against the ground truth.
    """
    true_positives = edges_predicted & ground_truth
    false_positives = edges_predicted - ground_truth
    false_negatives = ground_truth - edges_predicted

    precision = (len(true_positives) / (len(true_positives) + len(false_positives))
                 if (len(true_positives) + len(false_positives)) > 0 else 0)
    recall = (len(true_positives) / (len(true_positives) + len(false_negatives))
              if (len(true_positives) + len(false_negatives)) > 0 else 0)
    return precision, recall

def compare_pair(ref_folder: str, comp_folder: str, ref_file: str, comp_file: str) -> dict | None:
    """
    Loads the reference and comparison files on demand, computes metrics for the pair,
    and returns a dictionary with the results. If either file is missing, logs a warning and returns None.
    """
    ref_path = Path(ref_folder) / ref_file
    comp_path = Path(comp_folder) / comp_file

    if not ref_path.exists():
        logging.warning(f"Reference file {ref_path} does not exist. Skipping.")
        return None
    if not comp_path.exists():
        logging.warning(f"Comparison file {comp_path} does not exist. Skipping.")
        return None

    # Load files on demand
    ref_edges = load_edges(ref_folder, ref_file)
    comp_edges = load_edges(comp_folder, comp_file)

    precision, recall = graph_edges_metrics(comp_edges, ref_edges)
    logging.info(f"Compared {ref_file} with {comp_file}: Precision={precision:.2f}, Recall={recall:.2f}")

    return {
        "Reference Graph": ref_file,
        "Compared Graph": comp_file,
        "Precision": precision,
        "Recall": recall
    }

def compare_graphs(reference_folder: str, comparison_folder: str, method: str, output_dir: str, n_jobs: int = -1) -> None:
    """
    Compares all pairs of graphs (each reference file with each comparison file) by:
      - Loading each file on demand.
      - Scheduling each reference–comparison pair as an individual task.
    
    The results are saved as a Parquet file.
    """
    ref_dir = Path(reference_folder)
    comp_dir = Path(comparison_folder)

    if not ref_dir.is_dir():
        raise FileNotFoundError(f"Reference folder '{reference_folder}' does not exist.")
    if not comp_dir.is_dir():
        raise FileNotFoundError(f"Comparison folder '{comparison_folder}' does not exist.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    reference_files = sorted([p.name for p in ref_dir.iterdir() if p.is_file()])
    comp_files = sorted([p.name for p in comp_dir.iterdir() if p.is_file()])

    if not reference_files or not comp_files:
        logging.warning("No reference or comparison files found. Exiting.")
        return

    # Create a task for each reference-comparison pair.
    tasks = [
        delayed(compare_pair)(str(ref_dir), str(comp_dir), ref_file, comp_file)
        for ref_file in reference_files
        for comp_file in comp_files
    ]

    # Execute tasks in parallel.
    all_results = Parallel(n_jobs=n_jobs, backend='loky')(tasks)

    # Filter out any None results.
    results = [result for result in all_results if result is not None]

    if results:
        df = pd.DataFrame(results)
        results_file = Path(output_dir) / f"comparison_results_{method}.parquet"
        df.to_parquet(results_file, index=False)
        logging.info(f"Comparison results saved to {results_file}")
    else:
        logging.warning("No valid comparisons were performed; no results to save.")
