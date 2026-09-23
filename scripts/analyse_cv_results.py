# Code Summary
"""
This script analyses the results of hyperparameter tuning experiments for LSTM models.

It includes functions to:
    - combine results from multiple experiments,
    - check for duplicates,
    - calculate descriptive statistics,
    - print summary statistics, and
    - identify the best performing models based on specified metrics.
    
"""

# Imports
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_base_path, resolve_path, get_filepaths
from src.analysis.summary import (
    check_for_duplicates,
    print_summary_stats,
    calc_descriptive_stats,
    print_best_models,
)

CONFIG = {
    "top_n_models": 5,
    "check_duplicates": True,
    "print_best_performing_models": True,
}
    
def parse_args():
    """Parses command line arguments to analyse hyperparameter tuning results."""
    parser = argparse.ArgumentParser(
        description="Analyse hyperparameter tuning results for LSTM models."
    )
    parser.add_argument(
        "--pipeline-type",
        type=str,
        required=True,
        choices=["static", "end2end"],
        default="static",
        help="Type of pipeline to analyse: 'static' for static features or 'end2end' for end-to-end features."
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/hyperparam_tuning_analysis",
        help="Directory to save the analysis results."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    base_path = get_base_path()

    results_dirs = {
        "static": resolve_path(base_path, "experiments/hyperparam_tuning_results"),
        "end2end": resolve_path(base_path, "experiments/fine_tuning_results"),
    }

    directories = {
        "input": results_dirs[args.pipeline_type],
        "output": resolve_path(base_path, args.output_dir) / f"{args.pipeline_type}_analysis",
    }

    if not directories["input"].exists():
        raise FileNotFoundError(f"Input directory does not exist: {directories['input']}")
    directories["output"].mkdir(parents=True, exist_ok=True)

    csv_files = get_filepaths(
        directories, 
        folder_to_process="cv_results"
    )

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the input directory: {directories['input']}")

    combined_csv_path = directories["output"] / f"cv_results_{args.pipeline_type}.csv"

    merged_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    merged_df.sort_values(by="macro_f1", ascending=False, inplace=True)

    merged_df.to_csv(combined_csv_path, index=False)

    if CONFIG["check_duplicates"]:
        merged_df = check_for_duplicates(merged_df)

    print_summary_stats(merged_df)

    if CONFIG["print_best_performing_models"]:
        for dataset in merged_df["dataset_type"].unique():
            for model_type in merged_df["model_type"].unique():
                mean = calc_descriptive_stats(merged_df, measure="macro_f1", dataset_type=dataset, model_type=model_type)
    
                if np.isnan(mean):
                    continue

                print_best_models(merged_df, measure="macro_f1", n=CONFIG["top_n_models"], dataset_type=dataset, model_type=model_type)

if __name__ == "__main__":
    main()

    

    

    
    

    

    

