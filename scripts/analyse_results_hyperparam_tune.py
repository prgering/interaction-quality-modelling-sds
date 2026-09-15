# Code Summary
"""
This script analyzes the results of hyperparameter tuning experiments for LSTM models.

It includes functions to:
    - parse filenames for hyperparameter settings, 
    - read and process results files,
    - combine results from multiple experiments,
    - visualize performance metrics,
    - perform regression analysis to understand the impact of different hyperparameters on model performance. 
    
The script generates various plots and saves combined results to CSV files.
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
from src.analysis.plots import create_box_plots

CONFIG = {
    "top_n_models": 5,
    "log_f1_constant": 1e-6,
    "check_duplicates": True,
    "print_best_performing_models": True,
    "create_box_plots": True,
}
    
def parse_args():
    """Parses command line arguments to analyse hyperparameter tuning results."""
    parser = argparse.ArgumentParser(
        description="Analyse hyperparameter tuning results for LSTM models."
    )
    parser.add_argument(
        "--input-dir",
        default="experiments/hyperparam_tuning_results",
        help="Directory containing the hyperparameter tuning results."
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

    directories = {
        "input": resolve_path(base_path, args.input_dir),
        "output": resolve_path(base_path, args.output_dir),
    }

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    csv_files = get_filepaths(
        directories["input"], 
        folder_to_process="hyperparam_tuning_results"
    )

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the input directory: {directories['input']}")

    combined_csv_path = directories["output"] / "combined_tuning_results.csv"
    graph_dir = directories["output"] / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    merged_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    merged_df.sort_values(by="macro_f1", ascending=False, inplace=True)

    merged_df.to_csv(combined_csv_path, index=False)

    merged_df = check_for_duplicates(merged_df)

    print_summary_stats(merged_df)

    if CONFIG["print_best_performing_models"]:
        for dataset in merged_df["dataset_type"].unique():
            for model_type in merged_df["model_type"].unique():
                    mean = calc_descriptive_stats(merged_df, measure="f1", dataset_type=dataset, model_type=model_type)
        
                    if np.isnan(mean):
                        continue

                    print_best_models(merged_df, measure="f1", n=CONFIG["top_n_models"], dataset_type=dataset, model_type=model_type)
    
    # Create box plots for visualization
    if CONFIG["create_box_plots"]:
        create_box_plots(merged_df, output_dir=graph_dir)

if __name__ == "__main__":
    main()

    

    

    
    

    

    

