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

import ast
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from pathlib import Path
from collections import defaultdict
from datetime import datetime, date, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_base_path, resolve_path, get_filepaths

CONFIG = {
    "top_n_models": 5,
    "log_f1_constant": 1e-6,
    "check_duplicates": True,
    "print_best_performing_models": True,
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

def check_for_duplicates(df):
    """
    Checks for duplicate rows in the DataFrame and prints them if found. Removes them from the DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to check for duplicates.
    Returns:
        pd.DataFrame: The original DataFrame with duplicates removed.
    """

    initial_rows = len(df)
    duplicate_mask = df.duplicated(keep=False)
    duplicate_rows = df[duplicate_mask]

    print("Total number of duplicate rows found:", len(duplicate_rows))

    if not duplicate_rows.empty:
        print("\n--- Duplicate Rows Found ---")
        print(duplicate_rows)

    cleaned_df = df.drop_duplicates()

    print("Number of rows before removing duplicates:", initial_rows)
    print("Number of rows after removing duplicates:", len(cleaned_df))

    return cleaned_df

def print_summary_stats(df):
    """
    Prints summary statistics for the DataFrame, grouped by dataset type.
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
    """
    print("\n--- Summary Statistics by Dataset Type ---\n")
    for dataset in df["dataset_type"].unique():
        dataset_df = df[df["dataset_type"] == dataset]
        print(f"\n--- Summary Statistics for Dataset Type: {dataset} ---\n")
        print(dataset_df.describe(include="all"))


def print_best_models(df, measure, n, dataset_type=None, text_model=None, acoustic_model=None, model_type=None):
    """
    Prints the top n performing models, with optional filters by dataset_type and model_type.

    Args:
        df (pd.DataFrame): DataFrame containing model results.
        measure (str): The column name to sort by (e.g., 'f1_score').
        n (int): The number of top models to display.
        dataset_type (str, optional): The dataset type to filter by. If None, no filtering is applied.
        text_model (str, optional): The text embedding model to filter by. If None, no filtering is applied.
        acoustic_model (str, optional): The speech embedding model to filter by. If None, no filtering is applied.
        model_type (str, optional): The model type to filter by (e.g., 'lstm'). If None, no filtering is applied.
    """
    pd.set_option("display.max_columns", None)

    filtered_df = df.copy()
    filter_description = ""

    # Filter the DataFrame if a dataset_type is specified
    if dataset_type is not None:
        filtered_df = filtered_df[filtered_df["dataset_type"] == dataset_type].copy()
        filter_description += f" on {dataset_type} dataset"

    # Filter the DataFrame if a text_model is specified
    if text_model is not None:
        filtered_df = filtered_df[filtered_df["text_embedding_model"] == text_model].copy()
        filter_description += f" for {text_model} models"
    
    if acoustic_model is not None:
        filtered_df = filtered_df[filtered_df["speech_embedding_model"] == acoustic_model].copy()
        filter_description += f" for {acoustic_model} acoustic models"

    if model_type is not None:
        filtered_df = filtered_df[filtered_df["model_type"] == model_type].copy()
        filter_description += f" for {model_type} models"

    # Print the appropriate heading based on the filters
    if filter_description:
        print(f"\n\n--- Top {n} Performing Models (based on {measure}) {filter_description} ---")
    else:
        print(f"\n\n--- Top {n} Performing Models (based on {measure}) ---")

    # Sort the filtered data and print the top n rows
    top_n_models = filtered_df.sort_values(by=measure, ascending=False).head(n)
    print(top_n_models)

    pd.reset_option("display.max_columns")

def calc_descriptive_stats(df, measure, dataset_type=None, text_model=None, acoustic_model=None, model_type=None):
    """
    Calculates and prints descriptive statistics (max, min, range) for the specified measure,
    """
    pd.set_option("display.max_columns", None)

    filtered_df = df.copy()
    filter_description = ""

    # Filter the DataFrame if a dataset_type is specified
    if dataset_type is not None:
        filtered_df = filtered_df[filtered_df["dataset_type"] == dataset_type].copy()
        filter_description += f" on {dataset_type} dataset"

    # Filter the DataFrame if a text_model is specified
    if text_model is not None:
        filtered_df = filtered_df[filtered_df["text_embedding_model"] == text_model].copy()
        filter_description += f" for {text_model} models"
    
    if acoustic_model is not None:
        filtered_df = filtered_df[filtered_df["speech_embedding_model"] == acoustic_model].copy()
        filter_description += f" for {acoustic_model} acoustic models"

    if model_type is not None:
        filtered_df = filtered_df[filtered_df["model_type"] == model_type].copy()
        filter_description += f" for {model_type} models"

    summary_stats = filtered_df[measure].describe()

    mean = summary_stats["mean"]

    if mean is None or np.isnan(mean):
        return np.nan

    if filter_description:
        print(f"\n\n--- Descriptive Statistics for {measure} {filter_description} ---")
    else:
        print(f"\n\n--- Descriptive Statistics for {measure} ---")

    print(summary_stats)

    return mean


def create_box_plots(df, output_dir = None):
    """
    Generates box plots to visualize the distribution of Macro F1 scores from hyperparameter tuning.
    Args:
        df (pd.DataFrame): DataFrame containing model results.
        output_dir (str): Directory to save the generated plots.
    """
    sns.set_context("paper", font_scale=1.4) 
    sns.set_style("whitegrid")

    df_copy = df.copy()

    feature_order = ["system_auto", "speech_both"]
    model_order = ["svm", "lstm"]
    x_ticks_labels = ["SD", "SA"]
    model_labels = ["SVM", "LSTM"]

    plt.figure(figsize=(8, 5))

    flier_props = dict(marker="o", markersize=4, alpha=0.5, markeredgewidth=0.5)

    ax = sns.boxplot(
        data=df_copy,
        x="dataset_type",
        y="f1",
        hue="model_type",
        order=feature_order,
        hue_order=model_order,
        palette="Greys",
        width=0.7,
        flierprops=flier_props,
        medianprops={"color": "black", "linewidth": 1.5}
    )

    plt.xlabel("Feature Set", fontweight="bold")
    plt.ylabel("Macro F1", fontweight="bold")
    plt.xticks(range(2), x_ticks_labels)
    plt.ylim(0, 0.65)

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles, 
        model_labels, 
        title=None, 
        loc="upper center",
        bbox_to_anchor=(0.5, 1.125),
        ncol=2, 
        frameon=False,
    )

    plt.tight_layout()

    save_path = f"{output_dir}/ht_box_plot.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {save_path}")

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
    create_box_plots(merged_df, output_dir=graph_dir)

if __name__ == "__main__":
    main()

    

    

    
    

    

    

