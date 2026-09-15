import pandas as pd
import numpy as np

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
