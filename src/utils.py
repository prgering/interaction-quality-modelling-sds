import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

def set_all_seeds(seed):
    """
    Sets seeds for reproducibility across different libraries.
    
    Args:
        seed (int): The seed value to set.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        print("Warning: PyTorch not found. Skipping PyTorch seed setting.")

def set_rng_state(seed=42):
    """Resets the random number generators for NumPy and PyTorch/CUDA."""
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
        
def get_base_path():
    """
    Finds the project root directory relative to this utils.py
    file.
    """
    return Path(__file__).resolve().parent.parent

def resolve_path(base_path, p):
    """Resolves relative paths to the project root, keeping absolute paths intact."""
    if p is None:
        return None
        
    path_obj = Path(p)
    if path_obj.is_absolute():
        return path_obj.resolve()
    
    repo_relative = base_path / path_obj
    if repo_relative.exists():
        return repo_relative.resolve()
    
    parent_relative = base_path.parent / path_obj
    if parent_relative.exists():
        return parent_relative.resolve()
        
    return repo_relative

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calc_metrics(actual_labels, pred_vals, dataset_type, model_type, print_confusion_matrix=False, base_path=None, top_n = None, data_split = "test", auto_features_only=False, svm_kernel = None):
    """
    Calculates and prints the classification report and confusion matrix for the given predictions and actual labels.
    
    Args:
        actual_labels (np.ndarray or list): The true labels for the data.
        pred_vals (np.ndarray or list): The predicted labels from the model.
        dataset_type (str): The type of dataset used (default is "original").
        print_confusion_matrix (bool): Whether to print the confusion matrix (default is False).
        base_path (Path): The base path for saving the confusion matrix image (if applicable).
        auto_features_only (bool): Whether only automatic features were used.
    
    Returns:
        results (dict): A dictionary containing the classification report metrics.
    """
    # Calculate classification report and convert to a dictionary for return
    results = classification_report(actual_labels, pred_vals, digits=4, output_dict=True, zero_division=0)

    # Print the human-readable classification report
    print(f"Dataset: {dataset_type}\n")
    if auto_features_only:
        print(f"Auto Features Only\n")
    print(f"Classification Report:\n{classification_report(actual_labels, pred_vals, digits=4)}")

    if print_confusion_matrix:
        if base_path is None:
            print("Warning: 'base_path' must be provided to save the confusion matrix.")
        else:

            cm_actual_labels = [int(label) + 1 for label in actual_labels]
            cm_pred_vals = [int(label) + 1 for label in pred_vals]

            parent_folder = base_path / "experiments/output/confusion_matrices"
            os.makedirs(parent_folder, exist_ok=True)

            feature_tag = "_auto" if auto_features_only else ""
            top_n_tag = f"_top{top_n}" if top_n is not None else "_all"

            kernel_tag = f"_{svm_kernel}" if svm_kernel else ""

            display_labels = sorted(list(set(cm_actual_labels) | set(cm_pred_vals)))

            cm_normalized = confusion_matrix(cm_actual_labels, cm_pred_vals, normalize='true')
            disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=display_labels)
            disp_normalized_result = disp_normalized.plot(values_format=".2f", cmap='Blues')

            if hasattr(disp_normalized_result, 'im_'):
                disp_normalized_result.im_.set_clim(0.0, 1.0)
            
                if hasattr(disp_normalized_result, 'colorbar') and disp_normalized_result.colorbar is not None:
                    disp_normalized_result.figure_.colorbar(disp_normalized_result.im_, ax=disp_normalized_result.axes)

            filename_norm = f"cm_{dataset_type}{feature_tag}_{model_type}{kernel_tag}{top_n_tag}_normalized.png"
            filepath_norm = parent_folder / filename_norm
            plt.savefig(filepath_norm, dpi=600)
            plt.close()
            print(f"Normalized confusion matrix saved to: {filepath_norm}")
    
    return results

def optimize_dataframe_memory(df):
    """
    Optimizes a Pandas DataFrame's memory usage by downcasting numerical types
    and converting object columns to 'category' where appropriate.
    
    Args:
        df (pd.DataFrame): The DataFrame to optimize.
    
    Returns:
        pd.DataFrame: The memory-optimized DataFrame.
    """
    initial_memory = df.memory_usage(deep=True).sum() / (1024**3)
    print(f"  Initial DataFrame memory: {initial_memory:.4f} GB", flush = True)

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == object:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            # Heuristic for category conversion
            if num_unique_values / num_total_values < 0.5: 
                df[col] = df[col].astype('category')
        elif 'float' in str(col_type):
            df.loc[:, col] = pd.to_numeric(df[col], downcast='float')
        elif 'int' in str(col_type):
            df.loc[:, col] = pd.to_numeric(df[col], downcast='integer')

    final_memory = df.memory_usage(deep=True).sum() / (1024**3)
    print(f"  Optimized DataFrame memory: {final_memory:.4f} GB (Reduced by {(initial_memory - final_memory) / initial_memory * 100:.4f}%)", flush = True)
    return df


def get_df_size(df):
    """
    Helper function to get and print the size of a DataFrame in GB.
    """
    size_gb = df.memory_usage(deep=True).sum() / (1024**3)
    return size_gb


def load_feature_data(dataset_path, call_id_col, remove_first_turn = False):
    """
    Loads a specified feature DataFrame from the specified path.
    Args:
        dataset_path (str or Path): Path to the dataset CSV file.
        call_id_col (str): Name of the column containing call IDs.
        remove_first_turn (bool): Whether to remove the first turn of each call.
    Returns:
        pd.DataFrame: The loaded and preprocessed DataFrame.
    """
    
    df = pd.read_csv(dataset_path)

    if 'Utterance' in df.columns:
        df['Utterance'] = df['Utterance'].fillna('').astype(str)
    
    if remove_first_turn:
        df['ExchangeNum'] = df.groupby(call_id_col).cumcount() + 1
        df = df[df['ExchangeNum'] > 2].copy()
        print(f"Removed first turns. New DataFrame size: {df.shape}", flush = True)
    
    # Optimize the DataFrame memory usage
    print(f"Size of df: {get_df_size(df):.4f} GB", flush = True)
    print(f"Optimizing df...", flush = True)
    df = optimize_dataframe_memory(df)
    print(f"Size of df (after optimization): {get_df_size(df):.4f} GB\n", flush = True)

    return df


def train_opt_split(df, call_id_col, target_column, train_size=0.6):
    """
    Splits the dataframe into training and optimization sets based on unique call IDs.
    Args:
        df (pd.DataFrame): The input DataFrame to split.
        call_id_col (str): The name of the column containing call IDs.
        target_column (str): The name of the target variable column.
        train_size (float): Proportion of the data to include in the training set.
    Returns:
        pd.DataFrame, pd.DataFrame: The training and optimization DataFrames.
    """

    unique_callids = df[call_id_col].unique()
    train_filecodes, opt_filecodes = train_test_split(unique_callids, train_size=train_size, random_state=42)

    train_df = df[df[call_id_col].isin(train_filecodes)].copy()
    opt_df = df[df[call_id_col].isin(opt_filecodes)].copy()

    return train_df, opt_df