"""
Trains interaction quality classifier via 10-fold grouped cross-validation and hyperparameter
tuning. Hyperparameters are tuned using a grid search approach. The best model is selected based
on the macro F1 score.

The following hyperparameters are tuned:
    - Hidden size
    - Number of layers
    - Bidirectionality
    - Use of attention mechanism
    - Batch size
    - Learning rate
    - PCA explained variance ratio

The results are saved to a CSV file in the specified output directory.
"""

#imports
import sys
import argparse
from pathlib import Path
import pandas as pd
import torch
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cli_args import add_feature_set_args, add_pca_args, add_lstm_args
from src.models.data_loader import load_processed_data
from src.models.lstm_trainer import LstmManager
from src.utils import set_all_seeds, get_base_path, resolve_path


CONFIG = {
    "debug_limit": 500,
}

def parse_args():
    """Parses command line arguments for LSTM hyperparameter tuning."""
    parser = argparse.ArgumentParser(
        description="Tune LSTM hyperparameters for interaction quality prediction."
    )

    add_feature_set_args(parser)
    add_pca_args(parser)
    add_lstm_args(parser)

    parser.add_argument(
        "--input-dir",
        default="data/processed/preprocessed_features",
        help="Directory containing the extracted features."
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/hyperparam_tuning_results",
        help="Directory to save the hyperparameter tuning results."
    )
    parser.add_argument(
        "--seed", 
        default=42, 
        type=int, 
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to test code on a small subset of data."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_all_seeds(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_path = get_base_path()

    directories = {
        "input": resolve_path(base_path, args.input_dir),
        "output": resolve_path(base_path, args.output_dir),
    }

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    output_file_path = directories["output"] / f"tuning_results_{task_id}.csv"

    # Load preprocessed feature set for training and hyperparameter tuning
    train_df = load_processed_data(
        input_dir = directories["input"],
        args = args
    )

    if args.debug:
        train_df = train_df.head(CONFIG["debug_limit"])

    # Initialize the LSTM manager and run hyperparameter tuning
    trainer = LstmManager(train_df=train_df, args=args,device=device)

    results_dict = trainer.run_hyperparam_tuning()   

    records = []

    for key, results in results_dict.items():
        row = {
            **results["params"],
            "macro_recall": results["recall"],
            "macro_f1": results["f1"]
        }
        records.append(row)

    df = pd.DataFrame(records)
    df.sort_values(by="macro_f1", ascending=False, inplace=True)
    df.to_csv(output_file_path, index=False)
