"""
Trains interaction quality classifier using LSTM architectures. 

Supports both hyperparameter tuning (10-fold grouped cross-validation grid search) 
and final model evaluation on a held out test set. 

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
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cli_args import add_feature_set_args, add_pca_args, add_lstm_args
from src.models.data_loader import load_processed_data
from src.models.lstm_trainer import LstmManager
from src.utils import set_all_seeds, get_base_path, resolve_path


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
        "--cv-output-dir",
        default="experiments/hyperparam_tuning",
        help="Directory to save the hyperparameter tuning results."
    )
    parser.add_argument(
        "--eval-output-dir",
        default="experiments/final_evaluation",
        help="Directory to save the final evaluation results."
    )
    parser.add_argument(
        "--mode",
        choices=["tune", "evaluate"],
        required=True,
        help="Mode of operation: tune for hyperparameter tuning, evaluate for final model evaluation."
    )
    parser.add_argument(
        "--seed", 
        default=42, 
        type=int, 
        help="Random seed for reproducibility."
    )

    return parser.parse_args()

def run_tuning(trainer, output_dir):
    """Executes hyperparameter tuning sweep."""
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "local")
    output_file_path = output_dir / f"tuning_results_{task_id}.csv"

    results_dict = trainer.run_hyperparam_tuning()

    records = []
    for _, res in results_dict.items():
        row = {
            **res["params"],
            "macro_recall": res["recall"],
            "macro_f1": res["f1"],
        }
        records.append(row)

    df = pd.DataFrame(records).sort_values(by="macro_f1", ascending=False)
    df.to_csv(output_file_path, index=False)
    print(f"Tuning results saved to: {output_file_path}")

def run_evaluation(trainer, output_dir, dataset_type):
    """Executes training and evaluation of final model on test set."""

    subdirectories = {
        "metrics": output_dir / "metrics",
        "predictions": output_dir / "predictions",
        "true_labels": output_dir / "true_labels"
    }

    for path in subdirectories.values():
        path.mkdir(parents=True, exist_ok=True)

    metrics_file_path = subdirectories["metrics"] / f"final_evaluation.csv"

    results, true_labels, pred_labels = trainer.run_final_evaluation()

    row = {
        **results["params"],
        "macro_recall": results["recall"],
        "macro_f1": results["f1"],
    }

    df = pd.DataFrame([row])
    df.to_csv(metrics_file_path, index=False)
    print(f"Final evaluation results saved to: {metrics_file_path}")

    # Save true labels and predictions
    pred_filepath = subdirectories["predictions"] / f"predictions_best_model_frozen_{dataset_type}.npy"
    np.save(pred_filepath, pred_labels)
    print(f"Predictions saved to: {pred_filepath}")

    true_labels_path = subdirectories["true_labels"] / "true_labels_eval.npy"

    if true_labels_path.exists():
        existing_labels = np.load(true_labels_path)
        if not np.array_equal(existing_labels, true_labels):
            raise ValueError(
                f"True labels file already exists at {true_labels_path}" \
                 " and does not match the current true labels. " \
                 "Please check the files."
                )
        print(f"Verified true labels match existing file at: {true_labels_path}")
    else:
        np.save(true_labels_path, true_labels)
        print(f"True labels saved to: {true_labels_path}")


if __name__ == "__main__":
    args = parse_args()
    set_all_seeds(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_path = get_base_path()

    directories = {
        "input": resolve_path(base_path, args.input_dir),
        "cv_output": resolve_path(base_path, args.cv_output_dir),
        "eval_output": resolve_path(base_path, args.eval_output_dir),
    }
            
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    # Load training set and test set (if 'evaluate' mode is selected)
    train_df = load_processed_data(
        input_dir = directories["input"],
        args = args,
        feature_split = "train"
    )

    test_df = None
    if args.mode == "evaluate":
        test_df = load_processed_data(
            input_dir = directories["input"],
            args = args,
            feature_split = "test"
        )

    # Initialize the LSTM manager
    trainer = LstmManager(
        train_df=train_df, args=args,device=device, test_df=test_df
    )

    if args.mode == "tune":
        run_tuning(trainer, directories["cv_output"])
    elif args.mode == "evaluate":
        run_evaluation(trainer, directories["eval_output"], args.dataset_type)