# imports
import sys
import argparse
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_base_path, resolve_path, set_all_seeds
from src.fine_tuning import speech_transformer, system_transformer
from src.models.pca_pipeline import load_feature_data, train_opt_split
from src.models.lstm_trainer import build_param_grid, calculate_class_weights
from src.models.cli_args import (
    add_feature_set_args, 
    add_pca_args, 
    add_lstm_args,
    add_fine_tuning_args
)

CONFIG = {
    "debug_limit": 50,
    "train_size": 0.8,
}    

def parse_args():
    """Parses command line arguments for fine-tuning the LSTM model."""
    parser = argparse.ArgumentParser(
        description="Fine-tune LSTM hyperparameters for interaction quality prediction."
    )

    add_feature_set_args(parser)
    add_pca_args(parser)
    add_lstm_args(parser)
    add_fine_tuning_args(parser)

    parser.add_argument(
        "--input-dir",
        default="data/processed/extracted_features",
        help="Directory containing the extracted features."
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/fine_tuning_results",
        help="Directory to save the fine tuning results."
    )
    parser.add_argument(
        "--remove-first-turn",
        action="store_true",
        help="Remove the first turn from each call."
    )
    parser.add_argument(
        "--run-final-eval",
        action="store_true",
        help="Run final evaluation on the test set after fine-tuning."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to test code on a small subset of data."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_path = get_base_path()

    directories = {
        "input": resolve_path(base_path, args.input_dir),
        "output": resolve_path(base_path, args.output_dir),
    }

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    if args.dataset_type == "speech":
        input_file_path = directories["input"] / "speech_features_no_embeds.csv"
    else:
        input_file_path = directories["input"] / "filtered_system_features_no_embeds.csv"

    # Module Map
    module_map = {
        "speech": speech_transformer,
        "system": system_transformer
    }

    m = module_map[args.dataset_type]

    full_df = load_feature_data(
        dataset_path=input_file_path, 
        dataset_type=args.dataset_type,
        remove_first_turn=args.remove_first_turn
    )

    if args.debug:
        full_df = full_df.head(CONFIG["debug_limit"])
        print(f"Debug mode: Processing only the first {CONFIG['debug_limit']} rows of the dataset.")

    temp_df, test_df = train_opt_split(
        full_df, 
        grouping_col="CallID", 
        train_size=CONFIG["train_size"], 
        random_state=args.seed
    )

    train_df, val_df = train_opt_split(
        temp_df, 
        grouping_col="CallID", 
        train_size=CONFIG["train_size"], 
        random_state=args.seed
    )

    # Scale Static Features
    non_feature_cols = ["CallID", "IQMedian"]

    static_cols = [col for col in train_df.columns if col not in non_feature_cols]

    scaler = StandardScaler()
    train_df[static_cols] = scaler.fit_transform(train_df[static_cols])
    val_df[static_cols] = scaler.transform(val_df[static_cols])
    test_df[static_cols] = scaler.transform(test_df[static_cols])

    # Compute Class Weights for Imbalanced Data
    train_labels_flat = train_df["IQMedian"].values
    all_classes = np.unique(full_df["IQMedian"].values)

    class_weights = calculate_class_weights(
        train_labels_flat, 
        all_classes, 
        device
    )

    # Dataset & Static Feature Check
    temp_dataset = m.IQDataset(
        train_df, static_cols=static_cols
    )

    num_static = temp_dataset[0]["static_feats"][0].shape[0]

    # Build Hyperparameter Grid

    def unwrap_param_grid(param_grid):
        """Unwraps single-item lists if present."""
        unwrapped = {}
        for key, val in param_grid.items():
            if isinstance(val, list) and len(val) == 1:
                unwrapped[key] = val[0]
            else:
                unwrapped[key] = val
        return unwrapped
    
    param_grid = unwrap_param_grid(build_param_grid(args=args))


    # Set multiple random seeds if final_run_eval is True
    if args.run_final_eval:
        random_seeds = [42, 16613,42875, 66865, 68085]
    else:
        random_seeds = [42]

    seed_results = []

    # Loop through random seeds for training and evaluation
    for seed in random_seeds:
        print(f"\n" + "="*40)
        print(f"\nRunning with random seed: {seed}")
        print("="*40)
        set_all_seeds(seed)

        # Initialize Model
        model = m.IQModel(
            num_static_features=num_static, num_classes=5,
            hidden_size=param_grid["hidden_size"],
            num_layers=param_grid["num_layers"],
            bidirectional=param_grid["bidirectional"],
            use_attention=param_grid["use_attention"],
            window_size=param_grid["window_size"],
            num_freeze=param_grid["num_frozen_layers"],
        ).to(device)

        # Prepare DataLoaders
        train_loader, val_loader, test_loader = m.prepare_dataloaders(
            train_df, val_df, test_df, static_cols=static_cols
        )

        # Train Model
        best_model = m.train_model(
            model, train_loader, val_loader, device,
            class_weights_tensor=class_weights,
            transformer_lr=param_grid["transformer_lr"],
            head_lr=param_grid["head_lr"],
        )

        # Evaluate on Validation Set
        val_results, _, _ = m.evaluate_on_loader(
            best_model, val_loader, device
        )

        print("\nValidation Set Results:")
        recall_value = val_results["macro avg"]["recall"]
        f1_value = val_results["macro avg"]["f1-score"]
        
        print(f"Hyperparameters: {param_grid}, Macro Average Recall: {recall_value}, Macro Average F1: {f1_value}")

        # Final Evaluation on Test Set (if specified)
        if args.run_final_eval:
            test_results, test_targets, test_preds = m.evaluate_on_loader(
                best_model, 
                test_loader, 
                device,
                verbose=True
            )

            recall_value = test_results["macro avg"]["recall"]
            f1_value = test_results["macro avg"]["f1-score"]

            seed_results.append({
                "seed": seed,
                "recall": recall_value,
                "f1": f1_value,
                "preds": test_preds,
                "targets": test_targets
            })

    if args.run_final_eval and len(seed_results) > 0:
        
        sorted_results = sorted(seed_results, key=lambda x: x['f1'])

        median_idx = len(sorted_results) // 2

        median_seed_data = sorted_results[median_idx]

        print(
            f"\nMedian Seed: {median_seed_data['seed']}"
            f" with Macro F1: {median_seed_data['f1']:.4f}"
            f" and Macro Recall: {median_seed_data['recall']:.4f}"
        )

        subdirectories = {
            "metrics": directories["output"] / "metrics",
            "predictions": directories["output"] / "predictions",
            "true_labels": directories["output"] / "true_labels"
        }

        # Save the best seed predictions as a numpy file
        pred_filename = f'predictions_best_model_fine_tuned_{args.dataset_type}.npy'
    
        pred_filepath = subdirectories["predictions"] / pred_filename

        np.save(pred_filepath, np.array(median_seed_data["preds"]))
        print(f"\nBest seed predictions saved to {pred_filepath}")

        # Save true labels (only if they don't exist)
        true_labels = np.array(median_seed_data["targets"])

        true_labels_filename = f'true_labels_best_model_fine_tuned_{args.dataset_type}.npy'
        true_labels_path = subdirectories["true_labels"] / true_labels_filename

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
    main()
        
