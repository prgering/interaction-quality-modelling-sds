# imports
import os
import sys
import argparse
import random
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Local imports
from src.fine_tuning import speech_transformer, system_transformer
from .shared_comp import calculate_class_weights

from ..arg_parser import ArgParser
from ..utils import (
    calc_metrics,
    get_base_path,
    load_feature_data,
    set_all_seeds,
    train_opt_split
)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    set_all_seeds(42)

    # Set device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Code running on {device}")    

    # Get the base path for data files
    base_path = get_base_path()

    # Parse command line arguments
    dataset_type, lstm_hyperparam_dict, auto_features_only = ArgParser.parse_fine_tuned_lstm_args()

    print(f"Fine-tuning model on {dataset_type} Dataset\n" + "-"*30)

    # Module Map
    module_map = {
        "speech_features": speech_transformer,
        "system_features": system_transformer
    }

    m = module_map[dataset_type]

    # Training Config
    config = {
        "filecode_col_name": "CallID",
        "dv_col_name": "IQMedian",
        "remove_first_turn": True,
        "train_size": 0.8,
        "test_sample": False,
        "run_final_eval": False,
    }    

    # Define directory and file path
    directory_path = base_path / f"data/processed/{dataset_type}"

    input_file_dict = {
        "speech_features": "static_speech_features.csv",
        "system_features": "filtered_static_system_features.csv",
    }

    input_file_path = directory_path / input_file_dict[dataset_type]

    # Load the full feature DataFrame
    full_df = load_feature_data(dataset_path=input_file_path, 
                                call_id_col=config["filecode_col_name"], 
                                remove_first_turn=config["remove_first_turn"]
    )

    # Test Sample (for quick debugging)
    if config["test_sample"]:
        unique_callids = full_df[config["filecode_col_name"]].unique()
        sampled_callids = random.sample(list(unique_callids), k=15)
        full_df = full_df[full_df[config["filecode_col_name"]].isin(sampled_callids)].copy()
        print(f"Sampled DataFrame size: {full_df.shape}")

    # Filter for Auto Features if Specified

    if auto_features_only and dataset_type == "system_features":
        print("Filtering for 'AUTO' features (excluding DialogueAct and EmotionState).")
    
        full_df = full_df.drop(columns=full_df.filter(regex='DialogueAct|EmotionState').columns)

    # Split Data into Temp and Test Sets
    temp_df, test_df = train_opt_split(
        full_df, 
        call_id_col=config["filecode_col_name"], 
        target_column=config["dv_col_name"], 
        train_size=config["train_size"]
    )

    # Split Temp into Train and Validation Sets
    train_df, val_df = train_opt_split(
        temp_df, 
        call_id_col=config["filecode_col_name"], 
        target_column=config["dv_col_name"], 
        train_size=config["train_size"])

    print(f"Train set filecodes: {train_df[config['filecode_col_name']].unique()}")
    print(f"Validation set filecodes: {val_df[config['filecode_col_name']].unique()}")
    print(f"Test set filecodes: {test_df[config['filecode_col_name']].unique()}")
    
    # Scale Static Features
    non_feature_cols = [
        config["filecode_col_name"], 
        config["dv_col_name"], 
        "AudioPath", "StartTime", 
        "EndTime", "CombinedTranscript", 
        "IQMedian", "Prompt", "Utterance"
    ]

    static_cols = [col for col in train_df.columns if col not in non_feature_cols]

    scaler = StandardScaler()
    train_df[static_cols] = scaler.fit_transform(train_df[static_cols])
    val_df[static_cols] = scaler.transform(val_df[static_cols])
    test_df[static_cols] = scaler.transform(test_df[static_cols])

    # Compute Class Weights for Imbalanced Data
    train_labels_flat = train_df[config["dv_col_name"]].values
    all_classes = np.unique(full_df[config["dv_col_name"]].values)

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

    # Troubleshooting print statements
    print("Training fine-tuned pipeline with the following hyperparameters:")
    for key, value in lstm_hyperparam_dict.items():
        print(f"{key}: {value}")

    # Set multiple random seeds if final_run_eval is True
    if config["run_final_eval"]:
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
            hidden_size=lstm_hyperparam_dict["hidden_size"],
            num_layers=lstm_hyperparam_dict["num_layers"],
            bidirectional=lstm_hyperparam_dict["bidirectional"],
            use_attention=lstm_hyperparam_dict["use_attention"],
            window_size=lstm_hyperparam_dict["window_size"],
            num_freeze=lstm_hyperparam_dict["num_frozen_layers"],
        ).to(device)

        # Prepare DataLoaders
        train_loader, val_loader, test_loader = m._prepare_dataloaders(
            train_df, val_df, test_df, static_cols=static_cols
        )

        # Train Model
        best_model = m.train_model(
            model, train_loader, val_loader, device,
            class_weights_tensor=class_weights,
            transformer_lr=lstm_hyperparam_dict["transformer_lr"],
            head_lr=lstm_hyperparam_dict["head_lr"],
        )

        # Evaluate on Validation Set
        val_results, _, _ = m._evaluate_on_loader(
            best_model, val_loader, device
        )

        print("\nValidation Set Results:")
        recall_value = val_results["macro avg"]["recall"]
        f1_value = val_results["macro avg"]["f1-score"]
        
        print(f"Hyperparameters: {lstm_hyperparam_dict}, Macro Average Recall: {recall_value}, Macro Average F1: {f1_value}")

        # Final Evaluation on Test Set (if specified)
        if config["run_final_eval"]:
            test_results, test_targets, test_preds = m._evaluate_on_loader(
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

    if config["run_final_eval"] and len(seed_results) > 0:
        
        sorted_results = sorted(seed_results, key=lambda x: x['f1'])

        median_idx = len(sorted_results) // 2

        median_seed_data = sorted_results[median_idx]

        print(
            f"\nMedian Seed: {median_seed_data['seed']}"
            f" with Macro F1: {median_seed_data['f1']:.4f}"
            f" and Macro Recall: {median_seed_data['recall']:.4f}"
        )

        # Save the best seed predictions as a numpy file
        if auto_features_only:
            pred_filename = f'predictions_fine_tuned_lstm_{dataset_type}_auto.npy'
        else:
            pred_filename = f'predictions_fine_tuned_lstm_{dataset_type}.npy'
        
        output_dir = base_path / "experiments/output/iq_predictions"
        os.makedirs(output_dir, exist_ok=True)
        pred_filepath = output_dir / pred_filename

        np.save(pred_filepath, np.array(median_seed_data["preds"]))
        print(f"\nBest seed predictions saved to {pred_filepath}")

        # Save true labels (only if they don't exist)
        true_labels_path = "experiments/output/true_labels_best_model_lstm.npy"
        if not os.path.exists(true_labels_path):
            np.save(true_labels_path, np.array(median_seed_data["targets"]))
            print(f"\nTrue labels saved to {true_labels_path}")
        


        
