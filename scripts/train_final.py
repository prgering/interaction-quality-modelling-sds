#imports
import os
import pickle
import numpy as np
import torch
from pathlib import Path

# Custom local modules
from .data_loader import DataLoader
from .lstm_trainer import LstmManager
from ..arg_parser import ArgParser
from ..utils import set_all_seeds, get_base_path

if __name__ == "__main__":
    # Set random seeds for reproducibility
    set_all_seeds(42)

    # Set device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Code running on {device}")

    # Parse command line arguments
    (dataset_type, 
    lstm_hyperparam_dict, 
    pca_hyperparam_dict, 
    auto_features_only, 
    pretrained_model_dict) = ArgParser.parse_frozen_lstm_args()

    # Get the base path for data files
    base_path = get_base_path()

    # Define filecode and dependent variable column names
    filecode_col_name = "CallID"
    dv_col_name = "IQMedian"

    # Load the processed feature DataFrames
    train_df = DataLoader.load_processed_data(
        base_path = base_path, 
        dataset_type = dataset_type, 
        feature_split = "train", 
        pca_n_comp_dict = pca_hyperparam_dict,
        pretrained_model_dict = pretrained_model_dict
    )

    test_df = DataLoader.load_processed_data(
        base_path = base_path, 
        dataset_type = dataset_type, 
        feature_split = "test", 
        pca_n_comp_dict = pca_hyperparam_dict,
        pretrained_model_dict = pretrained_model_dict
    )

    # Run the experiments with the loaded DataFrame and specified parameters
    trainer = LstmManager(
        train_df = train_df, 
        filecode_col = filecode_col_name, 
        dv_col = dv_col_name, 
        dataset_type = dataset_type, 
        device = device, 
        base_path = base_path,
        test_df = test_df,
        auto_only = auto_features_only
    )

    (results, 
    true_labels,
    pred_labels_dict, 
    ) = trainer.run_final_evaluation(lstm_hyperparam_dict)

    # Save predictions to numpy files

    if auto_features_only:
        pred_filename = f'predictions_lstm_{dataset_type}_auto.npy'
    else:
        pred_filename = f'predictions_lstm_{dataset_type}.npy'

    output_dir = base_path / "experiments/output/iq_predictions"
    os.makedirs(output_dir, exist_ok=True)

    pred_filepath = output_dir / pred_filename

    np.save(pred_filepath, np.array(pred_labels_dict))
    print(f"\nPredictions saved to {pred_filepath}")

    true_labels_path = base_path / "experiments/output/true_labels_best_model_lstm.npy"
    if not os.path.exists(true_labels_path):
        np.save(true_labels_path, np.array(true_labels))
        print(f"\nTrue labels saved to {true_labels_path}")
    


    



        
