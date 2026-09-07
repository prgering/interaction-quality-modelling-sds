#imports
import sys
import argparse
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cli_args import add_feature_set_args, add_pca_args, add_lstm_args
from src.models.data_loader import DataLoader
from src.models.lstm_trainer import LstmPipeline
from src.utils import set_all_seeds, get_base_path, resolve_path


CONFIG = {

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
    }

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    # Load the processed feature DataFrame
    train_df = DataLoader.load_processed_data(input_dir = directories["input"], 
                                            dataset_type = dataset_type, 
                                            feature_split = "train", 
                                            pca_n_comp_dict = pca_hyperparam_dict,
                                            pretrained_model_dict = pretrained_model_dict
    )

    # Run the experiments with the loaded DataFrame and specified parameters
    trainer = LstmPipeline(train_df = train_df, 
                filecode_column = filecode_col_name, 
                dv_column = dv_col_name, 
                dataset_type = dataset_type, 
                lstm_hyperparam_dict = lstm_hyperparam_dict, 
                pca_hyperparam_dict = pca_hyperparam_dict, 
                device = device, 
                base_path = base_path,
                auto_features_only = auto_features_only
    )

    results_dict = trainer.run_training_and_tuning()                 
            
    for key, results in results_dict.items():
        recall_value = results["recall"]
        f1_value = results["f1"]
        print(f"Hyperparameters: {key}, Macro Average Recall: {recall_value}, Macro Average F1: {f1_value}")


        
