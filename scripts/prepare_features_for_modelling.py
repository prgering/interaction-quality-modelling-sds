"""
Prepares features for modelling by loading, preprocessing, and splitting the data.

PCA and scaling are applied to embedding features, whereas non-embedding features are
only scaled. The processed features are saved to separate CSV files depending on the
dataset split and PCA n-components value.
"""

#imports
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_all_seeds, get_base_path, resolve_path
from src.models.cli_args import add_feature_set_args, add_pca_args
from src.models.pca_pipeline import load_feature_data, train_opt_split, run_pre_processing_steps


CONFIG = {
    "debug_limit": 100,
    "train_size": 0.8,
}

def parse_args():
    """Parses command line arguments for PCA preprocessing."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter settings for preprocessing with PCA."
    )

    add_feature_set_args(parser)
    add_pca_args(parser)

    parser.add_argument(
        "--input-dir",
        default="data/processed/extracted_features",
        help="Directory containing the extracted features."
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/preprocessed_features",
        help="Directory to save the preprocessed features."
    )
    parser.add_argument(
        "--remove-first-turn",
        action="store_true",
        help="Remove the first turn from each call."
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
    set_all_seeds(args.seed)

    base_path = get_base_path()

    directories = {
        "input": resolve_path(base_path, args.input_dir),
        "output": resolve_path(base_path, args.output_dir),
    }

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    file_names = {
        "speech": "speech_features.csv",
        "system": "filtered_system_features.csv",
    }

    args.dataset_type = args.dataset_type.lower()

    input_file_path = directories["input"] / file_names[args.dataset_type]

    pca_hyperparams = {
        "n_comp_systemf_text": args.systemf_text_pca,
        "n_comp_speechf_text": args.speechf_text_pca,
        "n_comp_speechf_wav": args.speechf_wav_pca
    }

    # Load and split data into a train and test set

    print(f"Loading and preprocessing {args.dataset_type}", flush=True)
    
    df_features = load_feature_data(
        dataset_path=input_file_path,
        dataset_type=args.dataset_type,
        remove_first_turn=args.remove_first_turn
    )

    if args.debug:
        df_features = df_features.head(CONFIG["debug_limit"])

    train_df, eval_df = train_opt_split(
        df_features, grouping_col="CallID", 
        train_size=CONFIG["train_size"],
        random_state=args.seed
    )

    # Run preprocessing steps - PCA and Scaling
    dfs_dict = run_pre_processing_steps(
        train_df = train_df, 
        test_df = eval_df, 
        pca_params_dict = pca_hyperparams)

    def format_pca_val(val):
        if val is None:
            return ""
        return str(int(round(float(val) * 10)))

    sys_pca_str = format_pca_val(pca_hyperparams["n_comp_systemf_text"])
    sp_txt_pca_str = format_pca_val(pca_hyperparams["n_comp_speechf_text"])
    sp_wav_pca_str = format_pca_val(pca_hyperparams["n_comp_speechf_wav"])

    # Inspect a sample of the processed data

    train_df = dfs_dict["train"]
    for col in train_df.columns[:50]:  # Limit to first 50 columns for inspection
        print(f"\nColumn: {col}")
        print(train_df[col].unique())

    
    # Save processed DataFrames to CSV files
    for dataset_split, processed_df in dfs_dict.items():
        
        print(f"\nProcessed {dataset_split} DataFrame shape: {processed_df.shape}")
        
        filename = f"{dataset_split}_{args.dataset_type}_sytxtpca{sys_pca_str}_sptxtpca{sp_txt_pca_str}_spwpca{sp_wav_pca_str}"

        if args.debug:
            filename += "_debug"
        filename += ".csv"
        
        file_path = directories["output"] / filename

        print(f"Saving processed data to '{file_path}'...")
        processed_df.to_csv(file_path, index=False)
        print(f"Saved processed data with shape {processed_df.shape} to '{file_path}'", flush=True)                            


        
