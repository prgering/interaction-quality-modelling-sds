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


def parse_args():
    """Parses command line arguments for PCA preprocessing."""
    parser = argparse.ArgumentParser(description="Hyperparameter settings for preprocessing with PCA.")

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
        "--seed", 
        default=42, 
        type=int, 
        help="Random seed for reproducibility."
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
        dataset_type=args.dataset_type
    )

    train_df, eval_df = train_opt_split(
        df_features, grouping_col="CallID", train_size=0.8
    )

    # Run preprocessing steps - PCA and Scaling
    dfs_dict = run_pre_processing_steps(
        train_df = train_df, 
        test_df = eval_df, 
        pca_params_dict = pca_hyperparams)
    
    # Save processed DataFrames to CSV files
    for dataset_split, processed_df in dfs_dict.items():
        
        print(f"\nProcessed {dataset_split} DataFrame shape: {processed_df.shape}")
        
        filename = f"{dataset_split}_{args.dataset_type}"
        if args.dataset_type == "system":
            filename += f"_sytxtpca{pca_hyperparams['n_comp_systemf_text']}_sptxtpca_spwpca"
        elif args.dataset_type == "speech":
            filename += f"_sytxtpca_sptxtpca{pca_hyperparams['n_comp_speechf_text']}_spwpca{pca_hyperparams['n_comp_speechf_wav']}"
        filename += ".csv"
        
        file_path = directories["output"] / filename

        print(f"Saving processed data to '{file_path}'...")
        processed_df.to_csv(file_path, index=False)
        print(f"Saved processed data with shape {processed_df.shape} to '{file_path}'", flush=True)                            


        
