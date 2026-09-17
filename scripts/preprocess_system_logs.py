""" 
Loading, cleaning and pre-processing the system-log data from the Let's Go 
(LEGO) Corpus.

The following key operations are performed:
    1. Load the raw system log data from the LEGO Corpus.
    2. Clean the data by handling missing values and standardising data types and 
    labels
    3. Create binary variables for the high-level keys in the "SemanticParse" column.
    4. Extract text embeddings from the "Prompt" and "Utterance" columns using 
    pretrained language models.
    5. Save the cleaned and pre-processed features to CSV files for use in the 
    downstream classification task.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prep.system_logs import DataCleaner, DataPreprocessor
from src.utils import get_base_path, resolve_path, set_all_seeds

CONFIG = {
    "null_vals": ["", "<NA>", "nan","NA", "null", "None", "\\N", " "],
    # Note: Dropped by DataPreprocessor AFTER embedding generation
    "cols_to_drop": [
        # Columns with 0 variance
        "HelpRequest?", "#HelpRequests", "(#)HelpRequest", "%HelpRequest",
        # Redundant columns not needed for downstream tasks:
        "SemanticParse", "AudioFile", "Utterance",
    ],
    "keywords_to_drop": [
        # Manual annotation columns
        "SystemDialogueAct", "UserDialogueAct", "EmotionalState"
    ],
    "string_cols": ["Prompt", "Utterance", "SemanticParse"],
    "embed_cols": ["Prompt", "Utterance"],
    "pretrained_models": [
        "FacebookAI/roberta-base", 
        "TODBERT/TOD-BERT-JNT-V1", 
        "all-MiniLM-L6-v2"
    ],
    "dummy_cols": [
        "ASRRecognitionStatus", "ExMo", "Modality", "Activity", 
        "ActivityType", "RoleName", "LoopName"
    ],
}

def parse_args():

    parser = argparse.ArgumentParser(description="Preprocess system log features.")

    parser.add_argument("--output-dir", 
                        default="data/processed/extracted_features/",
                        help="Directory to save the processed system features.")
    parser.add_argument("--system-log-data",
                        default="data/raw/LetsGoIQ/corpus/csv/interactions_legov1.csv",
                        help="Path to the raw system log data CSV file.")
    parser.add_argument("--seed", 
                        default=42, 
                        type=int, 
                        help="Random seed for reproducibility.")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Run in debug mode with a smaller dataset for faster execution.")
    parser.add_argument("--skip-embeddings",
                        action="store_true",
                        help="Skip embedding generation for faster execution.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_all_seeds(args.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.empty_cache()

    base_path = get_base_path()

    output_dir = resolve_path(base_path, args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    system_logs_path = resolve_path(base_path, args.system_log_data)

    output_files = {
        "system_prompts": output_dir / "system_prompts_iq.csv",
        "processed_features": output_dir / "system_features.csv"
    }

    if args.debug:
        output_files = {
            key: path.with_stem(f"{path.stem}_debug") 
            for key, path in output_files.items()
        }

    if args.skip_embeddings:
        output_files = {
            key: path.with_stem(f"{path.stem}_no_embeds") 
            for key, path in output_files.items()
        }
    
    raw_df = pd.read_csv(
        system_logs_path, 
        encoding='latin1', 
        header=0,
        delimiter=';',
        na_values=CONFIG["null_vals"]
    )

    if args.debug:
        raw_df = raw_df.head(1000)

    cleaner = DataCleaner(raw_df, CONFIG)
    cleaned_df = cleaner.run_cleaning_pipeline()

    prompt_df = cleaned_df[["CallID","Prompt", "IQMedian"]]
    prompt_df.to_csv(
        output_files["system_prompts"], 
        index = False, 
        header = True
    )

    processor = DataPreprocessor(cleaned_df, CONFIG, device, skip_embedding=args.skip_embeddings)
    preprocessed_df = processor.run_preprocessing_pipeline()

    preprocessed_df.to_csv(
        output_files["processed_features"], 
        index = False, 
        header = True
    )