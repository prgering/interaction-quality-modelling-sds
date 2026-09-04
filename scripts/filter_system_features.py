""" 
Excluding dialogues from the system-derived feature set.

The criteria for exclusion include:
- Missing audio files
- Dialogues with no user or agent speech

Note that the old version of this code did not call "cols_to_drop"
properly, resulting in these columns being retained in the filtered 
system features. This version fixes that issue.
"""

# Imports
import argparse
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.prep.system_logs import DialogueExcluder
from src.utils import get_base_path, resolve_path, set_all_seeds


CONFIG = {
    "debug_limit": 100,  # Limit for debug mode
    "null_vals": ["", "<NA>", "nan","NA", "null", "None", "\\N", " "],
    "cols_to_drop": [
        "Transcript", "Speaker", 'StartTime', 'EndTime', "Prompt"
    ],
    # Excluded files where user talks to third party
    "excluded_files": [
        "20061127007", "20061128012", "20061125060", "20061126057"
    ],
}

def parse_args():

    parser = argparse.ArgumentParser(description="Filter system-dependent features.")

    parser.add_argument("--audio-dir",
                        default="data/raw/LetsGoIQ/audio",
                        help="Directory containing the raw audio files.")
    parser.add_argument("--system-features-path", 
                        default="data/processed/extracted_features/system_features_full.csv",
                        help="Path to the processed system features CSV file.")
    parser.add_argument("--user-transcript-path",
                        default="data/processed/transcripts/validated_user_transcript.csv",
                        help="Path to the validated user transcript CSV file.")
    parser.add_argument("--system-transcript-path",
                        default="data/processed/transcripts/validated_agent_transcript.csv",
                        help="Path to the validated agent transcript CSV file.")
    parser.add_argument("--output-dir",
                        default="data/processed/extracted_features",
                        help="Directory to save the filtered system features and exclusion report.")
    parser.add_argument("--seed", 
                        default=42, 
                        type=int, 
                        help="Random seed for reproducibility.")
    parser.add_argument("--force",
                        action="store_true",
                        help="Force re-running full pipeline even if output files exist.")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Enable debug mode to test code on a small subset of data.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_all_seeds(args.seed)

    base = get_base_path()

    directories = {
        "audio": resolve_path(base, args.audio_dir),
        "output": resolve_path(base, args.output_dir)
    }

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    input_file_paths = {
        "system_features": resolve_path(base, args.system_features_path),
        "user_transcript": resolve_path(base, args.user_transcript_path),
        "system_transcript": resolve_path(base, args.system_transcript_path)
    }

    df_system_features = pd.read_csv(input_file_paths["system_features"])
    df_user = pd.read_csv(input_file_paths["user_transcript"])
    df_system_transcript = pd.read_csv(input_file_paths["system_transcript"])
    
    output_files = {
        "filtered_system_features": (
            directories["output"] / "filtered_system_features.csv"
        ),
        "excluded_dialogues": (
            directories["output"] / "exclusion_reasons.txt"
        )
    }

    if args.debug:
        df_system_features = df_system_features.head(CONFIG["debug_limit"])
        df_user = df_user.head(CONFIG["debug_limit"])
        df_system_transcript = df_system_transcript.head(CONFIG["debug_limit"])

        output_files = {
            key: path.with_stem(f"{path.stem}_debug")
            for key, path in output_files.items()
        }
    
    excluder = DialogueExcluder(
        df_system_features, df_user, df_system_transcript,
        directories["audio"], CONFIG
    )
    
    filtered_df, excluded_files = excluder.run_pipeline()

    # ------------ Save Filtered Data ------------

    filtered_df.to_csv(
        output_files["filtered_system_features"], 
        index = False, 
        header = True
    )

    # ------------ Save Exclusion Reasons and Statistics ------------

    final_filecodes_count = len(filtered_df['CallID'].unique())
    filecodes_excluded_count = len(excluded_files.keys())
    total_file_count = filecodes_excluded_count + final_filecodes_count
    percent_excluded = (filecodes_excluded_count / total_file_count) * 100

    with open(output_files["excluded_dialogues"], 'w') as f:
        for filecode, reason in excluded_files.items():
            f.write(f"{filecode}: {reason}\n")

        f.write(
            f"{filecodes_excluded_count} files excluded out of"
            f"{total_file_count} \n{final_filecodes_count} files included" 
            f"\n{percent_excluded}% of files excluded"
        )




    

    


    
