"""
Extract system transcript and timestamps from mixed transcript using 
ASR and Fuzzy alignment.

The script performs the following key steps:
    1. Load the cleaned user transcript and agent prompt data.
    2. Use WhisperX to perform ASR, alignment and speaker diarisation on
    the original audio recordings.
    3. Use RapidFuzz to match the user transcript with segments of the 
    mixed transcript.
    4. Use RapidFuzz to match the agent prompts with the remaining unmatched 
    segments of the mixed transcript.
    5. Output the aligned agent transcript, a JSON file of unmatched prompts,
    and a CSV file of mismatched transcript segments for further analysis.
"""

import os
import inflect
import torch
import sys
from pathlib import Path
import argparse
from dotenv import load_dotenv

load_dotenv()
p = inflect.engine()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prep.system_speech import run_pipeline
from src.utils import get_base_path, set_all_seeds, resolve_path

CONFIG = {
    "asr_params": {
        "model_type": "large-v3",
        "batch_size": 16,
        "compute_type": "float16",
        "pyannote_auth_token": os.getenv("PYANNOTE_AUTH_TOKEN")
    },
    "alignment_params": {
        "max_window": 5,
        "proximity_window": 5,
        "user_time_tolerance": 0.5,
        "user_fuzz_threshold": 80,
        "word_fuzz_threshold": 60
    }
}

def parse_args():

    parser = argparse.ArgumentParser(description="Preprocess system log features.")

    parser.add_argument("--system-prompt-csv", 
                        default="data/processed/system_features/prompt_iq_legov1.csv",
                        help="Path to the system prompt CSV file.")
    parser.add_argument("--audio-dir",
                        default="data/raw/LetsGoIQ/audio",
                        help="Directory containing the raw audio files.")
    parser.add_argument("--processed-transcript-dir",
                        default="data/processed/transcripts",
                        help="Directory to save the processed transcript files.")
    parser.add_argument("--mixed-transcript-dir",
                        default="data/processed/mixed_transcripts",
                        help="Directory containing the mixed transcript files.")
    parser.add_argument("--output-filepath",
                        default="data/processed/transcripts/unedited_user_transcript.csv",
                        help="Path to save the unedited user transcript CSV file.")
    parser.add_argument("--seed", 
                        default=42, 
                        type=int, 
                        help="Random seed for reproducibility.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_all_seeds(seed=args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base = get_base_path()

    directories = {
        "audio": resolve_path(base, args.audio_dir),
        "processed_transcript": resolve_path(base, args.processed_transcript_dir),
        "mixed_transcript": resolve_path(base, args.mixed_transcript_dir),
    }

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    input_filepaths = {
        "user_csv": (
            directories["processed_transcript"] / 
            "validated_user_transcript.csv"
        ),
        "agent_csv": (
            resolve_path(base, args.system_prompt_csv)
        )
    }

    output_filepaths = {
        "aligned_agent_transcript": (
            directories["processed_transcript"] / 
            "aligned_agent_transcript.csv"
        ),
        "unmatched_prompts_json": (
            directories["processed_transcript"] / 
            "unmatched_agent_prompts.json"
        ),
        "mismatched_transcript_csv": (
            directories["processed_transcript"] / 
            "mismatched_agent_transcript.csv"
        )
    }

    run_pipeline(directories, input_filepaths, output_filepaths, CONFIG, device)