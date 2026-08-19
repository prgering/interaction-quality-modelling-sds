"""
Extract Speech features from audio recordings and validated transcripts in the LEGO Corpus.

This script performs the following steps:
    1. Load the validated agent and user transcripts from the LEGO Corpus.
    2. Load the corresponding audio recordings for each transcript entry.
    3. Extract a comprehensive set of speech features from the audio recordings,
        including eGeMAPS features (using the openSMILE toolkit), Speech embeddings (using
        pretrained HuBERT, Wav2Vec2, and WavLM models), and Text embeddings (using 
        pretrained RoBERTa, TOD-BERT, and MiniLM models).
    4. Combine the extracted features with the transcript data and save the results to a 
        CSV file for downstream analysis and modeling.
"""

import pandas as pd
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_base_path, set_all_seeds, resolve_path
from src.prep_speech.get_filepaths import get_filepaths
from src.prep_speech.speech_feature_extractor import (
    run_feature_extraction_pipeline
)

CONFIG = {
    "duration_threshold": 1.001,  # Threshold for response token duration
    "response_tokens": ["yeah", "yes", "yep", "correct", "wrong", "huh", "uh", "okay", "ok", "right", "alright", "oh", "sure", "really", "no", "nope"],
    "null_vals": ["", " ", "nan", "NaN", "N/A", "None", "null", "NULL", "none"],
    "pretrained_model_text": ["FacebookAI/roberta-base", "TODBERT/TOD-BERT-JNT-V1", "all-MiniLM-L6-v2"],
    "pretrained_model_speech": ["facebook/wav2vec2-base-960h", "microsoft/wavlm-base", "facebook/hubert-base-ls960"],
}

def parse_args():

    parser = argparse.ArgumentParser(description="Extract speech features.")

    parser.add_argument("--processed-system-features-path", 
                        default="data/processed/system_features/processed_system_features.csv",
                        help="Path to the processed system features CSV file.")
    parser.add_argument("--audio-dir",
                        default="data/raw/LetsGoIQ/audio",
                        help="Directory containing the raw audio files.")
    parser.add_argument("--user-transcript-path",
                        default="data/processed/transcripts/validated_user_transcript.csv",
                        help="Path to the user transcript CSV file.")
    parser.add_argument("--agent-transcript-path",
                        default="data/processed/transcripts/validated_agent_transcript.csv",
                        help="Path to the agent transcript CSV file.")
    parser.add_argument("--output-filepath-combined-transcript",
                        default="data/processed/transcripts/combined_transcript_no_embeddings.csv",
                        help="Path to save the combined transcript CSV file without embeddings.")
    parser.add_argument("--output-filepath-speech-features",
                        default="data/processed/speech_features/speech_features.csv",
                        help="Path to save the extracted speech features CSV file.")
    parser.add_argument("--seed", 
                        default=42, 
                        type=int, 
                        help="Random seed for reproducibility.")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    set_all_seeds(seed=args.seed)

    base_path = get_base_path()

    audio_dir = resolve_path(base_path, args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    input_file_paths = {
        "agent_transcript": resolve_path(base_path, args.agent_transcript_path),
        "user_transcript": resolve_path(base_path, args.user_transcript_path)
    }

    output_file_paths = {
        "combined_transcript_no_embeddings": resolve_path(base_path, args.output_filepath_combined_transcript),
        "speech_features": resolve_path(base_path, args.output_filepath_speech_features)
    }

    audio_files_dict = get_filepaths(directory_dict = {"audio": audio_dir}, folder_to_process = "audio")

    df_agent_transcript = pd.read_csv(input_file_paths["agent_transcript"])
    df_user_transcript = pd.read_csv(input_file_paths["user_transcript"])

    run_feature_extraction_pipeline(audio_files_dict=audio_files_dict,
                                    df_agent=df_agent_transcript,
                                    df_user=df_user_transcript,
                                    output_filepaths=output_file_paths,
                                    config_dict=CONFIG)
    