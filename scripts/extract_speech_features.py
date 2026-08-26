"""
Extract Speech features from audio recordings and validated transcripts in the LEGO Corpus.

This script performs the following steps:
    1. Load the validated system and user transcripts from the LEGO Corpus.
    2. Load the corresponding audio recordings for each transcript entry.
    3. Extract a comprehensive set of speech features from the audio recordings,
        including eGeMAPS features (using the openSMILE toolkit), speech embeddings (using
        pretrained HuBERT, Wav2Vec2, and WavLM models), and text embeddings (using 
        pretrained RoBERTa, TOD-BERT, and MiniLM models).
    4. Combine the extracted features with the transcript data and save the results to a 
        CSV file for downstream analysis and modeling.
"""

import pandas as pd
import sys
import argparse
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_base_path, set_all_seeds, resolve_path
from src.prep.feature_extraction import run_feature_extraction_pipeline

CONFIG = {
    "debug_limit": 50,  # Limit for debug mode
    "duration_threshold": 1.001,  # Threshold for response token duration
    "response_tokens": ["yeah", "yes", "yep", "correct", "wrong", "huh", "uh", "okay", "ok", "right", "alright", "oh", "sure", "really", "no", "nope"],
    "null_vals": ["", " ", "nan", "NaN", "N/A", "None", "null", "NULL", "none"],
    "pretrained_model_text": ["FacebookAI/roberta-base", "TODBERT/TOD-BERT-JNT-V1", "all-MiniLM-L6-v2"],
    "pretrained_model_speech": ["facebook/wav2vec2-base-960h", "microsoft/wavlm-base", "facebook/hubert-base-ls960"],
}

def parse_args():

    parser = argparse.ArgumentParser(description="Extract speech features.")

    parser.add_argument("--audio-dir",
                        default="data/raw/LetsGoIQ/audio",
                        help="Directory containing the raw audio files.")
    parser.add_argument("--transcripts-dir",
                        default="data/processed/transcripts/",
                        help="Directory to save combined transcript CSV file without embeddings.")
    parser.add_argument("--features-dir",
                        default="data/processed/extracted_features/",
                        help="Directory to save extracted features CSV files.")
    parser.add_argument("--user-transcript-path",
                        default="data/processed/transcripts/validated_user_transcript.csv",
                        help="Path to the user transcript CSV file.")
    parser.add_argument("--agent-transcript-path",
                        default="data/processed/transcripts/validated_agent_transcript.csv",
                        help="Path to the agent transcript CSV file.")
    parser.add_argument("--seed", 
                        default=42, 
                        type=int, 
                        help="Random seed for reproducibility.")
    parser.add_argument("--force",
                        action="store_true",
                        help="Force re-running full pipeline even if output files exist.")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Enable debug model to test code on a small subset of data.")
    parser.add_argument("--skip-embeddings",
                        action="store_true",
                        help="Skip extracting embeddings and only extract OpenSMILE features.")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    set_all_seeds(seed=args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base = get_base_path()

    directories = {
        "audio": resolve_path(base, args.audio_dir),
        "transcripts": resolve_path(base, args.transcripts_dir),
        "extracted_features": resolve_path(base, args.features_dir)
    }

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    input_file_paths = {
        "agent_transcript": resolve_path(base, args.agent_transcript_path),
        "user_transcript": resolve_path(base, args.user_transcript_path)
    }

    df_agent = pd.read_csv(input_file_paths["agent_transcript"])
    df_user = pd.read_csv(input_file_paths["user_transcript"])

    output_file_paths = {
        "combined_transcript": directories["transcripts"] / "combined_transcript.csv",
        "speech_features": directories["extracted_features"] / (
            "speech_features_no_embeds.csv" if args.skip_embeddings else "speech_features.csv"
        )
    }

    if args.debug:
        df_agent = df_agent.head(CONFIG["debug_limit"])
        df_user = df_user.head(CONFIG["debug_limit"])

        output_file_paths = {
            key: path.with_stem(f"{path.stem}_debug") 
            for key, path in output_file_paths.items()
        }

        print(f"Debug mode: Processing only the first {CONFIG['debug_limit']} rows of each transcript.")

    run_feature_extraction_pipeline(
        audio_dir=directories["audio"],
        df_agent=df_agent,
        df_user=df_user,
        output_filepaths=output_file_paths,
        config_dict=CONFIG,
        device=device,
        force=args.force,
        skip_embeddings=args.skip_embeddings
    )
    