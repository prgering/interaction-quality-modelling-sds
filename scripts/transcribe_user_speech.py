"""
Transcribe user speech from the LEGO Corpus.

The speech processing pipeline consists of the following key steps:
    1. Load user-amplified audio recordings from the LEGO Corpus.
    2. Perform Voice Activity Detection (VAD) using a pretrained Silero 
    model to identify speech segments.
    3. Normalise audio and apply Gaussian white noise to mask residual
    background noise.
    4. Perform second pass of VAD to isolate user speech segments.
    5. Transcribe the user speech segments using a pretrained Whisper ASR 
    model.
    """
import argparse
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_base_path, set_all_seeds, resolve_path
from src.prep.user_speech import UserSpeechProcessor

CONFIG = {
    "vad_params": {
        "inaccurate_user_vad_threshold": 0.8,
        "accurate_user_vad_threshold": 0.3,
        "min_speech_duration_ms": 50,
        "sampling_rate": 8000
    },
    "normalise_params": {
        "target_rms": 0.07,
        "white_noise_level": 0.05
    },
    "asr_params": {
        "model_type": "large-v3",
        "batch_size": 16,
        "compute_type": "float16"
    }
}

def parse_args():

    parser = argparse.ArgumentParser(description="Transcribe user speech.")

    parser.add_argument("--system-features-path", 
                        default="data/processed/system_features.csv",
                        help="Path to system features CSV file.")
    parser.add_argument("--audio-dir",
                        default="data/raw/LetsGoIQ/audio",
                        help="Directory containing the raw audio files.")
    parser.add_argument("--inaccurate-vad-dir",
                        default="data/processed/vad/inaccurate_user",
                        help="Directory to save the inaccurate VAD segments.")
    parser.add_argument("--accurate-vad-dir",
                        default="data/processed/vad/accurate_user",
                        help="Directory to save the accurate VAD segments.")
    parser.add_argument("--normalised-audio-dir",
                        default="data/processed/normalised_noisy_audio",
                        help="Directory to save the normalised audio files.")
    parser.add_argument("--output-filepath",
                        default="data/processed/transcripts/user_asr_output.csv",
                        help="Path to save the user speech ASR output.")
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
        "inaccurate_vad": resolve_path(base, args.inaccurate_vad_dir),
        "accurate_vad": resolve_path(base, args.accurate_vad_dir),
        "normalised_audio": resolve_path(base, args.normalised_audio_dir),
    }

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    args.output_filepath = resolve_path(base, args.output_filepath)

    speech_processor = UserSpeechProcessor(
        CONFIG, directories, args.output_filepath
    )

    speech_processor.run_pipeline()

    

    




