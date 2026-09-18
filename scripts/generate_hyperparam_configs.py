import itertools
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_base_path, resolve_path

EXPERIMENT_GRIDS = {
    "speech_pca": {
        "speechf_text_pca": [0.5, 0.6, 0.7, 0.8],
        "speechf_wav_pca": [0.5, 0.6, 0.7, 0.8],
    },
    "system_svm": {
        "systemf_text_pca": [0.5, 0.6, 0.7, 0.8],
        "text_embedding_model": ["sbert", "roberta", "todbert"],
    },
    "speech_svm": {
        "speechf_text_pca": [0.5, 0.6, 0.7, 0.8],
        "speechf_wav_pca": [0.5, 0.6, 0.7, 0.8],
        "text_embedding_model": ["sbert", "roberta", "todbert"],
        "speech_embedding_model": ["wav2vec", "hubert", "wavlm"],
    },
    "system_lstm": {
        "use_attention": [False, True],
        "bidirectional": [False, True],
        "hidden_size": [128, 256, 384],
        "systemf_text_pca": [0.5, 0.6, 0.7, 0.8],
        "text_embedding_model": ["sbert", "roberta", "todbert"],
    },
    "speech_lstm": {
        "use_attention": [False, True],
        "bidirectional": [False, True],
        "hidden_size": [128, 256, 384],
        "speechf_text_pca": [0.5, 0.6, 0.7, 0.8],
        "speechf_wav_pca": [0.5, 0.6, 0.7, 0.8],
    },
    "system_end2end": {
        "window_size": [5, 10, 15],
        "transformer_lr": [5e-6, 1e-5, 5e-5],
        "num_frozen_layers_sbert": [0, 3, 6],
    },
    "speech_end2end": {
        "window_size": [5, 10, 15],
        "transformer_lr": [5e-6, 1e-5, 5e-5],
        "num_frozen_layers_speech_encoders": [0, 6, 12],
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate hyperparameter combinations for hyperparameter tuning.")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="speech_lstm",
        help="Name of the experiment (default: 'speech_lstm')."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="slurm/configs",
        help="Directory to save the hyperparameter combinations (default: 'slurm/configs')."
    )
    parser.add_argument(
        "--output-filename",
        required=True,
        type=str,
        default=None,
        help="Output file to save the hyperparameter combinations (required)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    base = get_base_path()

    output_dir = resolve_path(base, args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment_name not in EXPERIMENT_GRIDS:
        raise ValueError(f"Experiment name '{args.experiment_name}' is not recognized. Please choose from {list(EXPERIMENT_GRIDS.keys())}.")

    grid = EXPERIMENT_GRIDS[args.experiment_name]
    keys = list(grid.keys())
    combinations = list(itertools.product(*grid.values()))

    output_file_path = output_dir / f"{args.output_filename}"

    # Write combinations to the file
    with open(output_file_path, 'w') as f:
        for combo in combinations:
            f.write(" ".join(map(str, combo)) + "\n")

    # Console Summary
    print(f"Experiment: '{args.experiment_name}'")
    print(f"Parameters: {', '.join(keys)}")
    print(f"Successfully generated {len(combinations)} combinations at:")
    print(f"  -> {output_file_path}")
    print(f"\nSlurm Header Setting: #SBATCH --array=1-{len(combinations)}")

if __name__ == "__main__":
    main()