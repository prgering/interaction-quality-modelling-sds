# Code Summary
"""
Script to perform permutation tests on model predictions for interaction quality.

Pairwise permutation tests are performed on Macro-F1 scores across the four
model variants (total six comparisons), with a Holm-Bonferroni correction applied for 
multiple comparisons.
"""

from pathlib import Path
import sys
import argparse
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_base_path, resolve_path, set_all_seeds
from src.analysis.permutation_tests import run_permutation_analysis


MODEL_VARIANTS = [
    "frozen_system",
    "frozen_speech",
    "fine_tuned_system",
    "fine_tuned_speech"
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse final evaluation results."
    )
    parser.add_argument(
        "--eval-results-dir",
        default="experiments/final_evaluation",
        help="Directory containing the hyperparameter tuning results."
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/final_evaluation_analysis",
        help="Directory to save the analysis results."
    )
    parser.add_argument(
        "--seed", 
        default=41, # Seed 41 retained for exact reproducibility with published results. 
        type=int, 
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--alpha",
        default=0.05,
        type=float,
        help="Significance level for Holm-Bonferroni correction."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    set_all_seeds(args.seed)

    base_path = get_base_path()

    eval_output_dir = resolve_path(base_path, args.eval_results_dir)

    input_dir = {
        "predictions": eval_output_dir / "predictions",
        "true_labels": eval_output_dir / "true_labels"
    }

    output_dir = resolve_path(base_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    true_labels = np.load(input_dir["true_labels"] / "true_labels_eval.npy")

    prediction_files = {
        variant: input_dir["predictions"] / f"predictions_best_model_{variant}.npy"
        for variant in MODEL_VARIANTS
    }

    for key, path in prediction_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Prediction file for {key} not found at {path}")

    results_df = run_permutation_analysis(
        prediction_files, true_labels, alpha=args.alpha
    )

    if results_df.empty:
        print("No model comparisons were performed.")
        return

    output_path = output_dir / "permutation_test_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Permutation test results saved to: {output_path}")

if __name__ == "__main__":
    main()