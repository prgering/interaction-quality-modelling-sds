from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from statsmodels.stats.multitest import multipletests

def macro_f1(y_true, y_pred):
    """Calculates the unweighted average F1 (Macro-F1) score."""
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

def macro_recall(y_true, y_pred):
    """Calculates the unweighted average Recall (Macro-Recall) score."""
    return recall_score(y_true, y_pred, average="macro", zero_division=0)

def permutation_test(metric_fn, y_true, y_pred_a, y_pred_b, tries=4096):
    """Performs a non-parametric permutation test to compare two models."""
    # Ensure input arrays are the same length
    assert y_true.shape[0] == y_pred_a.shape[0] == y_pred_b.shape[0]

    # Calculate observed performance
    metric_a = metric_fn(y_true, y_pred_a)
    metric_b = metric_fn(y_true, y_pred_b)
    benchmark = np.abs(metric_a - metric_b)
    
    # Permutation loop: randomly swap predictions and calculate performance difference
    samples = np.zeros(tries)
    for i in range(tries):
        msk = np.random.random(size=y_true.shape[0]) < 0.5
        
        y_pred_a_perm, y_pred_b_perm = y_pred_a.copy(), y_pred_b.copy()

        y_pred_a_perm[msk], y_pred_b_perm[msk] = y_pred_b[msk], y_pred_a[msk]

        # Record the absolute difference in performance for this permutation
        samples[i] = np.abs(
            metric_fn(y_true, y_pred_a_perm) - metric_fn(y_true, y_pred_b_perm)
        )

    # Significance calculation: proportion of permuted differences as extreme as observed   
    pvalue = (np.sum(samples >= benchmark) + 1) / (len(samples) + 1)

    return metric_a, metric_b, pvalue, benchmark

def perform_permutaton_test_wrapper(model_A_file, model_B_file, y_true=None):
    """Loads model predictions and runs permutation tests for Macro-F1 and Macro-Recall."""
    y_pred_A = np.load(model_A_file)
    y_pred_B = np.load(model_B_file)

    # Run the permutation test for F1 score
    f1_A, f1_B, p_value, _ = permutation_test(macro_f1, y_true, y_pred_A, y_pred_B)

    # Run the permutation test for Recall score
    rec_A, rec_B, _, _ = permutation_test(macro_recall, y_true, y_pred_A, y_pred_B)
    
    return p_value, f1_A, f1_B, rec_A, rec_B

def run_permutation_analysis(prediction_files, true_labels, alpha=0.05):
    """
    Runs permutation tests for all model combinations, applies Holm-Bonferroni correction,
    and saves results to CSV.
    """
    results = []
    for (model_a, file_a), (model_b, file_b) in combinations(prediction_files.items(), 2):
        comp_label = f"{model_a} vs. {model_b}"
        p_val, f1_a, f1_b, rec_a, rec_b = perform_permutaton_test_wrapper(
            file_a, file_b, true_labels
        )
        results.append({
            "comparison": comp_label,
            "p_value": p_val,
            "f1_model_a": f1_a,
            "f1_model_b": f1_b,
            "recall_model_a": rec_a,
            "recall_model_b": rec_b
        })

    if not results:
        print("No model comparisons were performed. Please check the input files.")
        return

    p_values = [r["p_value"] for r in results]
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method="holm")

    for i, res in enumerate(results):
        res["p_corrected"] = pvals_corrected[i]
        res["significant"] = reject[i]

    df = pd.DataFrame(results)

    column_order = [
        "comparison", "f1_model_a", "f1_model_b",
        "recall_model_a", "recall_model_b",
        "p_value", "p_corrected", "significant"
    ]

    return df[column_order].sort_values(by="p_corrected", ascending=True)