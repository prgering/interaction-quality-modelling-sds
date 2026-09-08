import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def calc_metrics(
        actual_labels, pred_vals, dataset_type, model_type, 
        print_confusion_matrix=False, base_path=None, top_n = None, 
        data_split = "test", svm_kernel = None
    ):
    """
    Calculates and prints the classification report and confusion matrix for 
    the given predictions and actual labels.
    """
    results = classification_report(
        actual_labels, pred_vals, 
        digits=4, output_dict=True, 
        zero_division=0
    )

    print(f"Dataset: {dataset_type}\n")
    print(f"Classification Report:\n{classification_report(actual_labels, pred_vals, digits=4)}")

    # if print_confusion_matrix:
    #     if base_path is None:
    #         print("Warning: 'base_path' must be provided to save the confusion matrix.")
    #     else:

    #         cm_actual_labels = [int(label) + 1 for label in actual_labels]
    #         cm_pred_vals = [int(label) + 1 for label in pred_vals]

    #         parent_folder = base_path / "experiments/output/confusion_matrices"
    #         os.makedirs(parent_folder, exist_ok=True)

    #         top_n_tag = f"_top{top_n}" if top_n is not None else "_all"

    #         kernel_tag = f"_{svm_kernel}" if svm_kernel else ""

    #         display_labels = sorted(list(set(cm_actual_labels) | set(cm_pred_vals)))

    #         cm_normalized = confusion_matrix(cm_actual_labels, cm_pred_vals, normalize='true')
    #         disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=display_labels)
    #         disp_normalized_result = disp_normalized.plot(values_format=".2f", cmap='Blues')

    #         if hasattr(disp_normalized_result, 'im_'):
    #             disp_normalized_result.im_.set_clim(0.0, 1.0)
            
    #             if hasattr(disp_normalized_result, 'colorbar') and disp_normalized_result.colorbar is not None:
    #                 disp_normalized_result.figure_.colorbar(disp_normalized_result.im_, ax=disp_normalized_result.axes)

    #         filename_norm = f"cm_{dataset_type}_{model_type}{kernel_tag}{top_n_tag}_normalized.png"
    #         filepath_norm = parent_folder / filename_norm
    #         plt.savefig(filepath_norm, dpi=600)
    #         plt.close()
    #         print(f"Normalized confusion matrix saved to: {filepath_norm}")
    
    return results
