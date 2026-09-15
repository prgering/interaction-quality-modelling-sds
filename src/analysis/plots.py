import seaborn as sns
import matplotlib.pyplot as plt

def create_box_plots(df, output_dir = None):
    """
    Generates box plots to visualize the distribution of Macro F1 scores from hyperparameter tuning.
    Args:
        df (pd.DataFrame): DataFrame containing model results.
        output_dir (str): Directory to save the generated plots.
    """
    sns.set_context("paper", font_scale=1.4) 
    sns.set_style("whitegrid")

    df_copy = df.copy()

    feature_order = ["system_auto", "speech_both"]
    model_order = ["svm", "lstm"]
    x_ticks_labels = ["SD", "SA"]
    model_labels = ["SVM", "LSTM"]

    plt.figure(figsize=(8, 5))

    flier_props = dict(marker="o", markersize=4, alpha=0.5, markeredgewidth=0.5)

    ax = sns.boxplot(
        data=df_copy,
        x="dataset_type",
        y="f1",
        hue="model_type",
        order=feature_order,
        hue_order=model_order,
        palette="Greys",
        width=0.7,
        flierprops=flier_props,
        medianprops={"color": "black", "linewidth": 1.5}
    )

    plt.xlabel("Feature Set", fontweight="bold")
    plt.ylabel("Macro F1", fontweight="bold")
    plt.xticks(range(2), x_ticks_labels)
    plt.ylim(0, 0.65)

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles, 
        model_labels, 
        title=None, 
        loc="upper center",
        bbox_to_anchor=(0.5, 1.125),
        ncol=2, 
        frameon=False,
    )

    plt.tight_layout()

    save_path = f"{output_dir}/ht_box_plot.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {save_path}")