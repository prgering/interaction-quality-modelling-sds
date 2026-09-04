import itertools
import argparse
import os

# Specify Experiment Name
parser = argparse.ArgumentParser(description="Generate hyperparameter combinations for hyperparameter tuning.")

parser.add_argument(
    "--experiment_name",
    type=str,
    default="speech_lstm",
    help="Name of the experiment (default: 'speech_lstm')."
)

args = parser.parse_args()
experiment_name = args.experiment_name

print(f"Generating hyperparameter combinations for experiment: {experiment_name}")

# Define the hyperparameter options
hidden_size_options = [128, 256, 384]
bidirectional_options = [False, True]
use_attention_options = [False, True]
auto_features_only_options = [False, True]
speechf_text_pca_options = [0.5, 0.6, 0.7, 0.8]
systemf_text_pca_options = [0.5, 0.6, 0.7, 0.8]
speechf_speech_pca_options = [0.5, 0.6, 0.7, 0.8]
text_embedding_model_options = ["sbert", "roberta", "todbert"]
speech_embedding_model_options = ["wav2vec", "hubert", "wavlm"]

# Hyperparameters for end-to-end model
num_lstm_layers_options = [2]
head_lr_options = [5e-4]
window_size_options = [5, 10, 15]
transformer_lr_options = [5e-6, 1e-5, 5e-5]
num_frozen_layers_sbert_options = [0, 3, 6]
num_frozen_layers_speech_encoders_options = [0, 6, 12]

combinations = []

if experiment_name == "system_svm":
    combinations = itertools.product(
        auto_features_only_options,
        systemf_text_pca_options,
        text_embedding_model_options,
    )
elif experiment_name == "speech_svm":
    combinations = itertools.product(
        speechf_text_pca_options,
        speechf_speech_pca_options,
        text_embedding_model_options,
        speech_embedding_model_options,
    )
elif experiment_name == "system_lstm":
    combinations = itertools.product(
        use_attention_options,
        bidirectional_options,
        hidden_size_options,
        auto_features_only_options,
        systemf_text_pca_options,
        text_embedding_model_options,
    )
elif experiment_name == "speech_lstm":
    combinations = itertools.product(
        use_attention_options,
        bidirectional_options,
        hidden_size_options,
        speechf_text_pca_options,
        speechf_speech_pca_options,
    )
elif experiment_name == "system_end2end":
    combinations = itertools.product(
        window_size_options,
        transformer_lr_options,
        num_frozen_layers_sbert_options,
    )
elif experiment_name == "speech_end2end":
    combinations = itertools.product(
        window_size_options,
        transformer_lr_options,
        num_frozen_layers_speech_encoders_options,
    )

# Define the output directory
output_directory = "/mnt/parscratch/users/acp23prg/iq_lego_repo/slurm/configs"
os.makedirs(output_directory, exist_ok=True)

output_filename = f"hyperparameter_combinations_{experiment_name}.txt"
output_file_path = os.path.join(output_directory, output_filename)

# Write combinations to the file
with open(output_file_path, 'w') as f:
    for combo in combinations:
        f.write(" ".join(map(str, combo)) + "\n")

print("Successfully wrote all hyperparameter combinations to the file.")