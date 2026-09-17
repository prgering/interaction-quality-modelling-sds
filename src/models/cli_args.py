import argparse

def add_feature_set_args(parser: argparse.ArgumentParser):
    """
    Adds common command-line arguments to a given parser object.
    """
    parser.add_argument("dataset_type",
                        choices=["system", "speech"],
                        default="system",
                        help="Specify the feature set to use."
    )

    parser.add_argument("--pretrained_text_model", type=str, default="sbert")
    parser.add_argument("--pretrained_speech_model",type=str,default="wav2vec2")

def add_pca_args(parser: argparse.ArgumentParser):
    """
    Adds PCA-related command-line arguments to a given parser object.
    """

    parser.add_argument("--systemf_text_pca", type=float, default=0.0)
    parser.add_argument("--speechf_text_pca", type=float, default=0.0)
    parser.add_argument("--speechf_wav_pca", type=float, default=0.0)

def add_svm_args(parser: argparse.ArgumentParser):
    """
    Adds SVM-specific command-line arguments to a given parser object.
    """
    parser.add_argument("--gamma", type=float, nargs='+', default=[0.01, 0.1, 1, 10])
    parser.add_argument("--c", type=float, nargs='+', default=[0.1, 1, 10, 100])
    parser.add_argument("--kernel", type=str, nargs='+', default=["rbf", "linear"])

def add_lstm_args(parser: argparse.ArgumentParser):
    """
    Adds LSTM-specific command-line arguments to a given parser object.
    """

    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument("--lr", type=float, nargs='+', default=[0.001, 0.0005])
    parser.add_argument("--batch_size", type=int, nargs='+', default=[5, 15, 25])
    parser.add_argument("--epochs", type=int, default=250)

def add_fine_tuning_args(parser: argparse.ArgumentParser):
    """
    Adds fine-tuning specific command-line arguments to a given parser object.
    """

    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--transformer_lr", type=float, default=1e-6)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--num_frozen_layers", type=int, default=9)