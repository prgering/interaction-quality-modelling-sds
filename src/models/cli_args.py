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

    parser.add_argument("--auto_features_only", action="store_true")
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


    # @staticmethod
    # def parse_fine_tuned_lstm_args():
    #     """
    #     Parses command line arguments for the fine-tuned LSTM model.

    #     Returns:
    #         tuple: A tuple containing the parsed arguments and dictionaries.
    #     """

    #     parser = argparse.ArgumentParser(description="Train fine-tuned LSTM model with specific hyperparameter combinations.")
        
    #     # Add parser arguments
    #     ArgParser._add_feature_set_args(parser)
    #     ArgParser._add_lstm_args(parser)


    #     parser.add_argument(
    #         "--window_size",
    #         type=int,
    #         default=10,
    #         help="Specify the window size for training (default: 5, 10, 15)."
    #     )

    #     parser.add_argument(
    #         "--transformer_lr",
    #         type=float,
    #         default=1e-6,
    #         help="Specify the learning rate for the transformer (default: 1e-6, 5e-6, 1e-5)."
    #     )

    #     parser.add_argument(
    #         "--head_lr",
    #         type=float,
    #         default=5e-4,
    #         help="Specify the learning rate for the classification head (default: 5e-4, 1e-3, 5e-3)."
    #     )

    #     parser.add_argument(
    #         "--num_frozen_layers",
    #         type=int,
    #         default=9,
    #         help="Specify the number of frozen layers in the transformer (default: 9)."
    #     )

    #     args = parser.parse_args()

    #     dataset_type = args.dataset_type
    #     auto_features_only = args.auto_features_only

    #     num_layers = args.num_layers[0] if isinstance(args.num_layers, list) else args.num_layers

    #     lstm_hyperparam_dict = {
    #         "hidden_size": args.hidden_size,
    #         "num_layers": num_layers,
    #         "bidirectional": args.bidirectional,
    #         "use_attention": args.use_attention,
    #         "window_size": args.window_size,
    #         "transformer_lr": args.transformer_lr,
    #         "head_lr": args.head_lr,
    #         "num_frozen_layers": args.num_frozen_layers
    #     }

    #     return dataset_type, lstm_hyperparam_dict, auto_features_only

    # @staticmethod
    # def parse_svm_args():
    #     """
    #     Parses command line arguments for the SVM model.
        
    #     Returns:
    #         tuple: A tuple containing the parsed arguments and dictionaries.
    #                (dataset_type, svm_hyperparam_dict, pca_hyperparam_dict)
    #     """
    #     parser = argparse.ArgumentParser(description="Train an SVM model with specified feature set.")

    #     ArgParser._add_feature_set_args(parser)
    #     ArgParser._add_pca_args(parser)
    #     ArgParser._add_svm_args(parser)

    #     args = parser.parse_args()

    #     dataset_type = args.dataset_type
    #     auto_features_only = args.auto_features_only
        
    #     svm_hyperparam_dict = {
    #         "C": args.c,
    #         "gamma": args.gamma,
    #         "kernel": args.kernel,
    #     }

    #     pca_hyperparam_dict = {
    #         "n_components_systemf_text": args.systemf_text_pca,
    #         "n_components_speechf_text": args.speechf_text_pca,
    #         "n_components_speechf_speech": args.speechf_speech_pca
    #     }

    #     pretrained_model_dict = {
    #         "pretrained_text_model": args.pretrained_text_model,
    #         "pretrained_speech_model": args.pretrained_speech_model
    #     }

    #     return dataset_type, svm_hyperparam_dict, pca_hyperparam_dict, auto_features_only, pretrained_model_dict

