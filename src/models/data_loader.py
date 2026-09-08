import pandas as pd

def format_pca_val(val):
    """Formats float/int PCA values to integer string equivalents."""
    if val is None:
        return ""
    if isinstance(val, float):
        return str(int(round(val * 10)))
    return str(val)

def filter_embed_columns(df, args):
    """Filters DataFrame columns to only include specified embedding columns."""

    non_feature_columns = ['CallID', 'IQMedian']

    text_model = args.pretrained_text_model
    speech_model = args.pretrained_speech_model

    text_tag = text_model.replace('-', '_').lower()
    speech_tag = speech_model.replace('-', '_').lower()

    columns_to_drop = []

    for col in df.columns:
        if 'non_pca' in col.lower() or col in non_feature_columns:
            continue  # Skip non-embedding columns
        if text_tag not in col.lower() and speech_tag not in col.lower():
            columns_to_drop.append(col)

    print(f"Columns dropped due to pretrained model filtering: {columns_to_drop}")
   
    return df.drop(columns=columns_to_drop, errors='ignore')

def load_processed_data(input_dir, args, feature_split="train"):
    """Loads a specified processed feature DataFrame from the specified path."""

    dataset_type = args.dataset_type

    system_pca = format_pca_val(args.systemf_text_pca)
    speech_text_pca = format_pca_val(args.speechf_text_pca)
    speech_wav_pca = format_pca_val(args.speechf_wav_pca)

    filename = f"{feature_split}_{dataset_type}_sytxtpca{system_pca}_sptxtpca{speech_text_pca}_spwpca{speech_wav_pca}"

    filename += ".csv"

    file_path = input_dir / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Feature file {file_path} does not exist. Please check the dataset type and feature type.")
    
    df = pd.read_csv(file_path)
    filtered_df = filter_embed_columns(df, args)

    print(f"Loaded {feature_split} data from '{file_path}' with shape {filtered_df.shape}")

    return filtered_df