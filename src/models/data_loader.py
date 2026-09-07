import pandas as pd
from pathlib import Path



def load_processed_data(
        input_dir, dataset_type, feature_split, 
        pca_n_comp_dict={}, pretrained_model_dict={}):
    """
    Loads a specified processed feature DataFrame from the specified path.
    """

    for value in pca_n_comp_dict.values():
        if isinstance(value, float):
            pca_n_comp_dict[value] = f"{value:.1f}"
        elif isinstance(value, int):
            pca_n_comp_dict[value] = str(value)
    # Check if the PCA values are floats for proper filename formatting
    if isinstance(pca_speechf_text_n, float):
        pca_speechf_text_n_str = f"{pca_speechf_text_n:.1f}"
    else:
        pca_speechf_text_n_str = str(pca_speechf_text_n)

    if isinstance(pca_systemf_text_n, float):
        pca_systemf_text_n_str = f"{pca_systemf_text_n:.1f}"
    else:
        pca_systemf_text_n_str = str(pca_systemf_text_n)
        
    if isinstance(pca_speechf_speech_n, float):
        pca_speechf_speech_n_str = f"{pca_speechf_speech_n:.1f}"
    else:
        pca_speechf_speech_n_str = str(pca_speechf_speech_n)
    
    # Identify and load target csv file based on dataset type and feature split
    filename = f"{feature_split}_{dataset_type}"

    if dataset_type == "system_features":
        filename += f"_sytxtpca{pca_systemf_text_n}_sptxtpca_spwpca.csv"
    elif dataset_type == "speech_features":
        filename += f"_sytxtpca_sptxtpca{pca_speechf_text_n_str}_spwpca{pca_speechf_speech_n_str}.csv"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be 'system_features' or 'speech_features'.")
    
    file_path = input_dir / filename
    print(f"Loading {feature_split} data from: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"Feature file {file_path} does not exist. Please check the dataset type and feature type.")
    
    df = pd.read_csv(file_path)

    # Filter dataframe based on the pretrained models to be used
    text_model = pretrained_model_dict.get('pretrained_text_model', None)
    speech_model = pretrained_model_dict.get('pretrained_speech_model', None)

    text_model_tag = text_model.replace('-', '_')
    speech_model_tag = speech_model.replace('-', '_')

    columns_to_drop = []

    for col in df.columns:
        if 'emb' not in col.lower() and 'dim' not in col.lower():
            continue  # Skip non-embedding columns
        if text_model_tag.lower() not in col.lower() and speech_model_tag.lower() not in col.lower():
            columns_to_drop.append(col)
    df_filtered = df.drop(columns=columns_to_drop, errors='ignore') 
    print(f"Columns dropped due to pretrained model filtering: {columns_to_drop}")
    print(f"Loaded {feature_split} data from '{file_path}' with shape {df_filtered.shape}")
    return df_filtered