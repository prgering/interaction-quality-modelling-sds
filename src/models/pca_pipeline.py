import pandas as pd
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, clone
from sklearn.compose import ColumnTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import optimize_dataframe_memory, get_df_size


def load_feature_data(dataset_path, dataset_type, remove_first_turn = False):
    """Loads a specified feature DataFrame from the specified path."""
    df = pd.read_csv(dataset_path)
    
    if remove_first_turn:
        df['ExchangeNum'] = df.groupby('CallID').cumcount() + 1
        df = df[df['ExchangeNum'] > 2].copy()

        df.drop(columns=['ExchangeNum'], inplace=True)
        
    # Optimize the DataFrame memory usage
    print(f"Size of {dataset_type} df: {get_df_size(df):.4f} GB", flush = True)
    print(f"Optimizing {dataset_type} df...", flush = True)
    df = optimize_dataframe_memory(df)
    print(f"Size of {dataset_type} df (after optimization): {get_df_size(df):.4f} GB\n", flush = True)

    nan_columns = df.isna().sum()
    print(f"{dataset_type} df columns with missing values:")
    print(nan_columns[nan_columns > 0])
    return df



def train_opt_split(df, grouping_col='CallID', train_size=0.8, random_state=42):
    """Splits the dataframe into training and optimization sets based on unique call IDs."""
    if grouping_col not in df.columns:
        raise ValueError(f"Grouping column '{grouping_col}' not found in DataFrame.")

    unique_callids = df[grouping_col].unique()
    train_filecodes, opt_filecodes = train_test_split(
        unique_callids, train_size=train_size, random_state=random_state
    )

    train_df = df[df[grouping_col].isin(train_filecodes)].copy()
    opt_df = df[df[grouping_col].isin(opt_filecodes)].copy()

    return train_df, opt_df


def run_pre_processing_steps(train_df, test_df, pca_params_dict):

    non_feature_columns = ['CallID', 'IQMedian']
    feature_columns = [col for col in train_df.columns if col not in non_feature_columns]

    X_train = train_df[feature_columns]
    y_train = train_df['IQMedian']

    X_test = test_df[feature_columns]
    y_test = test_df['IQMedian']

    nan_columns = X_train.isna().sum()
    print("X_train columns with missing values:")
    print(nan_columns[nan_columns > 0])

    # Define preprocessing pipelines for different feature groups
    speechf_text_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_params_dict['n_comp_speechf_text']))
    ])
    speechf_wav_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_params_dict['n_comp_speechf_wav']))
    ])
    systemf_text_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_params_dict['n_comp_systemf_text']))
    ])
    # Map rule conditions and pipelines to feature groups
    TRANSFORMER_CONFIG = {
        'utterance_sbert':   ({'inc': ['utterance', 'minilm']}, systemf_text_pipeline),
        'utterance_roberta': ({'inc': ['utterance', 'roberta']}, systemf_text_pipeline),
        'utterance_todbert': ({'inc': ['utterance', 'tod_bert']}, systemf_text_pipeline),

        'prompt_sbert':      ({'inc': ['prompt', 'minilm']}, systemf_text_pipeline),
        'prompt_roberta':    ({'inc': ['prompt', 'roberta']}, systemf_text_pipeline),
        'prompt_todbert':    ({'inc': ['prompt', 'tod_bert']}, systemf_text_pipeline),

        'speechf_sbert':     ({'inc': ['minilm'],   'exc': ['utterance', 'prompt']}, speechf_text_pipeline),
        'speechf_roberta':   ({'inc': ['roberta'],  'exc': ['utterance', 'prompt']}, speechf_text_pipeline),
        'speechf_todbert':   ({'inc': ['tod_bert'], 'exc': ['utterance', 'prompt']}, speechf_text_pipeline),

        'speechf_wav2vec':   ({'inc': ['wav2vec']}, speechf_wav_pipeline),
        'speechf_hubert':    ({'inc': ['hubert']}, speechf_wav_pipeline),
        'speechf_wavlm':     ({'inc': ['wavlm']}, speechf_wav_pipeline),
    }

    def match_columns(col_name, rule):
        """Checks if a column matches inclusion and exclusion constraints."""
        col_lower = col_name.lower()
        inc_match = all(inc in col_lower for inc in rule.get('inc', []))
        exc_match = not any(exc in col_lower for exc in rule.get('exc', []))
        return inc_match and exc_match 

    # Build ColumnTransformer with pipelines for each feature group
    transformers_list = []
    matched_pca_cols = set()
    for name, (rule, pipeline) in TRANSFORMER_CONFIG.items():
        matched_cols = [col for col in feature_columns if match_columns(col, rule)]
        if matched_cols:
            transformers_list.append((name, clone(pipeline), matched_cols))
            matched_pca_cols.update(matched_cols)

    # Handle non-PCA columns (those not matched by any rule)
    non_pca_cols = [
        col for col in feature_columns
        if col not in matched_pca_cols
    ]
    if non_pca_cols:
        transformers_list.append(('non_pca', StandardScaler(), non_pca_cols))

    # Fit and transform feature data
    preprocessor = ColumnTransformer(transformers_list, remainder='drop')
    
    preprocessor.fit(X_train)

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    processed_feature_names = preprocessor.get_feature_names_out()

    dataset_dfs = {
        "train": pd.DataFrame(X_train_processed, columns=processed_feature_names),
        "test": pd.DataFrame(X_test_processed, columns=processed_feature_names)
    }

    for col in non_feature_columns:
        if col in train_df.columns:
            if col == 'IQMedian':
                dataset_dfs['train'][col] = y_train.values - 1
                dataset_dfs['test'][col] = y_test.values - 1
            else:
                dataset_dfs['train'][col] = train_df[col].values
                dataset_dfs['test'][col] = test_df[col].values

    return dataset_dfs