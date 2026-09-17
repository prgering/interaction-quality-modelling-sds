# Imports
import numpy as np
import os
import pandas as pd
import re
import sys
import torch

from collections import defaultdict
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import is_valid_audio

class DataCleaner:
    """Clean system-log data prior to preprocessing."""
    def __init__(self, df = None, config = None):
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.config = config if config is not None else {}

    def standardise_data_types_and_labels(self):
        """Standardise specific columns in the system-derived feature set."""
        df = self.df

        # Numeric conversion of DD column
        df["DD"] = df["DD"].astype(str).str.strip()
        df["DD"] = pd.to_numeric(df["DD"], errors='coerce')

        # Remove leading/trailing whitespace and non-breaking spaces
        target_string_cols = self.config.get(
            "string_cols", ["Prompt", "Utterance", "SemanticParse"])
        
        for col in target_string_cols:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str)
                    .str.replace('\xa0', ' ')
                    .str.strip()
                )

        # Address inconsistent category names and strings
        mappings = {
            "ASRRecognitionStatus": {
                'no input': 'timeout',
                'no match': 'reject',
                'complete': 'success'
            },
            "Modality": {
                'voice': 'speech'
            }
        }

        for col, replace_map in mappings.items():
            if col in df.columns:
                df[col] = df[col].replace(replace_map)

        df["ASRRecognitionStatus"] = (
            df["ASRRecognitionStatus"].replace(r"^$", "no status", regex=True)
        )

        # Normalize Activity Labels
        prefix_to_remove = "/LetsGoPublic/PerformTask/"
        pattern = r'^' + re.escape(prefix_to_remove)

        df["Activity"] = (
            df["Activity"].astype(str).str.strip().str.replace(
                pat=pattern, repl='', regex=True)
        )

        # Standardise CallID format
        ids = df['CallID'].astype(str)
        df['CallID'] =  (
            ids.str.slice(0, 2) + '0' + ids.str.slice(2)
        )

        self.df = df

    def address_missing_values(self):
        """Address missing values via imputation and filtering."""
        df = self.df
        null_values = self.config.get("null_vals", [])

        # Identify NA's and replace with pd.NA
        if null_values:
            df = df.replace(null_values, pd.NA)

        # Remove NAs in IQMedian Column
        df = df[df['IQMedian'].notna()]

        # Replace NAs in Utterance Column with ""
        df["Utterance"] = df["Utterance"].fillna("").astype(str)

        # Replace NAs in DD column using group-wise ffill and bfill
        df['DD'] = df.groupby('CallID')['DD'].ffill().bfill()

        # Replace NAs in other columns with "missing"
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("missing")

        self.df = df

    def run_cleaning_pipeline(self):
        """Run the full cleaning pipeline on the LEGO corpus"""

        self.standardise_data_types_and_labels()
        
        self.address_missing_values()

        return self.df

class SemanticParser:
    """
    Parses hierarchical semantic strings into nested dictionaries.
    """
    def __init__(self):
        self.nested_counts = defaultdict(lambda: defaultdict(int))
    
    def _build_nested_structure(self, tokens, start_index):
        """
        Iteratively traverses tokens to build a nested dictionary representing 
        the hierarchical semantic structure.
        """
        stack = []
        current_structure = defaultdict(lambda: defaultdict(int))
        active_key = None
        idx = start_index

        while idx < len(tokens):
            token = tokens[idx]
            # Case 1: Enter a deeper level of hierarchy
            if token == "(":
                new_dict = defaultdict(lambda: defaultdict(int))

                if active_key is not None:
                    if not isinstance(current_structure[active_key], defaultdict):
                        current_structure[active_key] = defaultdict(int)
                    current_structure[active_key] = new_dict
                
                # Save current state before diving into the new structure
                stack.append((current_structure, active_key))
                current_structure = new_dict
                active_key = None
            
            # Case 2: Exit current level of hierarchy
            elif token == ")":
                if stack:
                    current_structure, active_key = stack.pop()
            
            # Case 3: Key-value pair or standalone key
            elif "[" in token and "]" in token:
                match = re.match(r"(\w+)\[(.*?)\]", token)
                if match:
                    key, value = match.groups()
                    if key not in current_structure:
                        current_structure[key] = defaultdict(int)
                    if value not in current_structure[key]:
                        current_structure[key][value] = 0

                    current_structure[key][value] += 1
                    active_key = key
            
            # Case 4: Standalone key (potentially a new active key)
            else:
                if active_key is not None:
                    # If we have an active key, this token is a value of that key
                    if not isinstance(
                        current_structure[active_key], defaultdict
                    ):
                        current_structure[active_key] = defaultdict(int)
                    if token not in current_structure[active_key]:
                        current_structure[active_key][token] = 0
                    current_structure[active_key][token] += 1
                else:
                    # Treat the token as a standalone key
                    if token not in current_structure:
                        current_structure[token] = 0
                    current_structure[token] += 1
                    active_key = token
            idx += 1
        return idx, current_structure
    
    def count_entities(self, parse_string):
        """Entry point to parse a semantic string and return the nested counts"""
        # Standardise no match case
        if parse_string.strip().lower() == "semantic no match":
            parse_string = "semantic_no_match"

        # Regex to match parentheses, key-value pairs, and standalone words
        tokens = re.findall(r"\(|\)|\w+\[.*?\]|\w+", parse_string)

        # Parse tokens iteratively to build nested dictionary
        _, nested_result = self._build_nested_structure(tokens, 0)
        return nested_result

class DataPreprocessor:
    """Preprocess the System-derived features prior to classification."""
    def __init__(self, df = None, config = None, device = None, skip_embedding = False):
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.config = config if config is not None else {}
        self.device = device
        self.skip_embedding = skip_embedding

    def extract_keys_sem_parse(self):
        """
        Identify unique high-level keys in the "SemanticParse" column using 
        the SemanticParser class.
        """
        sem_parser = SemanticParser()
        parses = self.df["SemanticParse"].dropna().tolist()

        # Count entities in each semantic parse
        entity_counts = [sem_parser.count_entities(e) for e in parses]

        # Extract first key from each parse to create list of unique keys
        first_keys = [list(d.keys())[0] if d else "" for d in entity_counts]

        return list(set(first_keys))

    def extract_text_embeddings(self):
        """Extract text embeddings using pretrained transformer models."""
        embedding_dfs = []

        device = self.device if self.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        model_pipeline = self.config.get("pretrained_models", [])
        # Iterate through each model in config
        for model_name in tqdm(model_pipeline, desc="Extracting Text Embeddings", unit="model"):
            # Create model tag for column naming
            model_tag = model_name.split('/')[-1].replace('-', '_')

            if model_tag == "all_MiniLM_L6_v2":
                model_tag = "sbert"

            tokenizer = None
            # Load model and tokenizer
            if any(x in model_name.lower() for x in ["minilm", "sbert"]):
                model = SentenceTransformer(model_name, device=device)
                is_sbert = True
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name).to(device)
                model.eval()
                is_sbert = False

            # Process each target column using the current model
            for column in self.config.get("embed_cols", []):
                input_strings = self.df[column].astype(str).tolist()

                if is_sbert:
                    # SBERT handles batching and conversion internally
                    emb_array = model.encode(
                        input_strings, convert_to_numpy=True,
                        show_progress_bar=False, batch_size=32 
                    )
                
                else:
                    # Manual loop for non-SBERT models
                    txt_embeds = []

                    loop_desc = f" -> Processing '{column}' ({model_tag})"

                    with torch.no_grad():
                        for text in tqdm(input_strings, desc=loop_desc, unit="text"):
                            inputs = tokenizer(
                                text, return_tensors="pt", truncation=True, 
                                padding=True
                            ).to(device)

                            outputs = model(**inputs)

                            cls_emb = (
                                outputs.last_hidden_state[:, 0, :]
                                .detach().cpu().numpy()
                            )
                            
                            txt_embeds.append(cls_emb.flatten())

                    emb_array = np.array(txt_embeds)

                prefix= f"{column}_{model_tag}_dim"

                # Generate unique feature names for each embedding dimension
                col_names = [
                    f"{prefix}{i}" for i in range(emb_array.shape[1])
                ]

                # Map embeddings to Dataframe
                emb_df = pd.DataFrame(emb_array, index=self.df.index, 
                    columns=col_names
                )

                embedding_dfs.append(emb_df)

        # Join all new embedding columns to the original dataframe
        if embedding_dfs:
            self.df = pd.concat([self.df] + embedding_dfs, axis = 1)

    def run_preprocessing_pipeline(self):
        """
        Run preprocessing pipeline to extract semantic parse information,
        derive text embeddings and drop unnecessary columns.

        This method was updated to ensure all window-level and dialogue-level 
        features were excluded (including those that started with 'Mean_'). 
        """
        # Extract Semantic Parse information into seperate binary variables
        sem_keys = self.extract_keys_sem_parse()

        for key in sem_keys:
            if key:
                col_name = f"{key}_present"
                pattern = rf'\b{re.escape(key)}\b'
                self.df[col_name] = (
                    self.df['SemanticParse']
                    .str.contains(pattern, case=False, regex=True)
                    .fillna(False)
                )
    
        # Derive Text Embeddings for the Utterance and Prompt cols
        if not self.skip_embedding:
            self.extract_text_embeddings()

        # Drop columns unnecessary for classification
        drop_cols = self.config.get("cols_to_drop", [])
        drop_keywords = self.config.get("keywords_to_drop", [])

        exact_drops_lower = {c.lower() for c in drop_cols}
        keywords_lower = [kw.lower() for kw in drop_keywords]

        cols_to_remove = [
            col for col in self.df.columns
            if col.lower() in exact_drops_lower 
            or any(kw in col.lower() for kw in keywords_lower)
        ]

        self.df = self.df.drop(columns=cols_to_remove, errors='ignore')

        # Exclude window-level and dialogue-level features
        self.df = self.df.filter(regex=r'^(?!.*Mean)[^#%]+$')

        # Dummy Coding
        self.df = pd.get_dummies(
            self.df, columns=self.config.get("dummy_cols", []), 
            drop_first=True
        )

        return self.df

class DialogueExcluder:
    """
    Manages the exclusion of audio files from the LEGO corpus.

    Filters the dataset based on the following criteria:
    - Missing or empty .wav audio files.
    - CallIDs listed in a predefined exclusion list.
    - Dialogues where the user does not speak at all.
    """

    def __init__(self, 
                 df_system_features = None, df_user = None, 
                 df_system_transcript = None, audio_folder = None, 
                 config = None
    ):
        """Initialise the DialogueExcluder with dataframes and config"""
        self.df_system_features = (
            df_system_features.copy() 
            if df_system_features is not None else None
        )

        self.user_transcripts = (
            df_user.copy() if df_user is not None 
            else None
        )
        self.system_transcripts = (
            df_system_transcript.copy() if df_system_transcript is not None 
            else None
        )

        self.audio_dir = audio_folder
        
        # Extract config settings
        config = config if config is not None else {}

        self.manual_exclude_list = config.get("excluded_files", [])
        self.null_vals = config.get("null_vals", [])
        self.drop_cols = config.get("cols_to_drop", [])
        
        self.valid_wav_paths = defaultdict(Path)
        self.exclusion_map = defaultdict(str)

    def check_wav_validity(self):
        """
        Scans the audio directory for valid .wav files. 
        Updates self.valid_wav_paths and self.exclusion_map.
        """

        if not self.audio_dir or not Path(self.audio_dir).is_dir():
            raise ValueError(f"Invalid audio folder path: {self.audio_dir}")
        
        # Find all dyadic .wav files in the audio directory
        wav_files = [
            os.path.join(dp, f) 
            for dp, _, files in os.walk(self.audio_dir)
            for f in files 
            if f.startswith("LetsGoPublic") and f.endswith("output.wav")
        ]

        for filepath in sorted(wav_files):
            # Extract 11-digit filecode from filename
            filecode = (re.sub(r"[^0-9]", "", filepath))[-11:]

            is_valid, error_msg = is_valid_audio(filepath, filecode)

            if not is_valid:
                self.exclusion_map[filecode] = error_msg
                continue

            self.valid_wav_paths[filecode] = Path(filepath)


    def validate_callids(self):
        """
        Cross-references CallIDs in the DataFrame with discovered .wav files.
        """
        if self.df_system_features is None:
            raise ValueError("DataFrame is not set.")

        df = self.df_system_features

        # Ensure CallID is a standardised string for comparison
        if pd.api.types.is_float_dtype(df['CallID']):
            df['CallID'] = df['CallID'].astype(int).astype(str)
        elif pd.api.types.is_integer_dtype(df['CallID']):
            df['CallID'] = df['CallID'].astype(str)
        elif pd.api.types.is_object_dtype(df['CallID']):
            df['CallID'] = df['CallID'].astype(str)

        df['CallID'] = df['CallID'].str.strip()

        # Identify CallIDs without valid .wav files
        valid_ids = set(str(k).strip() for k in self.valid_wav_paths.keys())
        all_ids = set(df['CallID'].dropna().unique())
        missing_ids = all_ids - valid_ids

        for call_id in missing_ids:
            self.exclusion_map[call_id] = "No corresponding .wav file found"

        # Keep only rows with valid CallIDs
        self.df_system_features = df[df['CallID'].isin(valid_ids)].copy()

    def exclude_dialogues_manually(self):
        """
        Removes dialogues explicitly listed in the configuration exclusion 
        list.
        """
        for call_id in self.manual_exclude_list:
            if call_id not in self.exclusion_map:
                self.exclusion_map[call_id] = "Manually excluded via config"

        self.df_system_features = self.df_system_features[
            ~self.df_system_features['CallID'].isin(self.manual_exclude_list)
        ]

    def exclude_silent_users(self):
        """
        Excludes dialogues where the user transcript is empty or contains
        only null values.
        """
        if self.user_transcripts is None:
            raise ValueError("User transcript DataFrame is not set.")

        working_user_df = self.user_transcripts.copy()
        
        # Standardise null values and drop empty transcripts
        if self.null_vals:
            working_user_df = working_user_df.replace(self.null_vals, pd.NA)

        working_user_df = (
            working_user_df[working_user_df['Transcript'].notna()]
        )
        working_user_df['CallID'] = (
            working_user_df['CallID'].astype(int).astype(str)
        )

        active_call_ids = set(working_user_df['CallID'].unique())
        
        # Track excluded CallIDs where user does not speak
        for call_id in self.df_system_features["CallID"].unique().tolist():
            if call_id not in active_call_ids:
                self.exclusion_map[call_id] = "User does not speak"

        self.df_system_features = (
            self.df_system_features[self.df_system_features['CallID'].isin(active_call_ids)]
        )        

    def _standardise_prompt(self, prompt):
        """
        Standardises a prompt string for comparison.
        """
        if pd.isna(prompt):
            return prompt

        s = str(prompt).lower()
        s = re.sub(r'\s([?.!,\'"](?:\s|$))', r'\1', s)
        return re.sub(r'\s+', ' ', s).strip()

    def exclude_silent_agents(self):
        """
        Filters dataset to only include rows with matching valid agent
        transcripts.
        """
        if self.system_transcripts is None or self.df_system_features is None:
            raise ValueError("System transcript or system features DataFrame is not set.")

        system_ts = self.system_transcripts.copy()
        main_df = self.df_system_features.copy()

        # 1. Add sequence numbers to help with fuzzy alignment
        main_df['PromptNumber'] = main_df.groupby('CallID').cumcount()
        system_ts['PromptNumber'] = system_ts.groupby('CallID').cumcount()
        

        # 2. Standardise CallID and Speaker formatting
        system_ts['CallID'] = system_ts['CallID'].ffill()
        if 'Speaker' in system_ts.columns:
            system_ts['Speaker'] = system_ts['Speaker'].ffill()

        system_ts['CallID'] = (
            system_ts['CallID']
            .astype(str)
            .str.replace(r'\.0$', '', regex=True)
            .str.strip()
        )

        # 3. Filter both dataframes to only include common CallIDs
        common_call_ids = (
            set(main_df['CallID']).intersection(set(system_ts['CallID']))
        )

        filtered_ds = (
            main_df[
                main_df['CallID'].isin(common_call_ids)
            ].copy()
        )

        filtered_system_ts = (
            system_ts[
                system_ts['CallID'].isin(common_call_ids)
            ].copy()
        )

        # 4. Filter valid transcripts from the agent dataframe
        if self.null_vals:
            filtered_system_ts = (
                filtered_system_ts.replace(
                    self.null_vals, pd.NA, regex=False
                )
            )

        valid_system_ts = filtered_system_ts[
            filtered_system_ts['Transcript'].notna() &
            (filtered_system_ts['Transcript'].str.strip() != "")
        ].copy()

        # Use standardised prompts for matching
        filtered_ds['Prompt'] = filtered_ds['Prompt'].apply(
            self._standardise_prompt
        )
        valid_system_ts['Prompt'] = valid_system_ts['Prompt'].apply(
            self._standardise_prompt
        )

        # 5. Merge based on the anchor columns (CallID, Prompt, IQMedian)
        merged_df = pd.merge(
            valid_system_ts,
            filtered_ds,
            on=['CallID', 'Prompt', 'IQMedian'],
            how='left',
            suffixes=('_system', '_main')
        )

        # 6. Drop rows where the merge failed (i.e., PromptNumber_main is NaN)
        valid_merged_df = merged_df.dropna(subset=['PromptNumber_main']).copy()
        
        # 7. Find the best match from valid rows
        valid_merged_df['Diff'] = abs(
            valid_merged_df['PromptNumber_system'] - 
            valid_merged_df['PromptNumber_main']
        )
        
        best_matches = valid_merged_df.loc[
            valid_merged_df.groupby(
                ['CallID', 'PromptNumber_system']
            )['Diff'].idxmin()
        ]
        
        # 8. Clean up and return the final dataframe
        temp_cols = ['PromptNumber_system', 'PromptNumber_main', 'Diff']
        target_cols_to_drop = set(temp_cols + list(self.drop_cols))
        
        self.df_system_features = best_matches.drop(
            columns=[col for col in target_cols_to_drop if col in best_matches.columns]
        )

    def run_pipeline(self):
        """
        Runs the full exclusion workflow in the correct logical order.
        """
        print("Starting exclusion pipeline...")
        
        self.check_wav_validity()
        self.validate_callids()
        self.exclude_dialogues_manually()
        self.exclude_silent_users()
        self.exclude_silent_agents()
        
        print(f"Pipeline complete. Remaining rows: {len(self.df_system_features)}")
        return self.df_system_features, self.exclusion_map