# Imports
import numpy as np
import pandas as pd
import re
import torch

from collections import defaultdict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

class DataCleaner:
    """Clean the System-derived features prior to preprocessing."""
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
            df["Activity"].astype(str).str.strip().replace(
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
    def __init__(self, df = None, config = None, device = None):
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.config = config if config is not None else {}
        self.device = device

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
        """
        # Extract Semantic Parse information into seperate binary variables
        sem_keys = self.extract_keys_sem_parse()

        self.df.loc[
            self.df['SemanticParse'] == "Semantic no match", 'SemanticParse'
        ] = "semantic_no_match"

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
        self.extract_text_embeddings()

        # Drop columns unnecessary for classification
        drop_cols = (
            self.config.get("cols_to_drop", [])
        )
        self.df = self.df.drop(drop_cols, axis=1)

        # Exclude window-level and dialogue-level features
        self.df = self.df.filter(regex=r'^[^#%]+$')

        # Dummy Coding
        self.df = pd.get_dummies(
            self.df, columns=self.config.get("dummy_cols", []), 
            drop_first=True
        )

        return self.df