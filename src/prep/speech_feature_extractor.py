#---------------------------------- Imports -------------------------------------------------
import re
import random
import pandas as pd
import numpy as np
import os
import soundfile as sf
import opensmile
import torch
import torchaudio
from torchaudio.transforms import Resample
from sentence_transformers import SentenceTransformer
from pathlib import Path
from collections import defaultdict
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoConfig, AutoFeatureExtractor

#---------------------------------- Functions -------------------------------------------------
def clean_text(text, is_agent = False):
    """Clean and preprocess the input text."""

    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)

    # Replace non-breaking spaces, newlines, and tabs with standard spaces
    text = text.replace('\xa0', ' ').replace('\n', ' ').replace('\t', ' ')

    # Replace multiple spaces with a single space and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()

    if is_agent:
        # For agent prompts and transcripts, capitalise the letter in bus numbers like 61a -> 61A
        text = re.sub(r'(\d+)([a-z])(?=\W|$)', lambda m: m.group(1) + m.group(2).upper(), text)

        # Handle title casing for fully capitalized words except exceptions
        def title_case_with_exceptions(word):
            word_stripped = re.sub(r'\W+$', '', word)
            if word_stripped == "CMU":
                return word
            elif re.fullmatch(r'\d+[a-zA-Z]', word_stripped):
                return word_stripped.upper() + word[len(word_stripped):] 
            elif word_stripped.isupper() and len(word_stripped) > 1:
                return word_stripped.capitalize() + word[len(word_stripped):]
            return word

        words = text.split()
        text = ' '.join([title_case_with_exceptions(word) for word in words])

    return text


def combine_user_agent_transcripts(df_agent, df_user, output_path = None):
    """Combine agent and user transcript dataframes into a single DataFrame."""

    # Clean df_agent and Filter out null transcripts
    df_agent_filtered = df_agent[df_agent['Transcript'].notna()].copy()

    # Filter df_user to only include call IDs present in df_agent_filtered
    callids_to_include = df_agent_filtered['CallID'].unique()
    df_user_filtered = df_user[df_user['CallID'].isin(callids_to_include)].copy()

    # Clean text in both dataframes to be consistent
    df_agent_filtered['Transcript'] = df_agent_filtered['Transcript'].apply(lambda x: clean_text(x, is_agent=True))
    df_user_filtered['Transcript'] = df_user_filtered['Transcript'].apply(lambda x: clean_text(x, is_agent=False))

    # Combine dataframes
    df_combined = pd.concat([df_agent_filtered, df_user_filtered], ignore_index=True)

    # Sort df by CallID and StartTime
    df_combined_sorted = df_combined.sort_values(by=['CallID', 'StartTime'], ascending=True).reset_index(drop=True)

    print(f"Combined dataset shape: {df_combined_sorted.shape}")

    # Choose whether to save df as csv at this point in processing
    if output_path:
        df_combined_sorted.to_csv(output_path, index=False)
        print(f"Combined data saved to: {output_path}")

    return df_combined_sorted

def extract_turn_taking_features(df):

    # 1. Turn Durations
    df["AgentTurnDuration"] = df["AgentEndTime"] - df["StartTime"]
    df["UserTurnDuration"] = df["UserEndTime"] - df["UserStartTime"]
    df["UserTurnDuration"] = df["UserTurnDuration"].fillna(0)

    df["ExchangeDuration"] = df["AgentTurnDuration"] + df["UserTurnDuration"].fillna(0)

    df["TurnTakingRatio"] = df["UserTurnDuration"] / df["AgentTurnDuration"]
    df["TurnTakingRatio"] = df["TurnTakingRatio"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 2. User Response Latency and Overlap
    df["UserResponseLatency"] = df["UserStartTime"] - df["AgentEndTime"]

    df["UserIsOverlap"] = df["UserResponseLatency"] < 0
    df["UserOverlapDuration"] = df["UserResponseLatency"].apply(
        lambda x: 0 if pd.isna(x) else (abs(x) if x < 0 else 0)
    )
    df["UserResponseLatency"] = df["UserResponseLatency"].fillna(0)

    # 3. Agent Response Latency and Overlap
    df['PreviousUserEndTime'] = df.groupby('CallID')['UserEndTime'].shift(1)
    df['AgentResponseLatency'] = df['StartTime'] - df['PreviousUserEndTime']
    df["AgentIsOverlap"] = df["AgentResponseLatency"] < 0
    df["AgentOverlapDuration"] = df["AgentResponseLatency"].apply(
        lambda x: 0 if pd.isna(x) else (abs(x) if x < 0 else 0)
    )
    df['AgentResponseLatency'] = df['AgentResponseLatency'].fillna(0)

    # 4. Agent Interruption by User
    df['UserInterruptsAgent'] = False
    df['PromptCompletionRatio'] = np.nan

    def check_interruption(row):
        is_interrupted = False
        prompt_completion_ratio = np.nan

        # Condition 1: User overlap exists
        user_overlapped = row['UserIsOverlap']

        # Condition 2: Agent's actual utterance is a truncated prefix of the intended prompt
        transcript_is_shorter = len(row['AgentTranscript']) < len(row['AgentPrompt'])
        
        # Calculate prompt completion ratio
        if len(row['AgentPrompt']) > 0:
            prompt_completion_ratio = len(row['AgentTranscript']) / len(row['AgentPrompt'])
        else:
            prompt_completion_ratio = 1.0 if len(row['AgentTranscript']) == 0 else 0.0

        # Define interruption based on both conditions
        if user_overlapped and transcript_is_shorter:
            is_interrupted = True

        return is_interrupted, prompt_completion_ratio

    # Apply the function row-wise
    df[['UserInterruptsAgent', 'PromptCompletionRatio']] = df.apply(
        lambda row: check_interruption(row), axis=1, result_type='expand'
    )

    df = df.drop(columns=['PreviousUserEndTime'])

    return df

def combine_exchange_level_data(df, response_token_list = None, duration_threshold = 1.0):
    """Convert utterance-level DataFrame into exchange-level DataFrame."""

    exchange_data = []

    # Group the DataFrame by 'CallID' to process each conversation
    for callid, group in df.groupby('CallID'):
        # Separate agent and user turns for easier access
        agent_turns = group[group['Speaker'] == 'agent'].reset_index(drop=True)
        user_turns = group[group['Speaker'] == 'user'].reset_index(drop=True)

        # Iterate through each agent turn to form an exchange
        for i in range(len(agent_turns)):
            agent_row = agent_turns.iloc[i]
            agent_transcript = str(agent_row['Transcript']).strip()

            next_agent_start_time = float('inf') # Set to infinity if it's the last agent turn
            if i + 1 < len(agent_turns):
                next_agent_start_time = agent_turns.iloc[i+1]['StartTime']
            
            # Initialize dictionary for the current exchange, starting with agent data
            current_exchange = {
                'CallID': callid,
                'StartTime': agent_row['StartTime'],
                'EndTime': next_agent_start_time if next_agent_start_time != float('inf') else agent_row['EndTime'],
                'CombinedTranscript': agent_transcript,
                'AgentEndTime': agent_row['EndTime'],
                'AgentTranscript': agent_transcript,
                'AgentPrompt': agent_row['Prompt'],
                'UserStartTime': pd.NA,
                'UserEndTime': pd.NA,
                'ResponseTokenCount': 0,
                'NumConsecutiveUserUtterances': 0,
                'HasConsecutiveUserUtterances': False,
                'IQMedian': agent_row['IQMedian']
            }
                
            # Filter user responses that occur between the current agent and the next agent
            relevant_user_responses = user_turns[
                (user_turns['StartTime'] > agent_row['StartTime']) & # User starts after agent's start
                (user_turns['StartTime'] < next_agent_start_time)
            ].sort_values(by='StartTime').reset_index(drop=True)
                
            # If relevant user responses are found
            if not relevant_user_responses.empty:

                internal_pause_durations = []
                previous_end_time = None

                num_consecutive_user_utterances = len(relevant_user_responses)
                boolean_consecutive_user_utterances = (num_consecutive_user_utterances > 1)
                earliest_user_start = relevant_user_responses['StartTime'].min()
                latest_user_end = relevant_user_responses['EndTime'].max()

                response_token_count = 0
                for idx, row in relevant_user_responses.iterrows():
                    transcript_lower = str(row['Transcript']).strip().lower()
                    turn_duration = row['EndTime'] - row['StartTime']
                    if response_token_list and \
                        (transcript_lower in response_token_list) and \
                        (turn_duration < duration_threshold):
                        response_token_count += 1
                    
                    if previous_end_time is not None:
                        pause_duration = round(row["StartTime"] - previous_end_time, 2)
                        if pause_duration > 0:
                            internal_pause_durations.append(pause_duration)
                                
                        previous_end_time = row['EndTime']

                # Combine all user transcripts
                combined_user_transcript = " ".join(str(row['Transcript']).strip() for idx, row in relevant_user_responses.iterrows())

                current_exchange["CombinedTranscript"] = (f"{agent_transcript} [SEP] {combined_user_transcript}")
                current_exchange['UserStartTime'] = earliest_user_start
                current_exchange['UserEndTime'] = latest_user_end
                current_exchange["ResponseTokenCount"] = response_token_count
                current_exchange["NumConsecutiveUserUtterances"] = num_consecutive_user_utterances
                current_exchange["HasConsecutiveUserUtterances"] = boolean_consecutive_user_utterances

                if internal_pause_durations:
                    current_exchange['NumInternalPauses'] = len(internal_pause_durations)
                    current_exchange['AvgInternalPauseDuration'] = np.mean(internal_pause_durations)
                    current_exchange['TotalInternalPauseDuration'] = np.sum(internal_pause_durations)
                    current_exchange['StdInternalPauseDuration'] = np.std(internal_pause_durations)
                else:
                    current_exchange['NumInternalPauses'] = 0.0
                    current_exchange['AvgInternalPauseDuration'] = 0.0 
                    current_exchange['TotalInternalPauseDuration'] = 0.0 
                    current_exchange['StdInternalPauseDuration'] = 0.0 
                    

                if next_agent_start_time == float('inf'):
                    # If this is the last exchange, the EndTime must be the latest event in the exchange (the user's end time)
                    latest_user_end = relevant_user_responses['EndTime'].max()
                    current_exchange['EndTime'] = latest_user_end

            else:
                current_exchange['NumInternalPauses'] = 0 
                current_exchange['AvgInternalPauseDuration'] = 0.0 
                current_exchange['TotalInternalPauseDuration'] = 0.0 
                current_exchange['StdInternalPauseDuration'] = 0.0 
            
            exchange_data.append(current_exchange)

    # Convert the list of exchange data into a new DataFrame
    exchange_df = pd.DataFrame(exchange_data)

    df_turn_take = extract_turn_taking_features(exchange_df)

    return df_turn_take

def extract_text_embeddings(df, column, pretrained_text_models, device = None):
    """
    Function to extract text embeddings using pretrained transformer models.
    """
    all_text_embed_dfs = []

    for model_name in pretrained_text_models:

        model_tag = model_name.split('/')[-1].replace('-', '_')

        print(f"Extracting embeddings using model: {model_tag}")

        if "minilm" in model_name.lower() or "sbert" in model_name.lower():
            model = SentenceTransformer(model_name, device = device)
            embeddings_array = model.encode(
                df[column].astype(str).tolist(), 
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32 
            )

        else:

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)

            model.eval()

            embeddings = []
            
            for text in df[column].astype(str).tolist():
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding.flatten())

            embeddings_array = np.array(embeddings)

        prefix= f"{column}_{model_tag}_Dim"

        embedding_dimension = embeddings_array.shape[1]
        column_names = [f"{prefix}{i}" for i in range(embedding_dimension)]
        embeddings_df = pd.DataFrame(embeddings_array, index=df.index, columns=column_names)

        all_text_embed_dfs.append(embeddings_df)

    if all_text_embed_dfs:
        df = pd.concat([df] + all_text_embed_dfs, axis = 1)
        print(f"Extracted text embeddings using models: {pretrained_text_models}")
    else:
        print("No text embeddings were extracted.")

    return df


def produce_speech_embeds(audio_file_path, callid, start_time, end_time, model, processor, device):
    """Produce Wav2Vec embeddings for a given audio segment."""

    # Load audio file
    waveform, sr = torchaudio.load(audio_file_path)

    # Preprocessing audio - Resampling
    if sr != 16000:
        waveform = Resample(orig_freq=sr, new_freq=16000)(waveform)
        sr = 16000

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Segment time-to-sample conversion and clipping
    total_samples = waveform.shape[-1]

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    start_sample = max(0, min(start_sample, total_samples - 1))
    end_sample = max(start_sample + 1, min(end_sample, total_samples))

    segment = waveform[:, start_sample:end_sample]

    # Generate Wav2Vec embeddings
    inputs = processor(segment.squeeze(0), sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

    mean_pooled_embedding = torch.mean(last_hidden_states, dim=1)

    exchange_embedding = mean_pooled_embedding.squeeze(0).cpu().numpy()

    return exchange_embedding


def acoustic_feature_extraction(audio_files_dict, df, pretrained_speech_models = None, device = None):
    """Extract acoustic embeddings and OpenSMILE features from audio_files."""

    # Opensmile Model Initialisation
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    all_opensmile_feature_names = [f'Opensmile{col}' for col in smile.feature_names]    
    
    # Initialise list to hold all acoustic embeddings
    all_speech_embed_dfs = []

    # Initialise placeholder for OpenSMILE df
    opensmile_df = None

    # Loop through each pretrained speech model
    for model_name in pretrained_speech_models:

        model_tag = model_name.split('/')[-1].replace('-', '_')
        model_prefix = f"{model_tag}_Emb"

        print(f'Extracting speech embeddings using model: {model_tag}')

        config = AutoConfig.from_pretrained(model_name)

        if config.model_type == "wavlm" or config.model_type == "hubert":
            speech_processor = AutoFeatureExtractor.from_pretrained(model_name)
        else:
            speech_processor = AutoProcessor.from_pretrained(model_name)
        
        speech_model = AutoModel.from_pretrained(model_name).to(device)
        speech_embed_dims = speech_model.config.hidden_size  # Dynamically get embedding dimension
        speech_model.eval()

        current_model_features = []

        # Iterate through each row (exchange)
        for index, row in df.iterrows():
            current_callid = str(int(row['CallID']))
            start = row['StartTime']
            end = row['EndTime']

            audio_path = audio_files_dict.get(current_callid)["dyad"]

            # OpenSMILE Feature Extraction
            if opensmile_df is None:
                if audio_path is None:
                    opensmile_features_dict = {col: np.nan for col in all_opensmile_feature_names}
                else:
                    try:
                        features_df = smile.process_file(audio_path, start=start, end=end)

                        if not features_df.empty:
                            features_series = features_df.iloc[0] 
                            opensmile_features_dict = {f'Opensmile{k}': v for k, v in features_series.to_dict().items()}
                        else:
                            print(f"No OpenSMILE features extracted for {audio_path} from {start}-{end}")
                            opensmile_features_dict = {col: np.nan for col in all_opensmile_feature_names}

                    except Exception as e:
                        print(f"Error processing OpenSMILE for {audio_path} from {start}-{end}: {e}")
                        opensmile_features_dict = {col: np.nan for col in all_opensmile_feature_names}

            # Speech Embedding Generation (Run for every model in loop)
            if audio_path is None:
                speech_features_dict = {f'{model_prefix}{i}': np.nan for i in range(speech_embed_dims)}
            else:
                try:
                    speech_embeds = produce_speech_embeds(audio_file_path= audio_path,
                                                            callid = current_callid, 
                                                            start_time= start, 
                                                            end_time=end, 
                                                            model = speech_model, 
                                                            processor= speech_processor, 
                                                            device = device)

                    speech_features_dict = {f'{model_prefix}{i}': val for i, val in enumerate(speech_embeds.flatten())}

                except Exception as e:
                    print(f"Unhandled error during Wav2Vec embedding generation for {audio_path} from {start}-{end}: {e}")
                    speech_features_dict = {f'{model_prefix}{i}': np.nan for i in range(speech_embed_dims)}
        
            current_data = {
                'OriginalIndex': index,
            }

            current_data.update(speech_features_dict)

            if opensmile_df is None:
                current_data.update(opensmile_features_dict)

            current_model_features.append(current_data)

        current_model_df = pd.DataFrame(current_model_features).set_index('OriginalIndex')

        if opensmile_df is None:
            opensmile_cols = [col for col in current_model_df.columns if col.startswith('Opensmile')]
            opensmile_df = current_model_df[opensmile_cols]
        
        speech_embed_cols = [col for col in current_model_df.columns if col.startswith(model_tag)]
        all_speech_embed_dfs.append(current_model_df[speech_embed_cols])

    df_final = pd.merge(df, opensmile_df, left_index=True, right_index=True, how='left')

    for speech_embed_df in all_speech_embed_dfs:
        df_final = pd.merge(df_final, speech_embed_df, left_index=True, right_index=True, how='left')

    print("\n All Acoustic features and embeddings extracted and merged.")
    return df_final
    

def prepare_features_for_ml(df, null_values, output_path = None):

    df["ExchangeDuration"] = df["EndTime"] - df["StartTime"]

    columns_to_drop = ['CombinedTranscript', 'StartTime', 'EndTime', 'AgentTranscript', 'AgentPrompt', 'UserStartTime', 'UserEndTime', 'AgentEndTime']
    df_filtered = df.drop(columns=columns_to_drop, errors='ignore')
    
    df_filtered = df_filtered.replace(null_values, pd.NA, regex=False)
    
    print(f"Number of NaNs in prepare_features_for_ml: {df_filtered.isnull().sum().sum()}")
    print("Columns with NaNs in prepare_features_for_ml:")
    print(df_filtered.isnull().sum()[df_filtered.isnull().sum() > 0])

    # Identify rows with NaNs
    rows_with_nan = df_filtered[df_filtered.isnull().any(axis=1)]
    print("Rows with NaNs in prepare_features_for_ml:")
    print(rows_with_nan)

    if output_path:
        output_path = Path(output_path)
        df_filtered.to_csv(output_path, index=False)
        print(f"Feature Set for saved to {output_path}")
    
#---------------------- Core Processing Function --------------------

def run_feature_extraction_pipeline(audio_files_dict, df_agent, df_user, output_filepaths, config_dict):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_transcripts_combined = combine_user_agent_transcripts(df_agent, df_user, 
                                                            output_path= output_filepaths["combined_transcript_no_embeddings"])

    df_exchange_level = combine_exchange_level_data(df_transcripts_combined,
                                                    response_token_list = config_dict["response_tokens"],
                                                    duration_threshold = config_dict["duration_threshold"])   

    df_with_text_emb = extract_text_embeddings(df_exchange_level, column= "CombinedTranscript", 
                                            pretrained_text_models= config_dict["pretrained_model_text"], device= device)

    df_features = acoustic_feature_extraction(audio_files_dict, df_with_text_emb, 
                                            pretrained_speech_models= config_dict["pretrained_model_speech"], 
                                            device= device)

    prepare_features_for_ml(df= df_features, null_values=config_dict["null_vals"], 
                            output_path = output_filepaths["combined_transcript_with_embeddings"])