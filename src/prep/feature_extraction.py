#---------------------------------- Imports -------------------------------------------------
import re
import pandas as pd
import numpy as np
import sys
import opensmile
import torch
import torchaudio
from torchaudio.transforms import Resample
from sentence_transformers import SentenceTransformer
from pathlib import Path
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoConfig, AutoFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.align_system_prompts import CONFIG
from src.utils import get_filepaths

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


def combine_transcripts(df_agent, df_user, output_path = None):
    """Combine agent and user transcript dataframes into a single DataFrame."""

    # Filter out null transcripts from df_agent
    df_agent_filtered = df_agent[df_agent['Transcript'].notna()].copy()

    # Filter df_user to only include call IDs present in df_agent_filtered
    callids_to_include = set(df_agent_filtered['CallID'].dropna().unique())
    df_user_filtered = df_user[
        df_user['Transcript'].notna() &
        df_user['CallID'].isin(callids_to_include)
    ].copy()

    # Clean text in both dataframes
    df_agent_filtered['Transcript'] = df_agent_filtered['Transcript'].apply(clean_text, is_agent=True)
    df_user_filtered['Transcript'] = df_user_filtered['Transcript'].apply(clean_text, is_agent=False)

    # Combine and sort dataframes
    df_combined = (
        pd.concat([df_agent_filtered, df_user_filtered], ignore_index=True)
        .sort_values(by=['CallID', 'StartTime'], ascending=True)
        .reset_index(drop=True)
    )

    print(f"Combined dataset shape: {df_combined.shape}")

    # Choose whether to save df as csv at this point in processing
    if output_path:
        df_combined.to_csv(output_path, index=False)
        print(f"Combined data saved to: {output_path}")

    return df_combined

def extract_turn_taking_features(df):

    # 1. Turn Durations
    df["AgentTurnDuration"] = df["AgentEndTime"] - df["StartTime"]
    df["UserTurnDuration"] = (df["UserEndTime"] - df["UserStartTime"]).fillna(0)
    df["ExchangeDuration"] = df["AgentTurnDuration"] + df["UserTurnDuration"].fillna(0)

    df["TurnTakingRatio"] = ((
        df["UserTurnDuration"] / df["AgentTurnDuration"]).replace(
            [np.inf, -np.inf], np.nan).fillna(0)
    )

    # 2. User Response Latency and Overlap
    df["UserResponseLatency"] = df["UserStartTime"] - df["AgentEndTime"]
    df["UserIsOverlap"] = df["UserResponseLatency"] < 0
    df["UserOverlapDuration"] = (-df["UserResponseLatency"]).clip(lower=0).fillna(0)
    df["UserResponseLatency"] = df["UserResponseLatency"].fillna(0)

    # 3. Agent Response Latency and Overlap
    df['PreviousUserEndTime'] = df.groupby('CallID')['UserEndTime'].shift(1)
    df['AgentResponseLatency'] = df['StartTime'] - df['PreviousUserEndTime']
    df["AgentIsOverlap"] = df["AgentResponseLatency"] < 0
    df["AgentOverlapDuration"] = (-df["AgentResponseLatency"]).clip(lower=0).fillna(0)
    df['AgentResponseLatency'] = df['AgentResponseLatency'].fillna(0)

    # 4. Agent Interruption by User & Prompt Completion Ratio
    agent_len = df['AgentTranscript'].fillna('').astype(str).str.len()
    prompt_len = df['AgentPrompt'].fillna('').astype(str).str.len()

    df['UserInterruptsAgent'] = df['UserIsOverlap'] & (agent_len < prompt_len)

    df['PromptCompletionRatio'] = np.where(
        prompt_len > 0,
        agent_len / prompt_len,
        np.where(agent_len == 0, 1.0, 0.0)
    )

    return df.drop(columns=['PreviousUserEndTime'])

def combine_exchange_level_data(df, response_token_list = None, duration_threshold = 1.0):
    """
    Convert utterance-level DataFrame into exchange-level DataFrame.
    
    Note that the old version of this function did not calculate InternalPause features correctly
    due to incorrect indentation of the previous_end_time assignment. This version fixes that issue
    and calculates features correctly.
    """

    # Convert token list to set for faster membership checking
    response_tokens = set(response_token_list) if response_token_list else None
    exchange_data = []

    # Group the DataFrame by 'CallID' to process each conversation
    for callid, group in df.groupby('CallID'):
        # Separate agent and user turns for easier access
        agent_turns = group[group['Speaker'] == 'agent'].to_dict('records')
        user_turns = group[group['Speaker'] == 'user'].to_dict('records')

        # Iterate through each agent turn to form an exchange
        for i, agent_row in enumerate(agent_turns):

            agent_transcript = str(agent_row['Transcript']).strip()

            # Determine boundary for next agent turn
            if i + 1 < len(agent_turns):
                next_agent_start_time = agent_turns[i+1]['StartTime']
            else:
                next_agent_start_time = float('inf')
            
            # Initialize dictionary for the current exchange, starting with agent data
            current_exchange = {
                'CallID': callid,
                'StartTime': agent_row['StartTime'],
                'EndTime': (
                    next_agent_start_time 
                    if next_agent_start_time != float('inf') 
                    else agent_row['EndTime']
                ),
                'CombinedTranscript': agent_transcript,
                'AgentEndTime': agent_row['EndTime'],
                'AgentTranscript': agent_transcript,
                'AgentPrompt': agent_row['Prompt'],
                'UserStartTime': pd.NA,
                'UserEndTime': pd.NA,
                'ResponseTokenCount': 0,
                'NumConsecutiveUserUtterances': 0,
                'HasConsecutiveUserUtterances': False,
                'NumInternalPauses': 0.0,
                'AvgInternalPauseDuration': 0.0,
                'TotalInternalPauseDuration': 0.0,
                'StdInternalPauseDuration': 0.0,
                'IQMedian': agent_row['IQMedian']
            }
                
            # Filter user responses between current agent and next agent start
            relevant_user_responses = [
                u for u in user_turns
                if agent_row['StartTime'] < u['StartTime'] < next_agent_start_time
            ]
                
            # If relevant user responses are found
            if relevant_user_responses:
                relevant_user_responses.sort(key=lambda x: x['StartTime'])

                earliest_user_start = min(u['StartTime'] for u in relevant_user_responses)
                latest_user_end = max(u['EndTime'] for u in relevant_user_responses)
                                             
                internal_pause_durations = []
                previous_end_time = None
                response_token_count = 0

                for user_turn in relevant_user_responses:
                    transcript_lower = str(user_turn['Transcript']).strip().lower()
                    turn_duration = user_turn['EndTime'] - user_turn['StartTime']

                    if response_tokens and \
                        (transcript_lower in response_tokens) and \
                        (turn_duration < duration_threshold):
                        response_token_count += 1
                    
                    if previous_end_time is not None:
                        pause_duration = round(
                            user_turn["StartTime"] - previous_end_time, 2
                        )

                        if pause_duration > 0:
                            internal_pause_durations.append(pause_duration)
                                
                    previous_end_time = user_turn['EndTime']

                # Combine all user transcripts
                combined_user_transcript = " ".join(str(u['Transcript']).strip() for u in relevant_user_responses)

                current_exchange["CombinedTranscript"] = (f"{agent_transcript} [SEP] {combined_user_transcript}")
                current_exchange['UserStartTime'] = earliest_user_start
                current_exchange['UserEndTime'] = latest_user_end
                current_exchange["ResponseTokenCount"] = response_token_count
                current_exchange["NumConsecutiveUserUtterances"] = len(relevant_user_responses)
                current_exchange["HasConsecutiveUserUtterances"] = len(relevant_user_responses) > 1

                if internal_pause_durations:
                    current_exchange['NumInternalPauses'] = float(len(internal_pause_durations))
                    current_exchange['AvgInternalPauseDuration'] = np.mean(internal_pause_durations)
                    current_exchange['TotalInternalPauseDuration'] = np.sum(internal_pause_durations)
                    current_exchange['StdInternalPauseDuration'] = np.std(internal_pause_durations)

                if next_agent_start_time == float('inf'):
                    current_exchange['EndTime'] = latest_user_end
            
            exchange_data.append(current_exchange)

    # Convert the list of exchange data into a new DataFrame
    exchange_df = pd.DataFrame(exchange_data)

    return extract_turn_taking_features(exchange_df)

def extract_text_embeddings(df, column, pretrained_text_models, device=None, batch_size=32):
    """
    Function to extract text embeddings using pretrained transformer models.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = df.reset_index(drop=True)
    texts = df[column].astype(str).tolist()
    all_text_embed_dfs = []

    for model_name in pretrained_text_models:
        model_tag = model_name.split('/')[-1].replace('-', '_')
        print(f"Extracting embeddings using model: {model_tag}")

        # Optimised path for SentenceTransformer models
        if "minilm" in model_name.lower() or "sbert" in model_name.lower():
            model = SentenceTransformer(model_name, device = device)
            embeddings_array = model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size 
            )
        # HuggingFace pipeline with batched processing for other transformer models
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            model.eval()

            embeddings_list = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                # Extract [CLS] tokens across the batch
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings_list.append(cls_embeddings)
            
            embeddings_array = np.vstack(embeddings_list)

        # Format embedding feature columns
        prefix= f"{column}_{model_tag}_Dim"

        column_names = [f"{prefix}{i}" for i in range(embeddings_array.shape[1])]
        embeddings_df = pd.DataFrame(embeddings_array, index=df.index, columns=column_names)

        all_text_embed_dfs.append(embeddings_df)

    if all_text_embed_dfs:
        df = pd.concat([df] + all_text_embed_dfs, axis = 1)
        print(f"Extracted text embeddings using models: {pretrained_text_models}")
    else:
        print("No text embeddings were extracted.")

    return df


def produce_speech_embeds(
        audio_file_path, start_time, end_time, model, processor, device
    ):
    """Produce Wav2Vec embeddings for a given audio segment."""

    # Read audio file metadata
    metadata = torchaudio.info(audio_file_path)
    original_sr = metadata.sample_rate
    total_frames = metadata.num_frames

    # Compute sample indices for the segment
    start_frame = int(start_time * original_sr)
    end_frame = int(end_time * original_sr)

    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))
    num_frames = end_frame - start_frame

    # Load audio segment directly using the computed frame indices
    waveform, sr = torchaudio.load(
        audio_file_path, frame_offset=start_frame, num_frames=num_frames
    )

    # Convert stereo to mono if necessary
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample segment to 16kHz if necessary
    if sr != 16000:
        waveform = Resample(orig_freq=sr, new_freq=16000)(waveform)
        sr = 16000

    # Pad extremely short segments to prevent CNN layer dimension errors
    min_samples = 400 
    if waveform.shape[-1] < min_samples:
        waveform = torch.nn.functional.pad(
            waveform, (0, min_samples - waveform.shape[-1])
        )

    # Process and run inference
    raw_audio_np = waveform.squeeze(0).cpu().numpy()

    inputs = processor(raw_audio_np, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

    mean_pooled_embedding = torch.mean(last_hidden_states, dim=1)

    return mean_pooled_embedding.squeeze(0).cpu().numpy()


def acoustic_feature_extraction(
        audio_files_dict, 
        df, 
        pretrained_speech_models = None, 
        device = None,
        skip_embeddings = False
    ):
    """Extract acoustic embeddings and OpenSMILE features from audio_files."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. OpenSMILE Feature Extraction
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    all_opensmile_feature_names = [f'Opensmile{col}' for col in smile.feature_names]    

    opensmile_rows = []

    for index, row in df.iterrows():
        current_callid = str(int(float(row['CallID']))) if not pd.isna(row['CallID']) else ""
        start, end = row['StartTime'], row['EndTime']
        audio_path = audio_files_dict.get(current_callid, {}).get("dyad")

        if audio_path:
            try:
                features_df = smile.process_file(audio_path, start=start, end=end)

                if not features_df.empty:
                    feat_dict = {
                        f'Opensmile{k}': v 
                        for k, v in features_df.iloc[0].to_dict().items()
                    }
                else:
                    print(f"No OpenSMILE features extracted for {audio_path} from {start}-{end}")
                    feat_dict = {col: np.nan for col in all_opensmile_feature_names}

            except Exception as e:
                print(f"Error processing OpenSMILE for {audio_path} from {start}-{end}: {e}")
                feat_dict = {col: np.nan for col in all_opensmile_feature_names}
            
        else:
            feat_dict = {col: np.nan for col in all_opensmile_feature_names}
            
        feat_dict['OriginalIndex'] = index
        opensmile_rows.append(feat_dict)

    opensmile_df = pd.DataFrame(opensmile_rows).set_index('OriginalIndex')

    # 2. Speech Embedding Extraction (Skipped if skip_embeddings is True)
    all_speech_embed_dfs = []

    if not skip_embeddings:
        for model_name in (pretrained_speech_models or []):

            model_tag = model_name.split('/')[-1].replace('-', '_')
            model_prefix = f"{model_tag}_Emb"

            print(f'Extracting speech embeddings using model: {model_tag}')

            # config = AutoConfig.from_pretrained(model_name)
            # if config.model_type in ["wavlm", "hubert"]:
            #     speech_processor = AutoFeatureExtractor.from_pretrained(model_name)
            # else:
            #     speech_processor = AutoProcessor.from_pretrained(model_name)
            
            speech_processor = AutoFeatureExtractor.from_pretrained(model_name)
            speech_model = AutoModel.from_pretrained(model_name).to(device)
            speech_model.eval()
            speech_embed_dims = speech_model.config.hidden_size  # Dynamically get embedding dimension
            
            current_model_features = []

            # Iterate through each row (exchange)
            for index, row in df.iterrows():
                current_callid = str(int(float(row['CallID']))) if not pd.isna(row['CallID']) else ""
                start, end = row['StartTime'], row['EndTime']

                audio_path = audio_files_dict.get(current_callid, {}).get("dyad")

                # Speech Embedding Generation (Run for every model in loop)
                if audio_path and not pd.isna(start) and not pd.isna(end):
                    try:
                        speech_embeds = produce_speech_embeds(
                            audio_file_path= audio_path, 
                            start_time= start, end_time=end, 
                            model = speech_model, processor= speech_processor, 
                            device = device
                        )

                        speech_features_dict = {
                            f'{model_prefix}{i}': val 
                            for i, val in enumerate(speech_embeds.flatten())
                        }

                    except Exception as e:
                        print(f"Unhandled error during Wav2Vec embedding generation for {audio_path} from {start}-{end}: {e}")
                        speech_features_dict = {
                            f'{model_prefix}{i}': np.nan 
                            for i in range(speech_embed_dims)
                        }
                        
                else:
                    speech_features_dict = {
                        f'{model_prefix}{i}': np.nan 
                        for i in range(speech_embed_dims)
                    }

                speech_features_dict['OriginalIndex'] = index
                current_model_features.append(speech_features_dict)

            model_df = pd.DataFrame(current_model_features).set_index('OriginalIndex')
            all_speech_embed_dfs.append(model_df)

            # Clear VRAM between models
            del speech_model, speech_processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Merge all features into a single DataFrame
    df_final = pd.concat([df, opensmile_df] + all_speech_embed_dfs, axis=1)
    print("\n All Acoustic features and embeddings extracted and merged.")
    return df_final
    

def prepare_features_for_ml(df, null_values, output_path = None):
    """
    Prepare extracted features by computing additional features, dropping
    unnecessary columns, and handling null values. Optionally save the resulting 
    DataFrame to a CSV file.
    """

    df_copy = df.copy()

    # Compute Exchange Duration if StartTime and EndTime are present
    if "StartTime" in df_copy.columns and "EndTime" in df_copy.columns:
        df_copy["ExchangeDuration"] = df_copy["EndTime"] - df_copy["StartTime"]

    # Drop metadata/transcript columns
    columns_to_drop = [
        'CombinedTranscript', 'StartTime', 'EndTime', 'AgentTranscript', 
        'AgentPrompt', 'UserStartTime', 'UserEndTime', 'AgentEndTime'
    ]

    df_filtered = df_copy.drop(columns=columns_to_drop, errors='ignore')

    # Standardise null values 
    df_filtered = df_filtered.replace(null_values, pd.NA, regex=False)

    # Compute NaN metrics
    null_counts = df_filtered.isnull().sum()
    total_nans = null_counts.sum()
    cols_with_nans = null_counts[null_counts > 0].index.tolist()
    rows_with_nan = df_filtered[df_filtered.isnull().any(axis=1)]

    print(f"Total NaN values in the feature set: {total_nans}")
    if total_nans > 0:
        print(f"Columns with NaN values: {cols_with_nans}")
        print(f"Number of rows with NaN values: {len(rows_with_nan)}")
        print("Rows with NaNs in prepare_features_for_ml:")
        print(rows_with_nan)

    # Export to CSV if output_path is provided
    if output_path:
        df_filtered.to_csv(output_path, index=False)
        print(f"Feature Set for saved to {output_path}")

    return df_filtered
    
#---------------------- Core Processing Function --------------------

def run_feature_extraction_pipeline(
        audio_dir, 
        df_agent, 
        df_user,
        output_filepaths,
        config_dict, 
        device = None,
        force = False,
        skip_embeddings = False,
        debug = False
    ):

    combined_transcript_path = output_filepaths.get("combined_transcript")

    if force or not Path(combined_transcript_path).exists():
        print("Stage 1: Combining agent and user transcripts...")
        df_combined = combine_transcripts(
            df_agent, df_user, 
            output_path=output_filepaths["combined_transcript"]
        )
    else:
        df_combined = pd.read_csv(combined_transcript_path)
        print(f"Loaded existing combined transcript from {combined_transcript_path}")

    if Path(output_filepaths["speech_features"]).exists() and not force:
        print(
            "Speech features already exist at "
            f"{output_filepaths['speech_features']}. Skipping extraction."
        )
        return
    
    print("Stage 2: Converting utterance-level data to exchange-level and " \
          "extracting turn-taking features...")
    
    df_exchange = combine_exchange_level_data(
        df_combined,
        response_token_list = config_dict["response_tokens"],
        duration_threshold = config_dict["duration_threshold"]
    ) 

    text_models = config_dict["debug_model_text"] if debug else config_dict["pretrained_model_text"]
    speech_models = config_dict["debug_model_speech"] if debug else config_dict["pretrained_model_speech"]

    if skip_embeddings:
        print("Skipping text embedding extraction as per configuration.")
        df_text_emb = df_exchange
    else:
        print("Stage 3: Extracting text embeddings...")
        df_text_emb = extract_text_embeddings(
            df_exchange, 
            column= "CombinedTranscript", 
            pretrained_text_models=text_models, 
            device= device
        )

    print("Stage 4: Extracting acoustic features...")

    audio_files_dict = get_filepaths(
        directory_dict = {"audio": audio_dir}, 
        folder_to_process = "audio"
    )

    if skip_embeddings:
        print("Only extracting OpenSMILE features as per configuration.")
    
    df_features = acoustic_feature_extraction(
        audio_files_dict, 
        df_text_emb, 
        pretrained_speech_models= speech_models, 
        device= device,
        skip_embeddings=skip_embeddings
    )

    print("Stage 5: Preparing features for ML...")

    df_final = prepare_features_for_ml(
        df= df_features,
        null_values=config_dict["null_vals"], 
        output_path = output_filepaths["speech_features"]
    )

    print(f"Feature extraction pipeline completed.")
    print(f"Final feature set shape: {df_final.shape}")