import torch
import os
import csv
import gc
import sys
import pandas as pd
import whisperx
import torchaudio
import json
import inflect
from rapidfuzz import fuzz
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    get_filepaths, get_asr_config, transcribe_and_align, save_transcript_csv
)
from src.prep.standardise_text import process_text_for_alignment

p = inflect.engine()

#---------------------------------- Functions -------------------------------------------------

class SystemSpeechProcessor:
    def __init__(self,
        config_dict = None,
        directory_dict = None,
        input_files_dict = None,
        output_files_dict = None,
        device = None,
        force = False,
        debug = False
    ):
        """Initialise SystemSpeechProcessor"""
        self.config = config_dict or {}

        self.device = device
        self.force = force
        self.debug = debug

        self.directory_dict = directory_dict or {}
        self.input_files_dict = input_files_dict or {}
        self.output_files_dict = output_files_dict or {}

        self.asr_params_dict = self.config.get("asr_params", {})
        self.alignment_params_dict = self.config.get("alignment_params", {})
    
    # --- Utility Methods ---
    def _get_alignment_params(self):
        """Extracts alignment params from config"""
        d = self.alignment_params_dict
        
        max_window = d.get("max_window", 5)
        proximity_window = d.get("proximity_window", 5)
        user_time_tolerance = d.get("user_time_tolerance", 0.5)
        user_fuzz_threshold = d.get("user_fuzz_threshold", 80)
        word_fuzz_threshold = d.get("word_fuzz_threshold", 60)

        return (
            max_window, 
            proximity_window, 
            user_time_tolerance, 
            user_fuzz_threshold, 
            word_fuzz_threshold
        )
    
    def _get_common_filecodes(self, mixed_dict, agent_df, user_df):
        """Finds intersection of all available filecodes."""
        mixed = set(mixed_dict.keys())
        agent = {str(x) for x in agent_df["CallID"].unique()}
        user = {str(x) for x in user_df["CallID"].unique()}
        return sorted(mixed & agent & user)
    
    def _load_json(self, filepath):
        """Loads JSON file with error handling."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading JSON file '{filepath}': {e}")
            return None
    
    def _prepare_agent_prompts(self, agent_df, filecode):
        """
        Prepares a list of agent prompts for a given filecode.

        Each entry in the list is a tuple containing:
        - The original raw prompt text.
        - The processed prompt text for output.
        - The cleaned prompt text for matching (used in fuzzy alignment).
        """

        subset = agent_df[agent_df["CallID"] == int(filecode)]

        processed_tuples = subset['Prompt'].apply(process_text_for_alignment)

        return [
            (*processed, iq)
            for processed, iq in zip(processed_tuples, subset['IQMedian'])
        ]

    # --- Core Processing Methods ---
    def transcribe_mixed_audio(self, wav_filepaths, output_dir=None):
        """
        Transcribes mixed audio files using WhisperX, including ASR, word-level alignment,
        and speaker diarisation. The results are saved as JSON and plain text files.
        """
        device_str, compute_type, model_type, batch_size = get_asr_config(
                    self.device, self.asr_params_dict, self.debug)

        pyannote_auth_token = self.asr_params_dict.get(
            "pyannote_auth_token", None
        )

        filecodes = wav_filepaths.keys()

        # Load WhisperX ASR, alignment, and diarisation models
        model = whisperx.load_model(model_type, device_str, compute_type=compute_type)

        model_a, metadata = whisperx.load_align_model(
            language_code="en", 
            device=self.device
        )

        diarize_model = whisperx.diarize.DiarizationPipeline(
            token=pyannote_auth_token, 
            device=self.device
        )

        try:
            with tqdm(total= len(filecodes), desc="Transcribing Mixed Audio") as pbar:
                for filecode in filecodes:
                    filepath = str(wav_filepaths[filecode]["dyad"])
                    tqdm.write(f"Processing File: {filecode} | Path: {filepath}")

                    # --- Transcription Pipeline ---
                    # Load audio file with error handling
                    try:
                        waveform, sr = torchaudio.load(filepath)
                    except Exception as e:
                        tqdm.write(f"Error loading audio file '{filepath}': {e}")
                        pbar.update(1)
                        continue

                    # Check sample rate and waveform shape. Adjust if necessary.
                    if sr != 16000:
                        waveform = torchaudio.transforms.Resample(
                            orig_freq=sr, new_freq=16000
                            )(waveform)
                        sr = 16000

                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)

                    audio = waveform.squeeze().numpy()
                    
                    tqdm.write("Performing ASR transcription and alignment...")
                    result = transcribe_and_align(audio, 
                        model, model_a, metadata, batch_size, device_str
                    )
                    
                    tqdm.write("  Performing speaker diarisation and assigning labels...")
                    diarize_segments = diarize_model(
                        audio, min_speakers=2, max_speakers=2
                    )
                    result = whisperx.assign_word_speakers(
                        diarize_segments, result
                    )

                    # Extract full transcript from the results
                    full_text_transcript = " ".join(
                        segment.get("text", "").strip() 
                        for segment in result["segments"]
                    )

                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    
                    # Save JSON output with timestamps and speaker labels
                    json_filepath = os.path.join(output_dir, f"{filecode}.json")
                    with open(json_filepath, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=4, ensure_ascii=False)
                    
                    # Save plain text transcript
                    text_filepath = os.path.join(output_dir, f"{filecode}.txt")
                    with open(text_filepath, "w", encoding="utf-8") as f:
                        f.write(full_text_transcript.strip())

                    pbar.update(1)
        finally:
            # Cleanup to free memory after transcription
            del model, model_a, diarize_model, metadata
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    def identify_user_segments(self, user_transcript_list, dyad_segments_data_list, time_tolerance=0.5, fuzz_threshold=80):
        """
        Identifies user segments within a list of dyad segment dictionaries and updates their 'is_user_segment' flag.
        """

        # ---- Prepare User Utterances ----
        user_utterances_to_match = [
            {
                'Text': process_text_for_alignment(text = u['Transcript'])[2],
                'StartTime': u['StartTime'],
                'EndTime': u['EndTime'],
                'Matched': False
            }
            for u in user_transcript_list
        ]

        identified_user_dyad_indices = []

        # ---- Alignment Loop ----

        # Iterate through dyad segments
        for dyad_idx, seg_data in enumerate(dyad_segments_data_list):
            dyad_cleaned_text_for_matching = seg_data['cleaned_text_for_matching']
            dyad_start = seg_data['start']
            dyad_end = seg_data['end']
            
            # Iterate through (unmatched) user utterances
            for user_utt_idx, user_utt in enumerate(user_utterances_to_match):
                if user_utt['Matched']:
                    continue

                # Check time: Must be within tolerance for both start and end times
                time_match = (
                    abs(dyad_start - user_utt['StartTime']) <= time_tolerance and
                    abs(dyad_end - user_utt['EndTime']) <= time_tolerance
                )

                if time_match:
                    # Check text similarity
                    fuzz_score = fuzz.ratio(dyad_cleaned_text_for_matching, user_utt['Text'])
                    
                    if fuzz_score >= fuzz_threshold:
                        # Match found: Update flags and record index
                        seg_data['is_user_segment'] = True
                        # Use the original text for tracking
                        seg_data['matched_user_utterance'] = user_utt['Text']

                        # Mark user utterance as used
                        user_utterances_to_match[user_utt_idx]['Matched'] = True
                        identified_user_dyad_indices.append(dyad_idx)

                        # Once a dyad segment is matched, move to the next dyad segment
                        break

        return dyad_segments_data_list, identified_user_dyad_indices


    def generate_alignment_candidates(self, agent_prompts_list, dyad_segments_with_words, used_agent_indices, 
                                    used_dyad_indices, max_window, current_agent_start_index, proximity_window=5, 
                                    min_fuzz_score_threshold=60):
        """
        Generates all potential matches for a single agent prompt and dyad segments within a window,
        without marking anything as 'used'.
        """
        potential_matches = []
        dyad_segment_count = len(dyad_segments_with_words)
        agent_idx = current_agent_start_index

        # 1. Pre-check and Setup for Current Agent Prompt
        if used_agent_indices[agent_idx]:
            return []
        
        # Extract all data for the current agent prompt
        agent_data = agent_prompts_list[agent_idx]

        agent_text_original_raw = agent_data[0]
        agent_text_processed_for_output = agent_data[1]
        agent_text_for_matching = agent_data[2]
        iq_median_for_window = agent_data[3]

        agent_indices_in_window = [agent_idx]

        # Identify dyad segments that are not user segments
        agent_dyad_indices = [
            i for i, seg_data in enumerate(dyad_segments_with_words)
            if not seg_data["is_user_segment"]
        ]

        # Create a mapping from original dyad index to its index in the "agent-only" segment list
        # This is used for proximity calculations
        original_to_effective_dyad_index_map = {
            original_idx: effective_idx
            for effective_idx, original_idx in enumerate(agent_dyad_indices)
        }

        # 2. Iterate through Dyad windows
        for dyad_window_size in range(1, max_window + 1):
            for dyad_start_index in range(dyad_segment_count - dyad_window_size + 1):
                dyad_indices_in_window = list(range(dyad_start_index, dyad_start_index + dyad_window_size))
                
                # --- Pruning and Filtering Checks ---

                # Skip if any segment in this window has already been used OR has been identified as a user segment
                if any(used_dyad_indices[idx] or dyad_segments_with_words[idx]["is_user_segment"]
                        for idx in dyad_indices_in_window):
                    continue

                # --- Text Alignment and Initial Score ---
                combined_dyad_text = " ".join(
                    dyad_segments_with_words[idx]['cleaned_text_for_matching'] 
                    for idx in dyad_indices_in_window)
                    
                fuzz_score = fuzz.ratio(agent_text_for_matching, combined_dyad_text)

                if fuzz_score < min_fuzz_score_threshold:
                    continue

                # --- Heuristic Scoring Adjustments ---
                total_score = fuzz_score
                window_len = len(dyad_indices_in_window)

                # Proximity Bonus: Rewards matches that occur close to where the  agent prompt 'should' occur 
                #                  chronologically relative to other agent prompts.
                
                index_proximity_bonus = 0
                first_dyad_original_idx = dyad_indices_in_window[0]

                if (effective_dyad_start_index := original_to_effective_dyad_index_map.get(first_dyad_original_idx)) is not None:
                    distance = abs(current_agent_start_index - effective_dyad_start_index)
                    
                    if distance <= proximity_window:
                        index_proximity_bonus = (proximity_window - distance + 1) * 5 
                
                # Shorter Alignment Bonus (Single segment is preferred)
                shorter_alignment_bonus = 5 if window_len == 1 else 0

                # Long Combination Penalty (Penalize combining many segments)
                long_combination_penalty = 10 if window_len >= 3 else 0
                
                total_score += (index_proximity_bonus + shorter_alignment_bonus - long_combination_penalty)

                # --- Store Candidate ---
                potential_matches.append({
                    "score": total_score,
                    "fuzz_score": fuzz_score,
                    "proximity_bonus": index_proximity_bonus,
                    "agent_indices": agent_indices_in_window,
                    "dyad_indices": dyad_indices_in_window,
                    "agent_text_for_matching": agent_text_for_matching,
                    "agent_text_original_raw": agent_text_original_raw,
                    "agent_text_processed_for_output": agent_text_processed_for_output,
                    "combined_dyad_text": combined_dyad_text,
                    "StartTime": dyad_segments_with_words[dyad_indices_in_window[0]]['start'],
                    "EndTime": dyad_segments_with_words[dyad_indices_in_window[-1]]['end'],
                    "IQMedian": iq_median_for_window,
                })

        return potential_matches

    def align_agent_with_mixed(
            self, 
            mixed_transcript_dict=None,
            agent_df=None,
            user_df=None,
        ):
        """
        Aligns agent prompts with segments of mixed transcript using two
        stages of fuzzy matching.
        """
        # Initialize containers for results
        agent_transcripts = defaultdict(list)
        unmatched_prompts = defaultdict(list)
        files_with_mismatches = set()  

        # Load alignment parameters
        (
            max_window, prox_window, user_time_tolerance, 
            user_fuzz_threshold, word_fuzz_threshold
        ) = self._get_alignment_params()

        # Prepare list of filecodes to process
        filecodes_to_align = self._get_common_filecodes(
            mixed_transcript_dict, agent_df, user_df
        )

        for filecode in tqdm(filecodes_to_align, desc="Aligning agent prompts"):
            # 1. Load and prepare data for current filecode
            mixed_transcript_path = mixed_transcript_dict[filecode]
            dyad_transcript_data = self._load_json(mixed_transcript_path)

            agent_prompts_list = self._prepare_agent_prompts(
                agent_df, filecode
            )
            
            num_agent_prompts = len(agent_prompts_list)

            # Extract and prepare user segments
            current_user_data = user_df[user_df["CallID"] == int(filecode)]
            user_utterances_for_file = current_user_data[
                ['Transcript', 'StartTime', 'EndTime']
                ].to_dict(orient='records')

            # Prepare dyad segments for processing
            dyad_segments_with_words = [
                {
                    "original_text": str(seg.get('text', '')),
                    "processed_text_for_output": processed_output,
                    "cleaned_text_for_matching": matching_text,
                    "start": seg['start'],
                    "end": seg['end'],
                    "words": seg.get('words', []),
                    "original_json_idx": i,
                    "is_user_segment": False,
                    "matched_user_utterance": None
                }
                for i, seg in enumerate(
                    dyad_transcript_data.get("segments", [])
                )
                for _, processed_output, matching_text in [
                    process_text_for_alignment(text = seg.get('text', ''))
                ]
            ]

            num_dyad_segments = len(dyad_segments_with_words)
            
            # -------- Speaker Label Identification --------
            current_agent_speaker_label = None

            first_segment = dyad_transcript_data["segments"][0]
            if words := first_segment.get("words"):
                speakers = defaultdict(int)
                for word_info in words:
                    if speaker := word_info.get("speaker"):
                        speakers[speaker] += 1
                
                if speakers:
                    current_agent_speaker_label = max(speakers, key=speakers.get)

            # ------- User Segment Identification -------
            dyad_segments_with_words, _ = self.identify_user_segments(
                user_utterances_for_file, dyad_segments_with_words,
                time_tolerance=user_time_tolerance, fuzz_threshold=user_fuzz_threshold
            )

            num_dyad_segments = len(dyad_segments_with_words)

            used_agent_indices = [False] * num_agent_prompts
            used_dyad_indices = [False] * num_dyad_segments
            file_segments_for_csv = []
            has_mismatch = False

            # --- First Pass: Global Alignment for Agent Prompts ---
            while True:
                possible_agent_matches = []
                
                # Collect all potential, currently unused matches
                for current_agent_idx_in_loop in range(num_agent_prompts):
                    if not used_agent_indices[current_agent_idx_in_loop]:
                        
                        candidates_current_prompt = self.generate_alignment_candidates(
                            agent_prompts_list, 
                            dyad_segments_with_words, 
                            used_agent_indices, 
                            used_dyad_indices, 
                            max_window, 
                            current_agent_idx_in_loop, 
                            prox_window
                        )

                        possible_agent_matches.extend(candidates_current_prompt)
                
                # Sort all collected candidates by score in descending order to 
                # find the best global match
                possible_agent_matches.sort(
                    key=lambda x: x['score'], reverse=True
                )

                best_match_in_iteration = None

                for match in possible_agent_matches:
                    # Ensure the match uses only UNUSED agent and dyad indices
                    if (all(
                        not used_agent_indices[i] 
                        for i in match['agent_indices']
                    ) and all(
                        not used_dyad_indices[i] 
                        for i in match['dyad_indices']
                    )):
                        
                        # --- Word-Level Validation ---
                        dyad_words_in_match = []
                        for dyad_idx in match['dyad_indices']:
                            dyad_words_in_match.extend(
                                dyad_segments_with_words[dyad_idx]['words']
                            )

                        # If the base score is high, trust the text
                        # and take all words
                        if match['score'] >= 95:
                            agent_words_from_dyad = dyad_words_in_match

                        # If speaker labels are available, filter
                        # to only include words with current speaker
                        elif current_agent_speaker_label:
                            agent_words_from_dyad = [
                                w for w in dyad_words_in_match 
                                if w.get('speaker') == current_agent_speaker_label
                            ]

                        # Reconstruct text from agent words for refined fuzzy matching
                        reconstructed_agent_text_raw = " ".join([w['word'] for w in agent_words_from_dyad]).strip()

                        _, reconstructed_agent_text_processed, reconstructed_agent_text_for_matching = process_text_for_alignment(
                            text = reconstructed_agent_text_raw
                        )

                        # Re-calculate fuzz score with reconstructed text
                        refined_fuzz_score = fuzz.ratio(match["agent_text_for_matching"], reconstructed_agent_text_for_matching)

                        # Only accept if the refined score meets the word-level threshold
                        if refined_fuzz_score >= word_fuzz_threshold: # Use a specific threshold for word-level
                            
                            match['StartTime'] = agent_words_from_dyad[0].get('start')
                            match['EndTime'] = agent_words_from_dyad[-1].get('end')
                            match['transcript_from_audio'] = reconstructed_agent_text_processed
                            
                            best_match_in_iteration = match
                            break # Found the best non-conflicting and word-validated match

                if best_match_in_iteration:
                    # Commit the best match found in this iteration
                    processed_agent_text_for_output = best_match_in_iteration["agent_text_processed_for_output"]

                    file_segments_for_csv.append({
                        "CallID": filecode,
                        "Speaker": "agent",
                        "Transcript": best_match_in_iteration.get('transcript_from_audio', processed_agent_text_for_output),
                        "StartTime": best_match_in_iteration["StartTime"],
                        "EndTime": best_match_in_iteration["EndTime"],
                        "AgentPrompt": processed_agent_text_for_output,
                        "IQMedian": best_match_in_iteration["IQMedian"]
                    })
                    
                    # Mark indices as used
                    for agent_index in best_match_in_iteration["agent_indices"]:
                        used_agent_indices[agent_index] = True
                    for dyad_index in best_match_in_iteration["dyad_indices"]:
                        used_dyad_indices[dyad_index] = True
                else:
                    break

            # --- Second Pass: Handle remaining unmatched agent prompts ---
            # Any prompt that wasn't matched in the first pass is recorded without timing.
            for agent_idx in range(num_agent_prompts):
                if not used_agent_indices[agent_idx]:
                    processed_unmatched_prompt_for_output = agent_prompts_list[agent_idx][1]
                    unmatched_prompts[filecode].append(processed_unmatched_prompt_for_output)
                    file_segments_for_csv.append({
                        "CallID": filecode,
                        "Speaker": "agent",
                        "Transcript": processed_unmatched_prompt_for_output, 
                        "StartTime": None,
                        "EndTime": None,
                        "AgentPrompt": processed_unmatched_prompt_for_output,
                        "IQMedian": agent_prompts_list[agent_idx][3]
                    })
                    print(f"Could not find good match for agent prompt: '{processed_unmatched_prompt_for_output}'")
                    has_mismatch = True

            # --- 5. Final Output Compilation ---
            # Sort all segments (user and agent) by StartTime for chronological output
            file_segments_for_csv.sort(key=lambda x: x['StartTime'] if x['StartTime'] is not None else float('inf'))
            agent_transcripts[filecode] = file_segments_for_csv

            if has_mismatch:
                files_with_mismatches.add(filecode)

            print(f"Aligned agent transcripts for File: {filecode}")

        return agent_transcripts, unmatched_prompts, files_with_mismatches

    def save_unmatched_to_json(self, unmatched_dict, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unmatched_dict, f, indent=4)
        print(f"\nUnmatched agent prompts saved to {output_file}\n")


    def save_mismatch_files_csv(self, transcript_dict, mismatch_files, output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            column_headers = ["CallID", "Speaker", "StartTime", "EndTime", "Transcript", "AgentPrompt", "IQMedian"]
            writer = csv.DictWriter(csvfile, fieldnames=column_headers)

            writer.writeheader()

            for filecode, entries in transcript_dict.items():
                if filecode in mismatch_files:
                    for entry in entries:
                        writer.writerow(entry)

        print(f"\nAligned segments and mismatches for relevant files saved to {output_file}\n")


    # ------ Core Processing Function ------

    def run_pipeline(self):
        """
        Runs pipeline to isolate system speech segments in the dyadic audio
        recordings
        """
        # Gather raw audio file paths
        wav_file_dict = get_filepaths(self.directory_dict, "audio")

        # Debug mode: limit number of files for faster testing
        if self.debug:
            debug_limit = 5
            wav_file_dict = dict(list(wav_file_dict.items())[:debug_limit])
            print(f"Debug mode: Limiting to first {debug_limit} audio files for processing.")

        # Stage 1: Transcribe mixed audio files to generate transcripts
        mixed_transcript_dir = self.directory_dict.get(
            "mixed_transcripts", ""
        )

        # Check if EVERY filecode in wav_file_dict has both .json and .txt outputs present
        missing_transcripts = any(
            not (mixed_transcript_dir / f"{filecode}.json").exists() or
            not (mixed_transcript_dir / f"{filecode}.txt").exists()
            for filecode in wav_file_dict.keys()
        )

        if self.force or missing_transcripts:
            print("Stage 1: Transcribing mixed audio files to generate transcripts.")
            self.transcribe_mixed_audio(wav_file_dict, output_dir=mixed_transcript_dir)
        else:
            print(
                "All required mixed transcripts already exist in directory."
                " Skipping transcription step."
            )

        # Stage 2: Align agent prompts with mixed transcripts

        aligned_transcript_file_path = self.output_files_dict.get(
            "aligned_agent_transcript", ""
        )
        unmatched_prompts_file_path = self.output_files_dict.get(
            "unmatched_prompts_json", ""
        )
        mismatched_transcripts_file_path = self.output_files_dict.get(
            "mismatched_transcript_csv", ""
        )

        if self.force or not aligned_transcript_file_path.exists():
            print("Stage 2: Aligning agent prompts with mixed transcripts.")

            # Load mixed transcript filepaths, agent prompts and user transcripts
            mixed_transcript_dict = get_filepaths(
                directory_dict=self.directory_dict,
                folder_to_process="mixed_transcripts"
            )
    
            agent_df = pd.read_csv(self.input_files_dict.get("agent_csv", ""))
            user_df = pd.read_csv(self.input_files_dict.get("user_csv", ""))

            agent_transcripts, unmatched_dict, mismatched_files_set = self.align_agent_with_mixed(
                mixed_transcript_dict=mixed_transcript_dict, 
                agent_df=agent_df, user_df=user_df
            )

            save_transcript_csv(
                transcript_dict=agent_transcripts, 
                output_file = aligned_transcript_file_path
            )
            
            self.save_unmatched_to_json(
                unmatched_dict=unmatched_dict, 
                output_file = unmatched_prompts_file_path
            )
            
            self.save_mismatch_files_csv(
                transcript_dict=agent_transcripts, 
                mismatch_files=mismatched_files_set, 
                output_file = mismatched_transcripts_file_path
            )
        else:
            print(
                "Aligned agent transcripts already exist in output directory."
                " Skipping alignment step."
            )
            return