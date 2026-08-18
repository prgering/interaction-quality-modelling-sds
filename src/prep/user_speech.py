import os
import csv
import sys
import soundfile as sf
import tempfile
import numpy as np
import torch
import whisperx
from tqdm import tqdm
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from collections import defaultdict
from scipy.signal import correlate
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_filepaths


class UserSpeechProcessor:
    def __init__(
        self, 
        config_dict = None, 
        directory_dict = None, 
        device = None,
        force = False,
        debug = False
    ):
        """Initialise UserSpeechProcessor"""
        config = config_dict or {}

        self.device = device
        self.force = force
        self.debug = debug

        self.directory_dict = directory_dict or {}
        
        self.vad_param_dict = config.get("vad_params", {})
        self.normalise_params_dict = config.get("normalise_params", {})
        self.asr_params_dict = config.get("asr_params", {})

    # --- Utility Methods ---

    @staticmethod
    def rms_normalise(
        audio, 
        speech_segments, 
        sample_rate, 
        target_rms = 0.07):
        """
        Normalise audio based on time-weighted RMS of speech segments.
        """

        rms_tracker = 0.0
        time_tracker = 0.0

        for start_sec, end_sec in speech_segments:
            start_idx = int(start_sec * sample_rate)
            end_idx = int(end_sec * sample_rate)
            segment = audio[start_idx:end_idx]
            
            if len(segment) == 0:
                continue

            segment_rms = np.sqrt(np.mean(segment ** 2)) + 1e-6
            segment_duration = end_idx - start_idx

            rms_tracker += segment_rms * segment_duration
            time_tracker += segment_duration
        
        if time_tracker == 0:
            return audio, False

        total_rms = rms_tracker / time_tracker
        scalar = target_rms / total_rms

        normalised_audio = np.clip(audio * scalar, -1.0, 1.0)

        return normalised_audio, True

    @staticmethod
    def add_white_noise(audio, noise_level=0.05):
        """Adds uniform white noise to the entire audio signal."""
        noise = np.random.normal(0, noise_level, size=audio.shape)
        noisy_audio = audio + noise
        return np.clip(noisy_audio, -1.0, 1.0)

    @staticmethod
    def calculate_audio_offset(input_audio, mixed_audio, sr1):
        """
        Calculate time offset between user amplified and dyadic audio
        using cross-correlation.
        """
        min_len = min(len(input_audio), len(mixed_audio))
        input_audio = input_audio[:min_len]
        mixed_audio = mixed_audio[:min_len]

        corr = correlate(mixed_audio, input_audio)
        lag = np.argmax(corr) - (len(input_audio) - 1)
        offset_seconds = lag / sr1

        print(f"Estimated offset: {offset_seconds:.3f} seconds")

        return offset_seconds
    
    # --- Core Processing Methods ---

    def perform_vad(self, wav_filepaths, model = None, stage2 = False):
        """
        Performs VAD on .wav files in the root_folder and saves RTTM files
        to appropriate VAD folder based on the stage of processing.
        """

        # VAD parameters set based on processing stage
        if stage2:
            vad_folder = self.directory_dict.get("accurate_vad", None)
            threshold = self.vad_param_dict.get("accurate_user_vad_threshold", 0.3)
        else:
            vad_folder = self.directory_dict.get("inaccurate_vad", None)
            threshold = self.vad_param_dict.get("inaccurate_user_vad_threshold", 0.8)

        # Early exit if VAD folder is not specified
        if not vad_folder:
            raise ValueError(
                f"VAD directory not specified for stage "
                f"{'2' if stage2 else '1'} in directory_dict."
                )

        min_speech_duration_ms = self.vad_param_dict.get("min_speech_duration_ms", 50)
        sr = self.vad_param_dict.get("sampling_rate", 8000)

        # Loop through files and perform VAD
        for filecode, value in tqdm(wav_filepaths.items(), desc="Performing VAD on full audio"):
            try:
                if isinstance(value, dict):
                    audio_path = value.get("user") if stage2 else value["user"]
                else:
                    audio_path = value

                if not audio_path:
                    raise ValueError(f"Audio path not found for {filecode}.")
            except (KeyError, TypeError, AttributeError) as e:
                print(f"Error accessing audio path for {filecode}: {e}")
                continue

            try:
                with torch.no_grad():
                    # Read audio file and convert to torch tensor
                    wav = read_audio(audio_path, sampling_rate=sr).to(
                        self.device
                    )

                    # Perform VAD using Silero VAD
                    speech_segments = get_speech_timestamps(
                        wav,
                        model,
                        min_speech_duration_ms = min_speech_duration_ms,
                        threshold = threshold,
                        sampling_rate = sr,
                        return_seconds = True
                    )
                    
            except Exception as e:
                print(f"Unable to Read Audiofile {filecode}: {e}")
                continue

            # Write VAD output to RTTM file
            output_rttm_path = vad_folder / f"{filecode}.rttm"

            with open(output_rttm_path, "w") as f:
                for speech_times in (speech_segments):
                    start_time = speech_times["start"]
                    end_time = speech_times["end"]
                    f.write(
                        f"SPEAKER {filecode} 1 {start_time:.5f}"
                        f" {end_time - start_time:.5f} <NA> <NA> 1.0"
                        " <NA>\n"
                    )

    def normalise_and_add_noise(self, wav_filepaths, rttm_filepaths):
        """
        Normalise audio based on RMS of speech segments and add Gaussian
        white noise to mask system speech in the background.
        """
        # Normalisation parameters
        target_rms = self.normalise_params_dict.get("target_rms", 0.07)
        noise_level = self.normalise_params_dict.get("white_noise_level", 0.05)

        # Get normalised audio output folder from directory_dict
        normalised_audio_folder = self.directory_dict.get("normalised_audio")
        if not normalised_audio_folder:
            raise ValueError(
                "Normalised audio directory not specified in directory_dict."
            )

        # Normalise map lookup kets to strings if they are Path objects
        wav_path_map = {
            str(k.stem if isinstance(k, Path) else k): v for k, v in wav_filepaths.items()
        }
        
        # Loop through RTTM files
        for rttm_filepath in tqdm(rttm_filepaths, desc="Normalizing Speech Segments"):
            rttm_filecode = rttm_filepath.stem

            # Check if corresponding audio file exists and can be read
            if rttm_filecode not in wav_path_map:
                print(f"Audio file not found for {rttm_filecode}. Skipping.")
                continue

            # Read audio file and handle potential errors
            try:
                audio_val = wav_path_map[rttm_filecode]
                audio_filepath = str(audio_val.get("user") if isinstance(audio_val, dict) else audio_val)
                audio_data, sr = sf.read(audio_filepath)
            except Exception as e:
                print(f"Error reading audio file {audio_filepath}: {e}")
                continue

            # Extract speech segments from RTTM file
            speech_segments = []
            try:
                with open(rttm_filepath, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5 and parts[0] == "SPEAKER":
                            start_time = float(parts[3])
                            duration = float(parts[4])
                            end_time = start_time + duration
                            speech_segments.append((start_time, end_time))
            except Exception as e:
                print(f"Error reading RTTM file {rttm_filepath}: {e}")
                continue

            # Perform RMS normalisation based on speech segments
            normalised_audio, normalised_outcome = self.rms_normalise(
                audio_data, 
                speech_segments, 
                sr, 
                target_rms=target_rms
            )

            # Add white noise and save normalised audio
            if normalised_outcome:
                final_audio = self.add_white_noise(
                    normalised_audio, noise_level=noise_level
                )
                output_filename = (
                    normalised_audio_folder / f"{rttm_filecode}.wav"
                )
                sf.write(output_filename, final_audio, sr)

            else:
                print(f"Skipped File: {rttm_filecode} — no speech segments.")


    def transcribe_speech_segments(self, wav_filepaths, vad_files):
        """Transcribes speech segments identified by VAD using WhisperX."""
        device_str = self.device.type if isinstance(self.device, torch.device) else str(self.device)

        # Resolve compute type and model type
        user_compute_type = self.asr_params_dict.get("compute_type", "float16")
        compute_type = user_compute_type if device_str == "cuda" else "float32"
        
        model_type = "tiny" if self.debug else self.asr_params_dict.get("model_type", "large-v3")
        batch_size = self.asr_params_dict.get("batch_size", 16)

        # Load WhisperX models
        model = whisperx.load_model(model_type, device=device_str, compute_type=compute_type)
        model_a, metadata = whisperx.load_align_model(language_code="en", device=device_str)

        all_transcripts = defaultdict(list)

        # Normalise map lookup kets to strings if they are Path objects
        wav_path_map = {
            str(k.stem if isinstance(k, Path) else k): v for k, v in wav_filepaths.items()
        }
        
        # Loop through VAD files and check for corresponding audio files
        for vad_filepath in tqdm(vad_files, desc="Transcribing Speech Segments"):
            vad_filepath = Path(vad_filepath)
            vad_filecode = vad_filepath.stem 
            if vad_filecode not in wav_path_map:
                print(f"Warning: Audio file not found for {vad_filecode}. Skipping.")
                continue
            # Read user amplified and dyadic audio files
            # Calculate offset between user and dyadic audio
            try:
                audio_val = wav_path_map[vad_filecode]
                user_audio_filepath = str(audio_val.get("user"))
                dyad_audio_filepath = str(audio_val.get("dyad"))
                user_audio_data, sr1 = sf.read(user_audio_filepath)
                dyad_audio_data, sr2 = sf.read(dyad_audio_filepath)
                offset_time = self.calculate_audio_offset(user_audio_data, dyad_audio_data, sr1)

            except Exception as e:
                print(f"Error reading audio file {wav_filepaths[vad_filecode]}: {e}")
                continue
                
            # Extract speech segments from VAD RTTM file
            speech_segments = []
            
            with open(vad_filepath, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0] == "SPEAKER":
                        start_time = float(parts[3])
                        duration = float(parts[4])
                        end_time = start_time + duration
                        speech_segments.append((start_time, end_time))

            # Process audio segments
            for i, (start_sec, end_sec) in enumerate(speech_segments):
                start_idx = int(start_sec * sr1)
                end_idx = int(end_sec * sr1)
                segment = user_audio_data[start_idx:end_idx].astype(np.float32)
        
                if len(segment) == 0:
                    continue

                segment_start_shift = start_sec + offset_time

                # ASR Transcription and alignment on speech segment
                try:
                    result = model.transcribe(
                        segment, language = "en", batch_size = batch_size,
                    )

                    result = whisperx.align(
                        result["segments"], model_a, metadata, 
                        segment, device_str, 
                        return_char_alignments=False
                    )

                    # Adjust timestamps and save metadata for each segment
                    for entry in result['segments']:
                        words_data = [
                            {
                                "word": word.get("word", ""),
                                "start": word.get("start", 0.0) + segment_start_shift,
                                "end": word.get("end", 0.0) + segment_start_shift,
                            }
                            for word in entry.get("words", [])
                        ]

                        aligned_entry = {
                            "start": entry["start"] + segment_start_shift,
                            "end": entry["end"] + segment_start_shift,
                            "text": entry["text"],
                            "words": words_data,
                            "Speaker": "user"
                        }

                        all_transcripts[vad_filecode].append(aligned_entry)

                except Exception as e:
                    print(f"Error transcribing segment {i} in {vad_filecode}: {e}")
        
        return all_transcripts

    def generate_transcript_csv(self, transcript_dict, output_file):
        """Generates CSV file for user transcript"""
        if not output_file:
            print(
                "No output file specified for transcript CSV."
                "Skipping CSV generation.")
            return

        with open(
            output_file, 'w', newline='', encoding='utf-8'
        ) as csvfile:
            column_headers = [
                "CallID", "Speaker", "StartTime", "EndTime", "Transcript"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=column_headers)
                    
            writer.writeheader()

            for filecode, entries in transcript_dict.items():
                for entry in entries:
                    entry["CallID"] = filecode
                    entry["StartTime"] = f"{entry['start']:.2f}"
                    entry["EndTime"] = f"{entry['end']:.2f}"
                    entry["Transcript"] = entry["text"]
                    del entry["text"]
                    del entry["start"]
                    del entry["end"]
                    del entry["words"]

                    writer.writerow(entry)

        print(f"\nTranscript saved to {output_file}\n")         
    
    def run_pipeline(self):
        """Runs the full speech processing pipeline"""

        # Gather raw audio file paths
        wav_file_dict = get_filepaths(
            self.directory_dict, 
            folder_to_process = "audio"
        )

        if self.debug:
            debug_limit = 5
            wav_file_dict = dict(list(wav_file_dict.items())[:debug_limit])
            print(f"Debug mode enabled: Processing only {debug_limit} files.")

        # Stage 1: Perform VAD on raw audio files
        inaccurate_vad_dir = self.directory_dict.get("inaccurate_vad")
        has_inaccurate_vad = inaccurate_vad_dir.exists() and any(inaccurate_vad_dir.iterdir())

        if self.force or not has_inaccurate_vad:
            print("Stage 1: Performing VAD on raw audio files...")
            vad_model = load_silero_vad().to(self.device)
            self.perform_vad(wav_file_dict, vad_model)
        else:
            print("Stage 1: Skipping VAD (output files already exist).")

        inaccurate_vad_files_list = get_filepaths(
            self.directory_dict, 
            folder_to_process = "inaccurate_vad"
        )

        # Stage 2: Normalise audio and add white noise
        norm_audio_dir = self.directory_dict.get("normalised_audio")
        has_normalised_audio = norm_audio_dir.exists() and any(norm_audio_dir.iterdir())

        if self.force or not has_normalised_audio:
            print("Stage 2: Normalising audio and adding white noise...")
            self.normalise_and_add_noise(
                wav_file_dict, 
                inaccurate_vad_files_list
            )
        else:
            print("Stage 2: Skipping normalisation (output files already exist).")

        normalised_file_dict = get_filepaths(
                self.directory_dict, 
                folder_to_process = "normalised_audio"
            )
        
        # Stage 3: Perform VAD on normalised audio files
        accurate_vad_dir = self.directory_dict.get("accurate_vad")
        has_accurate_vad = accurate_vad_dir.exists() and any(accurate_vad_dir.iterdir())

        if self.force or not has_accurate_vad:
            print("Stage 3: Performing VAD on normalised audio files...")
            if 'vad_model' not in locals():
                vad_model = load_silero_vad().to(self.device)

            self.perform_vad(normalised_file_dict, vad_model, stage2= True)
        else:
            print("Stage 3: Skipping VAD (output files already exist).")

        accurate_vad_files_list = get_filepaths(
            self.directory_dict, 
            folder_to_process = "accurate_vad"
        )

        # Stage 4: Transcribe speech segments using WhisperX
        output_dir = self.directory_dict.get("output")
        if self.debug:
            output_file = output_dir / "debug_user_asr_output.csv"
        else:
            output_file = output_dir / "user_asr_output.csv"

        has_transcript_csv = output_file.exists() and output_file.stat().st_size > 0

        if self.force or not has_transcript_csv:
            print("Stage 4: Transcribing speech segments using WhisperX...")
            transcripts_dict = self.transcribe_speech_segments(
                wav_filepaths= wav_file_dict, 
                vad_files = accurate_vad_files_list
            )
            self.generate_transcript_csv(transcripts_dict, output_file)
        else:
            print("Stage 4: Skipping transcription (output CSV already exists).")





