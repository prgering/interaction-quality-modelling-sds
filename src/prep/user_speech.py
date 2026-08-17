import os
import csv
import sys
import soundfile as sf
import tempfile
import numpy as np
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
        output_filepath = None,
        device = None
    ):
        """Initialise UserSpeechProcessor"""
        config = config_dict or {}

        self.device = device

        self.directory_dict = directory_dict or {}
        self.output_file = output_filepath or None
        
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
            threshold = self.vad_param_dict.get(
                "accurate_user_vad_threshold", 0.3
            )
        else:
            vad_folder = self.directory_dict.get("inaccurate_vad", None)
            threshold = self.vad_param_dict.get(
                "inaccurate_user_vad_threshold", 0.8
            )

        min_speech_duration_ms = self.vad_param_dict.get(
            "min_speech_duration_ms", 50
        )
        sr = self.vad_param_dict.get("sampling_rate", 8000)

        # Loop through files and perform VAD
        filecodes = wav_filepaths.keys()
        
        with tqdm(total=len(filecodes), desc="Performing VAD on full audio") as pbar:
            for filecode in filecodes:
                # Extract audio path from dict based on processing stage
                try:
                    if stage2:
                        audio_path = str(wav_filepaths[filecode].get("user", wav_filepaths[filecode]))
                    else:
                        audio_path = str(wav_filepaths[filecode]["user"])
                except (KeyError, TypeError) as e:
                    print(
                        f"Skipping {filecode}: Audio path not properly"
                        f"structured. Error: {e}"
                    )
                    pbar.update(1)
                    continue

                try:
                    wav = read_audio(audio_path, sampling_rate=sr).to(
                        self.device
                    )
                except Exception as e:
                    print(f"Unable to Read Audiofile {filecode}: {e}")
                    pbar.update(1)
                    continue

                # Perform VAD using Silero VAD model
                speech_segments = get_speech_timestamps(
                    wav,
                    model,
                    min_speech_duration_ms = min_speech_duration_ms,
                    threshold = threshold,
                    sampling_rate = sr,
                    return_seconds = True
                )

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

                pbar.update(1)

    def normalise_and_add_noise(self, wav_filepaths, rttm_filepaths):
        """
        Normalise audio based on RMS of speech segments and add Gaussian
        white noise to mask system speech in the background.
        """
        # Normalisation parameters
        target_rms = self.normalise_params_dict.get("target_rms", 0.07)
        noise_level = self.normalise_params_dict.get("white_noise_level", 0.05)

        # Get normalised audio output folder from directory_dict
        normalised_audio_folder = self.directory_dict.get(
            "normalised_audio", None
        )
        if normalised_audio_folder is None:
            print("Warning: 'normalised_audio' not specified in directory_dict.")
            return
        
        # Loop through RTTM files
        with tqdm(
            total=len(rttm_filepaths), desc="Normalizing Speech Segments"
        ) as pbar:
            for rttm_filepath in rttm_filepaths:
                rttm_filecode = rttm_filepath.stem
                # Check if corresponding audio file exists and can be read
                if rttm_filecode not in wav_filepaths.keys():
                    print(
                        f"Audio file not found for {rttm_filecode}. Skipping."
                    )
                    pbar.update(1)
                    continue
                try:
                    audio_filepath = str(wav_filepaths[rttm_filecode]["user"])
                    audio_data, sr = sf.read(audio_filepath)
                except Exception as e:
                    print(f"Error reading audio file {audio_filepath}: {e}")
                    pbar.update(1)
                    continue

                # Extract speech segments from RTTM file
                speech_segments = []
                
                with open(rttm_filepath, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts[0] == "SPEAKER" and parts[1] == rttm_filecode:
                            start_time = float(parts[3])
                            duration = float(parts[4])
                            end_time = start_time + duration
                            speech_segments.append((start_time, end_time))

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

                pbar.update(1)

    def transcribe_speech_segments(self, wav_filepaths, vad_files):
        """Transcribes speech segments identified by VAD using WhisperX."""
        # ASR parameters
        batch_size = self.asr_params_dict.get("batch_size", 16)
        compute_type = self.asr_params_dict.get("compute_type", "float16")
        model_type = self.asr_params_dict.get("model_type", "large-v3")

        # Load WhisperX ASR model and alignment model
        model = whisperx.load_model(
            model_type, 
            device = self.device, 
            compute_type = compute_type
        )

        model_a, metadata = whisperx.load_align_model(
            language_code="en", device=self.device
        )

        all_transcripts = defaultdict(list)
        
        # Loop through VAD files and check for corresponding audio files
        with tqdm(total=len(vad_files), desc=f"Performing ASR") as pbar:
            for vad_filepath in vad_files:
                vad_filecode = vad_filepath.stem 
                if vad_filecode not in wav_filepaths.keys():
                    print(f"Warning: Audio file not found for {vad_filecode}. Skipping.")
                    pbar.update(1)
                    continue
                # Read user amplified and dyadic audio files
                # Calculate offset between user and dyadic audio
                try:
                    user_audio_data, sr1 = sf.read(str(wav_filepaths[vad_filecode]["user"]))
                    dyad_audio_data, sr2 = sf.read(str(wav_filepaths[vad_filecode]["dyad"]))
                    offset_time = self.calculate_audio_offset(user_audio_data, dyad_audio_data, sr1)

                except Exception as e:
                    print(f"Error reading audio file {wav_filepaths[vad_filecode]}: {e}")
                    pbar.update(1)
                    continue
                
                # Extract speech segments from VAD RTTM file
                speech_segments = []
                
                with open(vad_filepath, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts[0] == "SPEAKER" and parts[1] == vad_filecode:
                            start_time = float(parts[3])
                            duration = float(parts[4])
                            end_time = start_time + duration
                            speech_segments.append((start_time, end_time))
                # Loop through speech segments
                for i, (start_sec, end_sec) in enumerate(speech_segments):
                    start_idx = int(start_sec * sr1)
                    end_idx = int(end_sec * sr1)
                    segment = user_audio_data[start_idx:end_idx]
            
                    if len(segment) == 0:
                        continue

                    segment_start_shift = start_sec + offset_time

                    # Save segment to temporary file for ASR processing
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav_file:
                        sf.write(temp_wav_file.name, segment, sr1)

                        # ASR Transcription and alignment on speech segment
                        try:
                            result = model.transcribe(
                                temp_wav_file.name, language = "en"
                            )

                            result = whisperx.align(
                                result["segments"], model_a, metadata, 
                                temp_wav_file.name, self.device, 
                                return_char_alignments=False
                            )

                            # Adjust timestamps and save metadata for each segment
                            for entry in result['segments']:
                                words_data = []
                                for word in entry.get("words", []):
                                    words_data.append({
                                        "word": word.get("word", ""),
                                        "start": (
                                            word.get("start", 0.0) 
                                            + segment_start_shift
                                        ),
                                        "end": (
                                            word.get("end", 0.0) 
                                            + segment_start_shift
                                        ),
                                    })
                                entry["words"] = words_data

                                aligned_entry = {
                                    "start": (
                                        entry["start"] + start_sec + offset_time
                                    ),
                                    "end": (
                                        entry["end"] + start_sec + offset_time
                                    ),
                                    "text": entry["text"],
                                    "words": entry["words"]
                                }

                                aligned_entry["Speaker"] = "user"

                                all_transcripts[vad_filecode].append(aligned_entry)
                        except Exception as e:
                            print(f"Error transcribing segment {i} in {vad_filecode}: {e}")
                
                pbar.update(1)

        return all_transcripts

    def generate_transcript_csv(self, transcript_dict):
        """Generates CSV file for user transcript"""
        if not self.output_file:
            print(
                "No output file specified for transcript CSV."
                "Skipping CSV generation.")
            return

        with open(
            self.output_file, 'w', newline='', encoding='utf-8'
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

        print(f"\nTranscript saved to {self.output_file}\n")         
    
    def run_pipeline(self):
        """Runs the full speech processing pipeline"""
        
        wav_file_dict = get_filepaths(
            self.directory_dict, 
            folder_to_process = "audio"
        )

        vad_model = load_silero_vad().to(self.device)
        
        self.perform_vad(wav_file_dict, vad_model)

        inaccurate_vad_files_list = get_filepaths(
            self.directory_dict, 
            folder_to_process = "inaccurate_vad"
        )

        self.normalise_and_add_noise(
            wav_file_dict, 
            inaccurate_vad_files_list
        )

        normalised_file_dict = get_filepaths(
            self.directory_dict, 
            folder_to_process = "normalised_audio"
        )

        self.perform_vad(normalised_file_dict, stage2= True)

        accurate_vad_files_list = get_filepaths(
            self.directory_dict, 
            folder_to_process = "accurate_vad"
        )

        transcripts_dict = self.transcribe_speech_segments(
            wav_filepaths= wav_file_dict, 
            vad_files = accurate_vad_files_list
        )

        self.generate_transcript_csv(transcripts_dict)





