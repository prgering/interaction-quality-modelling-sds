import csv
import re
import soundfile as sf
from collections import defaultdict
from pathlib import Path

def list_files(directory, pattern="*"):
    """Recursively list and sort all matching files in a directory."""
    path = Path(directory)
    if not path.is_dir():
        raise NotADirectoryError(f"Directory not found: {path}")
    return sorted(path.rglob(pattern))


def is_valid_audio(filepath, filecode):
    """Helper to validate audio files."""
    path = Path(filepath)
    if not path.exists():
        print(f"ERROR: Audio file DOES NOT EXIST at: {path}. Skipping {filecode}.")
        return False, "Audio file does not exist."
    if path.stat().st_size == 0:
        print(f"ERROR: Audio file {path} is EMPTY. Skipping {filecode}.")
        return False, "Audio file is empty."
    try:
        with sf.SoundFile(path, 'r') as f:
            if f.frames == 0:
                print(f"ERROR: Audio file {path} has 0 frames. Skipping {filecode}.")
                return False, "Audio file has 0 frames."
    except Exception as e:
        print(f"ERROR: Could not read {path}: {e}. Skipping {filecode}.")
        return False, "Could not read audio file."
    return True, None

def _get_audio_files(directory, folder_type):
    """Processes audio files into a dyad/user dictionary structure."""
    prefix = "LetsGoPublic" if folder_type == "audio" else ""
    files = list_files(directory, pattern = f"{prefix}*.wav")

    wav_filepaths = defaultdict(dict)
    for filepath in files:
        filecode = (re.sub(r"[^0-9]", "", filepath.name))[-11:]

        is_valid, reason = is_valid_audio(filepath, filecode)
        if not is_valid:
            print(f"Skipping {filecode}: {reason}")
            continue

        key = "dyad" if "output" in filepath.name else "user"
        wav_filepaths[filecode][key] = str(filepath)
    return wav_filepaths

def _get_transcript_files(directory):
    """Returns JSON transcript files in a dictionary keyed by filecode."""
    files = list_files(directory, pattern="*.json")
    return {p.stem: p for p in files}


def get_filepaths(directory_dict, folder_to_process = None):
    """Returns filepaths for the specified folder_to_process."""
    target_dir = directory_dict.get(folder_to_process, None)
    
    if not target_dir:
        print(f"Warning: No directory found for folder_to_process '{folder_to_process}'.")
        return None

    if folder_to_process in ("inaccurate_vad", "accurate_vad"):
        return list_files(target_dir, pattern="*.rttm")

    if folder_to_process == "cv_results":
        return list_files(target_dir, pattern="*.csv")

    handlers = {
        "audio": lambda d: _get_audio_files(d, folder_type="audio"),
        "normalised_audio": lambda d: _get_audio_files(d, folder_type="normalised_audio"),
        "transcripts": _get_transcript_files,
        "mixed_transcripts": _get_transcript_files
    }

    handler = handlers.get(folder_to_process, None)
    if handler:
        return handler(target_dir)

    print(f"Warning: Unknown folder_to_process '{folder_to_process}'.")
    return None

def save_transcript_csv(transcript_dict, output_file, is_user_transcript = False):
    """Saves user or system transcript dictionaries to a formatted CSV file."""
    if not output_file:
        print("No output file specified for transcript CSV. Skipping CSV generation.")
        return

    base_headers = ["CallID", "Speaker", "StartTime", "EndTime", "Transcript"]
    column_headers = base_headers if is_user_transcript else base_headers + ["AgentPrompt", "IQMedian"]

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=column_headers, extrasaction='ignore')
        writer.writeheader()

        for filecode, entries in transcript_dict.items():
            for entry in entries:
                if is_user_transcript:
                    row = {
                        "CallID": filecode,
                        "Speaker": entry.get("Speaker", "user"),
                        "StartTime": f"{entry['start']:.2f}" if isinstance(entry.get('start'), (int, float)) else entry.get('start'),
                        "EndTime": f"{entry['end']:.2f}" if isinstance(entry.get('end'), (int, float)) else entry.get('end'),
                        "Transcript": entry.get("text", entry.get("Transcript", ""))
                    }
                else:
                    row = dict(entry)
                    row["CallID"] = filecode

                writer.writerow(row)

    print(f"Transcript saved to {output_file}")