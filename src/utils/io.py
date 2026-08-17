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
        return False
    if path.stat().st_size == 0:
        print(f"ERROR: Audio file {path} is EMPTY. Skipping {filecode}.")
        return False
    try:
        with sf.SoundFile(path, 'r') as f:
            if f.frames == 0:
                print(f"ERROR: Audio file {path} has 0 frames. Skipping {filecode}.")
                return False
    except Exception as e:
        print(f"ERROR: Could not read {path}: {e}. Skipping {filecode}.")
        return False
    return True

def _get_audio_files(directory, folder_type):
    """Processes audio files into a dyad/user dictionary structure."""
    prefix = "LetsGoPublic" if folder_type == "audio" else ""
    files = list_files(directory, pattern = f"{prefix}*.wav")

    wav_filepaths = defaultdict(dict)
    for filepath in files:
        filecode = (re.sub(r"[^0-9]", "", filepath.name))[-11:]

        if not is_valid_audio(filepath, filecode):
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

    handlers = {
        "audio": lambda d: _get_audio_files(d, folder_type="audio"),
        "normalised_audio": lambda d: _get_audio_files(d, folder_type="normalised_audio"),
        "transcripts": _get_transcript_files,
    }

    handler = handlers.get(folder_to_process, None)
    if handler:
        return handler(target_dir)

    print(f"Warning: Unknown folder_to_process '{folder_to_process}'.")
    return None
