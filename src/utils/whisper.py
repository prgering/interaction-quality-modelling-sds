import torch
import whisperx

def get_asr_config(device, asr_params_dict, debug):
    """Standardises model parameters, compute types, and string representation of device."""
    # Determine device string representation
    device_str = device.type if isinstance(device, torch.device) else str(device)

    # Resolve compute type and model type
    user_compute_type = asr_params_dict.get("compute_type", "float16")
    compute_type = user_compute_type if device_str == "cuda" else "float32"

    model_type = "tiny" if debug else asr_params_dict.get("model_type", "large-v3")
    batch_size = asr_params_dict.get("batch_size", 16)
    
    return device_str, compute_type, model_type, batch_size

def transcribe_and_align(audio_data, model, model_align, metadata, batch_size, device_str):
    """Unified helper for WhisperX transcription and alignment pass."""
    result = model.transcribe(audio_data, language="en", batch_size=batch_size)
    aligned_result = whisperx.align(
        result["segments"], 
        model_align, 
        metadata, 
        audio_data, 
        device_str, 
        return_char_alignments=False
    )
    return aligned_result