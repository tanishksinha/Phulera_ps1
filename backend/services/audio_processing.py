import librosa
import noisereduce as nr
import numpy as np

def load_and_preprocess_audio(file_path: str, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """
    Loads an audio file, resamples it to target_sr, and applies noise reduction.
    """
    try:
        # Load audio (mono by default in librosa)
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        # Apply noise reduction
        # Using a portion of the audio to profile the noise (e.g., first 0.5 seconds if available)
        if len(y) > int(sr * 0.5):
            noise_profile = y[:int(sr * 0.5)]
            reduced_noise = nr.reduce_noise(y=y, sr=sr, y_noise=noise_profile)
        else:
            reduced_noise = nr.reduce_noise(y=y, sr=sr)
            
        return reduced_noise, sr
    except Exception as e:
        print(f"Error processing audio {file_path}: {e}")
        raise e

# formatted
