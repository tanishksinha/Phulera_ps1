import librosa
import numpy as np

DEFAULT_FS = 22050
DEFAULT_WINDOW_SIZE = 4096
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_AMP_MIN = 10

audio_data, sr = librosa.load("data/raw/data/blues/blues.00000.wav", sr=DEFAULT_FS)
stft = np.abs(librosa.stft(audio_data, n_fft=DEFAULT_WINDOW_SIZE, 
                            hop_length=int(DEFAULT_WINDOW_SIZE * (1 - DEFAULT_OVERLAP_RATIO))))
spectrogram = librosa.amplitude_to_db(stft, ref=np.max)

print("Max val:", np.max(spectrogram))
print("Min val:", np.min(spectrogram))
