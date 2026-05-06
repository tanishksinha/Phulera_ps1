import librosa
import numpy as np
import hashlib
import pickle
import os
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure, iterate_structure

# Fingerprinting parameters
DEFAULT_FS = 22050
DEFAULT_WINDOW_SIZE = 4096
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_FAN_VALUE = 15
DEFAULT_AMP_MIN = 10
PEAK_NEIGHBORHOOD_SIZE = 20
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200

def get_2D_peaks(arr2D, plot=False, amp_min=DEFAULT_AMP_MIN):
    """
    Finds peaks in a 2D array (spectrogram) using a local maximum filter.
    """
    # Define an 8-connected neighborhood
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # Find local maxima using the filter
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
    
    # Apply a background threshold
    background = (arr2D == 0)
    eroded_background = np.logical_not(background)
    
    # Apply minimum amplitude threshold
    amplitudes = arr2D > amp_min

    # Find the peaks
    peaks = local_max ^ background
    peaks = peaks & eroded_background & amplitudes

    # Get the row and column indices of the peaks
    frequencies, times = np.where(peaks)
    
    # Return as a list of tuples (time, frequency)
    return list(zip(times, frequencies))

def generate_hashes(peaks, fan_value=DEFAULT_FAN_VALUE):
    """
    Generates hashes from a list of (time, frequency) peaks.
    """
    # Sort peaks by time
    peaks.sort(key=lambda x: x[0])
    hashes = []

    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):
                t1 = peaks[i][0]
                t2 = peaks[i + j][0]
                t_delta = t2 - t1

                if t_delta >= MIN_HASH_TIME_DELTA and t_delta <= MAX_HASH_TIME_DELTA:
                    f1 = peaks[i][1]
                    f2 = peaks[i + j][1]

                    # Hash format: (f1|f2|t_delta)
                    # We hash this string to create a compact fingerprint
                    hash_str = f"{f1}|{f2}|{t_delta}"
                    hash_val = hashlib.sha1(hash_str.encode('utf-8')).hexdigest()[:20]

                    # Store the hash along with its absolute time offset (t1)
                    hashes.append((hash_val, t1))
    return hashes

def fingerprint_audio(audio_data, fs=DEFAULT_FS):
    """
    Takes an audio array and returns a list of (hash, time_offset) pairs.
    """
    # 1. Generate Spectrogram
    # Using librosa's STFT, then taking the magnitude and converting to dB
    stft = np.abs(librosa.stft(audio_data, n_fft=DEFAULT_WINDOW_SIZE, 
                               hop_length=int(DEFAULT_WINDOW_SIZE * (1 - DEFAULT_OVERLAP_RATIO))))
    spectrogram = librosa.amplitude_to_db(stft, ref=np.max)

    # 2. Extract Peaks
    peaks = get_2D_peaks(spectrogram, amp_min=DEFAULT_AMP_MIN)

    # 3. Generate Hashes from Peaks
    hashes = generate_hashes(peaks, fan_value=DEFAULT_FAN_VALUE)

    return hashes

class FingerprintDB:
    """
    A simple in-memory database for our fingerprint hashes.
    In a real app, this would be backed by Redis or SQLite.
    """
    def __init__(self):
        # Maps hash_val -> list of (song_id, time_offset)
        self.hash_table = {}
        # Maps song_id -> song metadata
        self.songs = {}

    def insert_song(self, song_id, metadata, hashes):
        self.songs[song_id] = metadata
        for hash_val, offset in hashes:
            if hash_val not in self.hash_table:
                self.hash_table[hash_val] = []
            self.hash_table[hash_val].append((song_id, offset))

    def match_hashes(self, query_hashes):
        """
        Finds the song that has the most matching hashes with consistent time offsets.
        """
        # Dictionary to count matches: song_id -> match_count
        matches_per_song = {}
        
        # Dictionary to track time offset consistency: song_id -> list of (target_offset - query_offset)
        offset_deltas = {}

        for hash_val, query_offset in query_hashes:
            if hash_val in self.hash_table:
                for target_song_id, target_offset in self.hash_table[hash_val]:
                    delta = target_offset - query_offset
                    
                    if target_song_id not in offset_deltas:
                        offset_deltas[target_song_id] = {}
                    
                    if delta not in offset_deltas[target_song_id]:
                        offset_deltas[target_song_id][delta] = 0
                    
                    offset_deltas[target_song_id][delta] += 1
                    
        # Find the song with the highest number of consistent time offsets
        best_match = None
        max_matches = 0
        
        for song_id, deltas in offset_deltas.items():
            # Find the delta with the most matches for this song
            if not deltas:
                continue
            best_delta_for_song = max(deltas, key=deltas.get)
            match_count = deltas[best_delta_for_song]
            
            if match_count > max_matches:
                max_matches = match_count
                best_match = song_id
                
        return best_match, max_matches

    def save_to_disk(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'hash_table': self.hash_table, 'songs': self.songs}, f)

    def load_from_disk(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.hash_table = data.get('hash_table', {})
                self.songs = data.get('songs', {})

# Global instance for our demo
db = FingerprintDB()

# formatted
