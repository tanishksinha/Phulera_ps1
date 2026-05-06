import numpy as np
import warnings
import pickle
import os
import urllib.request
from pathlib import Path

# Suppress warnings from panns_inference
warnings.filterwarnings("ignore")

def ensure_panns_files():
    # panns_inference relies on wget which breaks on Windows. We must download manually.
    home_dir = str(Path.home())
    panns_dir = os.path.join(home_dir, "panns_data")
    os.makedirs(panns_dir, exist_ok=True)
    
    csv_path = os.path.join(panns_dir, "class_labels_indices.csv")
    model_path = os.path.join(panns_dir, "Cnn14_mAP=0.431.pth")
    
    if not os.path.exists(csv_path):
        print("Downloading PANNs CSV...")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv", csv_path)
    
    if not os.path.exists(model_path):
        print("Downloading PANNs Model (200MB+)... This might take a minute.")
        urllib.request.urlretrieve("https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth", model_path)

try:
    ensure_panns_files()
    from panns_inference import AudioTagging
    at = AudioTagging(checkpoint_path=None, device='cpu')
    MODEL_LOADED = True
except ImportError:
    print("Warning: panns_inference not installed. Neural matching disabled.")
    at = None
    MODEL_LOADED = False
except Exception as e:
    print(f"Error loading panns_inference model: {e}")
    at = None
    MODEL_LOADED = False
    print("Warning: panns_inference not installed. Neural matching disabled.")
    at = None
    MODEL_LOADED = False
except Exception as e:
    print(f"Error loading panns_inference model: {e}")
    at = None
    MODEL_LOADED = False

def get_audio_embedding(audio_data: np.ndarray, sr: int) -> np.ndarray:
    """
    Returns the neural embedding for an audio clip using PANNs.
    Audio data should be a 1D numpy array, ideally sampled at 32000 Hz for PANNs.
    """
    if not MODEL_LOADED:
        return np.zeros(2048) # Return a dummy embedding if model failed to load
        
    # PANNs expects audio shape to be (batch_size, samples)
    audio_batch = audio_data[None, :] 
    
    # Get embedding (the second return value is the embedding)
    _, embedding = at.inference(audio_batch)
    return embedding[0]

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Computes cosine similarity between two embeddings.
    """
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(emb1, emb2) / (norm1 * norm2)

class NeuralDB:
    def __init__(self):
        self.embeddings = {} # song_id -> embedding
        
    def insert_embedding(self, song_id, embedding):
        self.embeddings[song_id] = embedding
        
    def find_best_match(self, query_embedding, threshold=0.7):
        """
        Returns (best_song_id, confidence) using cosine similarity.
        """
        best_match = None
        highest_sim = 0.0
        
        for song_id, emb in self.embeddings.items():
            sim = cosine_similarity(query_embedding, emb)
            if sim > highest_sim:
                highest_sim = sim
                best_match = song_id
                
        if highest_sim >= threshold:
            return best_match, highest_sim
        return None, highest_sim

    def save_to_disk(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'embeddings': self.embeddings}, f)

    def load_from_disk(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data.get('embeddings', {})

neural_db = NeuralDB()

# formatted
