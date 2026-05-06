import numpy as np
import warnings

# Suppress warnings from panns_inference
warnings.filterwarnings("ignore")

try:
    from panns_inference import AudioTagging
    # Initialize the model globally so it's only loaded once.
    # Set device to 'cuda' if GPU is available, else 'cpu'.
    # We'll use CPU by default for broader compatibility unless configured otherwise.
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

neural_db = NeuralDB()
