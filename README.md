# Audio Identification & Source Detection System

## Team Information
- **Team Name**: Phulera
- **Year**: 2nd year
- **All-Female Team**: No

## Architecture Overview
###Audio Identification & Source Detection System

An enterprise-grade, dual-engine audio identification system designed to scale to thousands of songs, handle heavy noise/distortion, and return real-time matches with high accuracy.
#### Describe your approach here. Keep it short and clear.

    This system uses a decoupled, microservice-style architecture. The frontend is built in Streamlit for rich visualization, communicating with an asynchronous FastAPI backend. 

The core matching logic relies on a **Dual-Engine Pipeline**:
1. **Engine 1: Exact Match Fingerprinting (Dejavu-style)**
   - Extracts a spectrogram from the audio snippet.
   - Identifies local maximum peaks (constellations).
   - Hashes pairs of peaks with their time deltas to create robust, highly compressed fingerprints.
2. **Engine 2: Fuzzy Neural Fallback (PANNs)**
   - Extracts a 2048-dimensional neural embedding using Pretrained Audio Neural Networks.
   - Uses Cosine Similarity to find the nearest acoustic match when the audio is too heavily distorted for exact fingerprinting.

**Data Flow:**
`Audio Snippet -> noisereduce preprocessing -> Engine 1 (Hash Lookup) -> Engine 2 (Fallback) -> JSON Result & Confidence`

## Setup & Execution Instructions 

### 1. Environment Setup
Install the necessary dependencies in a Python 3.9+ environment:
```bash
pip install -r requirements.txt
pip install tinytag  # Optional, for bulk ingestion metadata
```

### 2. Start the Backend API
The backend must be running for the system to process requests.
```bash
uvicorn backend.main:app --reload
```
You can verify the backend health via the `GET /` endpoint (Issue 18) or view the Swagger UI at `http://localhost:8000/docs`.

### 3. Start the UI Dashboard
Open a separate terminal window and run:
```bash
streamlit run frontend/app.py
```

### 4. Bulk Dataset Ingestion (Issue 1)
To load your initial dataset of copyright-free songs, place all `.mp3` or `.wav` files into a directory (e.g., `./data/songs/`), and run the bulk ingestion script:
```bash
python scripts/bulk_ingest.py ./data/songs/
```
This script will parse metadata and load the records into the active backend.

### 5. Automated Accuracy Evaluation (Issue 14)
To test the system against a known benchmark, create a JSON file mapping filenames to expected song titles, and run:
```bash
python scripts/evaluate_accuracy.py ./data/test_clips/ ground_truth.json
```

---

Feature Extraction & Storage: We use librosa to transform audio into spectrograms, identifying local intensity peaks (the "constellation map"). These peaks are paired and hashed using a Dejavu-style algorithm. These fingerprints are stored in a high-speed Inverted Index (Python dictionary) mapping hash -> (song_id, offset). For persistence, the index is serialized to disk using binary pickle, allowing for instant loading on startup.

Matching Algorithm: We employ a Dual-Engine Pipeline:

Engine 1 (Hash Lookup): We perform $O(1)$ lookups in our index. By calculating the time-delta between query and database hashes, we use a voting mechanism to find the most likely match.
Engine 2 (Neural Fallback): If the hashes fail due to extreme distortion, we use PANNs (Pretrained Audio Neural Networks) to extract a 2048-dimensional acoustic embedding, comparing it against the database using Cosine Similarity.
Scalability: The system is built on an Asynchronous FastAPI backend served by Uvicorn. This allows for high-concurrency, handling multiple simultaneous queries without blocking. The use of an Inverted Index ensures that lookup time remains nearly constant ($O(1)$) even as the dataset grows to thousands of songs.

Latency & Accuracy: To ensure low latency, we rely on hash-based lookups which take milliseconds. For accuracy, we implement a noisereduce preprocessing layer on all incoming queries to strip background static. The system only returns a match if it exceeds a high-confidence threshold, falling back to the neural engine when acoustic similarity is the more reliable metric.


**Note:** Please do not change the format or spelling of anything in this README. The fields are extracted using a script, so any changes to the structure or formatting may break the extraction process.
