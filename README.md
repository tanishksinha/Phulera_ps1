# Audio Identification & Source Detection System

An enterprise-grade, dual-engine audio identification system designed to scale to thousands of songs, handle heavy noise/distortion, and return real-time matches with high accuracy. 

Built for the **Code2Create Challenge – Round 2**.

---

## 🏗️ Core Architecture & Data Flow (Issue 16)

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

---

## 🗄️ Feature Storage Schema & Memory Optimization (Issues 2 & 15)

The system currently employs an **In-Memory Inverted Index** for storing audio fingerprints.
- **Storage Strategy**: A Python dictionary mapping `hash_value -> list[(song_id, absolute_time_offset)]`. 
- **Scalability Trade-offs**: While an in-memory dictionary is $O(1)$ for extremely rapid retrieval (satisfying the low latency constraint), it is bounded by RAM. 
- **Memory Optimization**: The hashes are shortened SHA-1 strings. For a dataset of a "few thousand songs", this comfortably fits within a standard 8GB RAM environment footprint. To scale to millions of songs, this dictionary schema is designed to be trivially swapped out for a distributed **Redis Hash Map** or a sharded **PostgreSQL** table without altering the core lookup logic.

---

## 🚀 Setup & Execution Instructions (Issue 17)

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

## 🛡️ Key Handling Strategies

* **Concurrency (Issue 10)**: The `/identify/` endpoint is an `async` FastAPI route served by Uvicorn (an ASGI server). This allows the system to handle multiple simultaneous audio queries without blocking the main event loop.
* **Invalid Queries (Issue 12)**: The system inspects uploaded arrays. If the audio is completely silent (array of zeros) or less than 1 second long, it rejects the query with a `400 Bad Request` prior to wasting CPU cycles on feature extraction.
* **Latency Tracking (Issue 13)**: The backend runs a middleware interceptor that calculates total processing time and injects it into the HTTP response header `X-Process-Time`, logging it to the console for telemetry.
