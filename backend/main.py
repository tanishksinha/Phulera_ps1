import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import time
import pickle

# We'll use absolute imports assuming we run the server from the project root.
from backend.services.audio_processing import load_and_preprocess_audio
from backend.services.fingerprint import fingerprint_audio, db
from backend.services.neural_match import get_audio_embedding, neural_db, cosine_similarity

app = FastAPI(title="Audio ID System")

DB_DIR = "data/db"
FINGERPRINT_DB_PATH = os.path.join(DB_DIR, "fingerprints.pkl")
NEURAL_DB_PATH = os.path.join(DB_DIR, "neural.pkl")
METADATA_PATH = os.path.join(DB_DIR, "metadata.pkl")

@app.on_event("startup")
def load_databases():
    print("Loading databases from disk...")
    db.load_from_disk(FINGERPRINT_DB_PATH)
    neural_db.load_from_disk(NEURAL_DB_PATH)
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            global SONG_METADATA
            SONG_METADATA = pickle.load(f)
    print(f"Loaded {len(SONG_METADATA)} songs.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    # Log latency for telemetry
    print(f"[{request.method}] {request.url.path} - Latency: {process_time:.4f}s")
    return response

# In-memory storage for ingested songs
# In a real system, this would be a database like PostgreSQL or MongoDB
# song_id -> {title, artist, genre}
SONG_METADATA = {}

@app.post("/ingest/")
async def ingest_audio(
    file: UploadFile = File(...),
    song_id: Optional[str] = Form(None),
    title: str = Form("Unknown Title"),
    artist: str = Form("Unknown Artist")
):
    """
    Ingests an audio file, extracts features, and adds it to the DB.
    """
    if song_id is None:
        import uuid
        song_id = str(uuid.uuid4())
        
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        # Load and preprocess
        audio_data, sr = load_and_preprocess_audio(temp_path)
        
        # 1. Fingerprinting
        hashes = fingerprint_audio(audio_data, sr)
        metadata = {"title": title, "artist": artist, "filename": file.filename}
        SONG_METADATA[song_id] = metadata
        db.insert_song(song_id, metadata, hashes)
        
        # 2. Neural Embedding (using 32000 sr for PANNs if we resampled, else using current sr)
        # In a robust system we'd resample specifically for PANNs if needed.
        emb = get_audio_embedding(audio_data, sr)
        neural_db.insert_embedding(song_id, emb)
        
        return {"status": "success", "song_id": song_id, "hashes_extracted": len(hashes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/identify/")
async def identify_audio(file: UploadFile = File(...)):
    """
    Identifies a song from an audio clip using the dual-engine approach.
    """
    temp_path = f"temp_query_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        # Load and preprocess
        audio_data, sr = load_and_preprocess_audio(temp_path)
        
        # Issue 12: Handle Invalid Query Inputs
        if len(audio_data) < sr * 1.0: # Less than 1 second
            raise HTTPException(status_code=400, detail="Audio snippet too short. Must be at least 1 second.")
        if np.max(np.abs(audio_data)) == 0:
            raise HTTPException(status_code=400, detail="Audio snippet is completely silent.")
        
        # --- Engine 1: Fingerprinting (Exact Match) ---
        query_hashes = fingerprint_audio(audio_data, sr)
        print(f"DEBUG: Extracted {len(query_hashes)} query hashes.")
        best_match_id, match_count = db.match_hashes(query_hashes)
        print(f"DEBUG: Fingerprint best match: {best_match_id} with {match_count} matches.")
        
        fingerprint_confidence = 0.0
        if match_count > 0:
            # Simple heuristic for confidence based on matched hash count
            fingerprint_confidence = min(match_count / 15.0, 1.0)
            
        # --- Engine 2: Neural Embedding (Fuzzy Match / Fallback) ---
        emb = get_audio_embedding(audio_data, sr)
        best_neural_id, neural_confidence = neural_db.find_best_match(emb)
        print(f"DEBUG: Neural best match: {best_neural_id} with confidence {neural_confidence}.")
        
        # Decide which engine to trust more
        # Fingerprint is usually more accurate if it finds a match (>0.3 confidence)
        final_id = None
        final_confidence = 0.0
        engine_used = "none"
        
        if fingerprint_confidence > 0.3 and best_match_id:
            final_id = best_match_id
            final_confidence = fingerprint_confidence
            engine_used = "fingerprint"
        elif best_neural_id and neural_confidence > 0.7:
            final_id = best_neural_id
            final_confidence = neural_confidence
            engine_used = "neural"
            
        print(f"DEBUG: final_id={final_id}, final_confidence={final_confidence}, engine_used={engine_used}")
        print(f"DEBUG: Is final_id in SONG_METADATA? {final_id in SONG_METADATA if final_id else False}")

        if final_id and final_id in SONG_METADATA:
            metadata = SONG_METADATA[final_id]
            return {
                "match": True,
                "song_id": final_id,
                "title": metadata["title"],
                "artist": metadata["artist"],
                "filename": metadata.get("filename", "N/A"),
                "confidence": final_confidence,
                "engine": engine_used
            }
        else:
            return {
                "match": False,
                "message": "No match found."
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Audio ID System is running.", "songs_indexed": len(SONG_METADATA)}

@app.post("/save_db/")
def save_database():
    print("Saving databases to disk...")
    os.makedirs(DB_DIR, exist_ok=True)
    db.save_to_disk(FINGERPRINT_DB_PATH)
    neural_db.save_to_disk(NEURAL_DB_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(SONG_METADATA, f)
    return {"status": "success", "message": "Databases saved to disk."}


# formatted
