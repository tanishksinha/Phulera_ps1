import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import time

# We'll use absolute imports assuming we run the server from the project root.
from backend.services.audio_processing import load_and_preprocess_audio
from backend.services.fingerprint import fingerprint_audio, db
from backend.services.neural_match import get_audio_embedding, neural_db, cosine_similarity

app = FastAPI(title="Audio ID System")

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
    song_id: str = None,
    title: str = "Unknown Title",
    artist: str = "Unknown Artist"
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
        metadata = {"title": title, "artist": artist}
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
        best_match_id, match_count = db.match_hashes(query_hashes)
        
        fingerprint_confidence = 0.0
        if match_count > 0:
            # Simple heuristic for confidence based on matched hash count
            fingerprint_confidence = min(match_count / 15.0, 1.0)
            
        # --- Engine 2: Neural Embedding (Fuzzy Match / Fallback) ---
        emb = get_audio_embedding(audio_data, sr)
        best_neural_id, neural_confidence = neural_db.find_best_match(emb)
        
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
            
        if final_id and final_id in SONG_METADATA:
            metadata = SONG_METADATA[final_id]
            return {
                "match": True,
                "song_id": final_id,
                "title": metadata["title"],
                "artist": metadata["artist"],
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
    return {"status": "ok", "message": "Audio ID System is running."}


# formatted
