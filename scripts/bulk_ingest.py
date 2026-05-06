import os
import requests
import argparse
import time

# Optional metadata extraction
try:
    from tinytag import TinyTag
    HAS_TINYTAG = True
except ImportError:
    HAS_TINYTAG = False

API_URL = "http://localhost:8000"

def bulk_ingest(directory_path: str):
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return

    print(f"Scanning {directory_path} for audio files...")
    
    valid_extensions = {".mp3", ".wav", ".flac", ".ogg"}
    files_to_ingest = []
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                files_to_ingest.append(os.path.join(root, file))
                
    if not files_to_ingest:
        print("No audio files found.")
        return
        
    print(f"Found {len(files_to_ingest)} files. Starting ingestion...")
    
    success_count = 0
    start_time = time.time()
    
    for file_path in files_to_ingest:
        filename = os.path.basename(file_path)
        title = filename
        artist = "Unknown"
        
        if HAS_TINYTAG:
            try:
                tag = TinyTag.get(file_path)
                if tag.title:
                    title = tag.title
                if tag.artist:
                    artist = tag.artist
            except Exception:
                pass
                
        print(f"Ingesting: {title} by {artist}...")
        
        try:
            with open(file_path, "rb") as f:
                # The backend expects 'file' parameter
                files = {"file": (filename, f, "audio/mpeg" if filename.endswith(".mp3") else "audio/wav")}
                data = {"title": title, "artist": artist}
                
                response = requests.post(f"{API_URL}/ingest/", files=files, data=data)
                
                if response.status_code == 200:
                    success_count += 1
                else:
                    print(f"  Failed: {response.text}")
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            
    elapsed = time.time() - start_time
    print(f"--- Ingestion Complete ---")
    print(f"Successfully ingested {success_count}/{len(files_to_ingest)} files in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk ingest an initial dataset of songs.")
    parser.add_argument("directory", help="Path to the directory containing audio files.")
    args = parser.parse_args()
    
    # Check if API is running
    try:
        requests.get(f"{API_URL}/")
    except requests.exceptions.ConnectionError:
        print("Error: Backend API is not running. Please start it with 'uvicorn backend.main:app' first.")
        exit(1)
        
    bulk_ingest(args.directory)
