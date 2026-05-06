import asyncio
from backend.main import identify_audio, SONG_METADATA, db, neural_db
from fastapi import UploadFile
import io

print("Songs loaded:", len(SONG_METADATA))
print("Fingerprints loaded:", len(db.hash_table))

async def test():
    with open("data/raw/data/blues/blues.00000.wav", "rb") as f:
        file_bytes = f.read()
    
    upload_file = UploadFile(filename="blues.00000.wav", file=io.BytesIO(file_bytes))
    result = await identify_audio(upload_file)
    print("Result:", result)

asyncio.run(test())
