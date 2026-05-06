import requests

API_URL = "http://localhost:8000"
file_path = "data/raw/data/blues/blues.00000.wav"

with open(file_path, "rb") as f:
    files = {"file": ("blues.00000.wav", f, "audio/wav")}
    response = requests.post(f"{API_URL}/identify/", files=files)
    
print(response.json())
