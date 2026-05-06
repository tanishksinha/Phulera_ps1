import requests

API_URL = "http://localhost:8000"
response = requests.get(f"{API_URL}/")
print(response.json())
