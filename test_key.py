import os
import httpx
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

print(f"Testing key starting with: {api_key[:10]}...")

response = httpx.post(
    "https://api.openai.com/v1/embeddings",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={
        "input": "test",
        "model": "text-embedding-3-small"
    }
)

print(f"Status: {response.status_code}")
if response.status_code != 200:
    print(f"Error: {response.text}")
else:
    print("Success!")
