import os, requests
from dotenv import load_dotenv
from utils import config

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY", config.GROQ_API_KEY)
API_URL = "https://api.groq.com/openai/v1/models"

headers = {"Authorization": f"Bearer {API_KEY}"}
resp = requests.get(API_URL, headers=headers)
data = resp.json()

for modelo in data.get("data", []):
    print(modelo["id"])
