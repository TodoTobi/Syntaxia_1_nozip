# utils/config.py
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    raise RuntimeError("Falta python-dotenv. Instalá con: pip install python-dotenv")

# Intenta cargar api.env (si existe), sino .env
root = Path(__file__).resolve().parents[1]
api_env = root / "api.env"
if api_env.exists():
    load_dotenv(api_env)
else:
    load_dotenv(root / ".env")

@dataclass
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    base_url: str = os.getenv("BASE_URL", "https://api.groq.com/openai/v1")
    llm_model: str = os.getenv("LLM_MODEL", "llama3-8b-8192")

    def validate(self):
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY no está configurada. Definila en api.env o .env."
            )

settings = Settings()
settings.validate()
 