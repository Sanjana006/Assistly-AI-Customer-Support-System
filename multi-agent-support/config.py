from dotenv import load_dotenv
import os

# Load .env from multi-agent-support/ first, then fall back to repo root
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(base_dir, ".env"))           # multi-agent-support/.env
load_dotenv(dotenv_path=os.path.join(base_dir, "..", ".env"))     # repo root .env (fallback)

def _get_secret(key: str, default: str | None = None) -> str | None:
    """Read from Streamlit secrets first (cloud), then env vars (local)."""
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

class Config:
    GROQ_API_KEY = _get_secret("GROQ_API_KEY")

    # Use the same fast model for all agents — 70b was causing 57s delays
    DEV_MODEL  = "llama-3.1-8b-instant"
    PROD_MODEL = "llama-3.1-8b-instant"

    USE_PROD_MODEL = os.getenv("USE_PROD", "false").lower() == "true"

    @classmethod
    def get_model(cls) -> str:
        return cls.PROD_MODEL if cls.USE_PROD_MODEL else cls.DEV_MODEL

    _db_path = os.path.join(base_dir, "data", "support.db")
    # Ensure database directory exists to avoid SQLite operational errors
    os.makedirs(os.path.dirname(_db_path), exist_ok=True)
    
    DATABASE_URL    = os.getenv("DATABASE_URL", f"sqlite:///{_db_path}")
    CHROMA_PATH     = os.getenv("CHROMA_PATH", os.path.join(base_dir, "knowledge_base", "chroma_db"))
    COLLECTION_NAME = "support_kb"

    MAX_RETRIES = 3

    # Raised from 0.7 — only escalate truly angry customers, not mildly negative
    ESCALATION_FRUSTRATION_THRESHOLD = 0.85

    # Raised from 0.7 — only escalate truly angry customers, not mildly negative
    QA_MIN_SCORE = 0.75

    USE_LOCAL_CLASSIFIER = os.getenv("USE_LOCAL_CLASSIFIER", "false").lower() == "true"
    LOCAL_MODEL_PATH     = os.getenv("LOCAL_MODEL_PATH", os.path.join(base_dir, "fine_tuned_classifier"))