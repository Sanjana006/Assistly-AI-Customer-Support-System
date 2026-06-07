from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Use the same fast model for all agents — 70b was causing 57s delays
    DEV_MODEL  = "llama-3.1-8b-instant"
    PROD_MODEL = "llama-3.1-8b-instant"   # keep instant for now, fast enough

    USE_PROD_MODEL = os.getenv("USE_PROD", "false").lower() == "true"

    @classmethod
    def get_model(cls) -> str:
        return cls.PROD_MODEL if cls.USE_PROD_MODEL else cls.DEV_MODEL

    DATABASE_URL    = os.getenv("DATABASE_URL", "sqlite:///./data/support.db")
    CHROMA_PATH     = os.getenv("CHROMA_PATH", "./knowledge_base/chroma_db")
    COLLECTION_NAME = "support_kb"

    MAX_RETRIES = 3

    # Raised from 0.7 — only escalate truly angry customers, not mildly negative
    ESCALATION_FRUSTRATION_THRESHOLD = 0.85

    QA_MIN_SCORE = 0.75