from dotenv import load_dotenv
import os

# Load .env relative to config.py directory
base_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Use the same fast model for all agents — 70b was causing 57s delays
    DEV_MODEL  = "llama-3.1-8b-instant"
    PROD_MODEL = "llama-3.1-8b-instant"

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

    USE_LOCAL_CLASSIFIER = os.getenv("USE_LOCAL_CLASSIFIER", "false").lower() == "true"
    LOCAL_MODEL_PATH     = os.getenv("LOCAL_MODEL_PATH", "./fine_tuned_classifier")