from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Groq model names — llama3 is fast + free, mixtral is smarter
    DEV_MODEL  = "llama3-8b-8192"       # fastest, cheapest, great for dev
    PROD_MODEL = "llama3-70b-8192"      # smarter, still very fast on Groq

    USE_PROD_MODEL = os.getenv("USE_PROD", "false").lower() == "true"

    @classmethod
    def get_model(cls) -> str:
        return cls.PROD_MODEL if cls.USE_PROD_MODEL else cls.DEV_MODEL

    # Everything below is unchanged
    DATABASE_URL  = os.getenv("DATABASE_URL", "sqlite:///./data/support.db")
    CHROMA_PATH   = os.getenv("CHROMA_PATH", "./knowledge_base/chroma_db")
    COLLECTION_NAME = "support_kb"

    MAX_RETRIES = 3
    ESCALATION_FRUSTRATION_THRESHOLD = 0.7
    QA_MIN_SCORE = 0.75