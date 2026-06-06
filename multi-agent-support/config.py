from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Model settings (Groq)
    DEV_MODEL = "mixtral-8x7b-32768"
    PROD_MODEL = "llama3-70b-8192"
    
    # Use this flag to switch modes
    USE_PROD_MODEL = os.getenv("USE_PROD", "false").lower() == "true"
    
    @classmethod
    def get_model(cls) -> str:
        return cls.PROD_MODEL if cls.USE_PROD_MODEL else cls.DEV_MODEL
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/support.db")
    
    # ChromaDB
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./knowledge_base/chroma_db")
    COLLECTION_NAME = "support_kb"
    
    # Agent settings
    MAX_RETRIES = 3
    ESCALATION_FRUSTRATION_THRESHOLD = 0.7  # frustration score 0-1
    QA_MIN_SCORE = 0.75                    # min quality to send response