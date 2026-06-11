from dotenv import load_dotenv
import os

# Load .env from multi-agent-support/ first, then fall back to repo root
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(base_dir, ".env"))           # multi-agent-support/.env
load_dotenv(dotenv_path=os.path.join(base_dir, "..", ".env"))     # repo root .env (fallback)

def _is_writable_path(path: str) -> bool:
    dir_path = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(dir_path, exist_ok=True)
        test_file = os.path.join(dir_path, f".write_test_{os.getpid()}")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception:
        return False

# ── Streamlit Cloud: inject secrets into os.environ ──────────────────────────
# st.secrets is available when running on Streamlit Cloud. Injecting here
# ensures all os.getenv() calls below (and in third-party libraries like
# langchain-groq) pick up the right values without needing per-call lookups.
try:
    import streamlit as st
    # Only access st.secrets if on Streamlit Cloud or if a secrets.toml file exists,
    # as querying st.secrets on recent Streamlit versions crashes with a 'No secrets found' screen.
    _has_secrets = (
        os.path.exists("/mount/src")  # Streamlit Cloud
        or os.path.exists(os.path.expanduser("~/.streamlit/secrets.toml"))
        or os.path.exists(os.path.join(os.getcwd(), ".streamlit", "secrets.toml"))
        or os.path.exists(os.path.join(base_dir, ".streamlit", "secrets.toml"))
        or os.path.exists(os.path.join(base_dir, "..", ".streamlit", "secrets.toml"))
    )
    if _has_secrets:
        for _key in ["GROQ_API_KEY", "DATABASE_URL", "CHROMA_PATH", "USE_PROD"]:
            if _key in st.secrets and not os.getenv(_key):
                os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass  # Running locally or st.secrets not available — .env values are used

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Use the same fast model for all agents — 70b was causing 57s delays
    DEV_MODEL  = "llama-3.1-8b-instant"
    PROD_MODEL = "llama-3.1-8b-instant"

    USE_PROD_MODEL = os.getenv("USE_PROD", "false").lower() == "true"

    @classmethod
    def get_model(cls) -> str:
        return cls.PROD_MODEL if cls.USE_PROD_MODEL else cls.DEV_MODEL

    _db_path = os.path.join(base_dir, "data", "support.db")
    _chroma_default = os.path.join(base_dir, "knowledge_base", "chroma_db")

    # Detect Streamlit Cloud or other standard hosting platforms (Render, Heroku, Railway, HF)
    _is_deployed = (
        os.path.exists("/mount/src")
        or os.getenv("PORT") is not None
        or os.getenv("RENDER") is not None
        or os.getenv("RAILWAY_STATIC_URL") is not None
    )

    # If deployed or default db path is not writable, fall back to /tmp
    if _is_deployed or not _is_writable_path(_db_path):
        _db_path = "/tmp/support.db"
        _chroma_default = "/tmp/chroma_db"
    else:
        # Ensure database directory exists to avoid SQLite operational errors
        os.makedirs(os.path.dirname(_db_path), exist_ok=True)

    DATABASE_URL    = os.getenv("DATABASE_URL", f"sqlite:///{_db_path}")
    CHROMA_PATH     = os.getenv("CHROMA_PATH", _chroma_default)

    # Force using writable directory /tmp for SQLite and Chroma in deployed/read-only environments
    _db_dir = DATABASE_URL.replace("sqlite:///", "") if DATABASE_URL.startswith("sqlite://") else ""
    if _is_deployed or not _db_dir or not _is_writable_path(_db_dir):
        if DATABASE_URL.startswith("sqlite://"):
            db_name = os.path.basename(DATABASE_URL) or "support.db"
            DATABASE_URL = f"sqlite:////tmp/{db_name}"
        
        if CHROMA_PATH and not CHROMA_PATH.startswith("/tmp"):
            src_chroma = CHROMA_PATH
            chroma_dir = os.path.basename(CHROMA_PATH) or "chroma_db"
            dest_chroma = f"/tmp/{chroma_dir}"
            if os.path.exists(src_chroma) and not os.path.exists(dest_chroma):
                import shutil
                try:
                    shutil.copytree(src_chroma, dest_chroma)
                except Exception as e:
                    print(f"Warning: Failed to copy chroma_db to {dest_chroma}: {e}")
            CHROMA_PATH = dest_chroma
    COLLECTION_NAME = "support_kb"

    MAX_RETRIES = 3

    # Raised from 0.7 — only escalate truly angry customers, not mildly negative
    ESCALATION_FRUSTRATION_THRESHOLD = 0.85

    # Raised from 0.7 — only escalate truly angry customers, not mildly negative
    QA_MIN_SCORE = 0.75

    USE_LOCAL_CLASSIFIER = os.getenv("USE_LOCAL_CLASSIFIER", "false").lower() == "true"
    LOCAL_MODEL_PATH     = os.getenv("LOCAL_MODEL_PATH", os.path.join(base_dir, "fine_tuned_classifier"))