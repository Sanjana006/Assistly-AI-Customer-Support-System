from langchain_groq import ChatGroq
from pydantic import SecretStr
from config import Config

def get_groq_client():
    api_key = SecretStr(Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None
    return ChatGroq(
        model=Config.get_model(),
        api_key=api_key,
        max_tokens=500,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
