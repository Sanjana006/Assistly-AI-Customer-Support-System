from langchain_groq import ChatGroq          # ← was: from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, SecretStr
from typing import Optional, Literal
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class ClassificationResult(BaseModel):
    intent: Literal[
        "order_status", "refund_request", "shipping_inquiry",
        "product_complaint", "account_issue", "payment_issue",
        "cancellation_request", "general_inquiry", "human_request"
    ] = Field(description="The primary intent of the customer message")
    sentiment: Literal["positive", "neutral", "negative", "angry"]
    frustration_score: float = Field(ge=0.0, le=1.0)
    urgency: Literal["low", "medium", "high"]
    order_id: Optional[str] = Field(default=None)
    customer_email: Optional[str] = Field(default=None)
    summary: str

def classify_message(state: dict) -> dict:
    api_key = SecretStr(Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None
    llm = ChatGroq(                          # ← was: ChatAnthropic(
        model=Config.get_model(),
        api_key=api_key,                     # ← was: Config.ANTHROPIC_API_KEY
        max_tokens=500
    )

    system_prompt = """You are a customer support classifier for an e-commerce platform.
Analyze the customer message and extract intent, sentiment, frustration score,
urgency, any order IDs or emails, and a brief summary.
Always respond with valid JSON matching the exact schema provided."""

    structured_llm = llm.with_structured_output(ClassificationResult)

    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Customer message: {state['user_message']}")
    ])

    if not isinstance(result, ClassificationResult):
        raise TypeError("Expected ClassificationResult from structured LLM")

    state["classification"]    = result.model_dump()
    state["intent"]            = result.intent
    state["sentiment"]         = result.sentiment
    state["frustration_score"] = result.frustration_score
    return state