from langchain_groq import ChatGroq          # ← changed
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, SecretStr
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class QAResult(BaseModel):
    quality_score: float = Field(ge=0.0, le=1.0)
    is_accurate: bool
    is_empathetic: bool
    is_complete: bool
    issues: list[str] = Field(default_factory=list)
    improved_response: str

def qa_check(state: dict) -> dict:
    api_key = SecretStr(Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None
    llm = ChatGroq(                          # ← changed
        model=Config.get_model(),
        api_key=api_key,                     # ← changed
        max_tokens=800
    )
    structured_llm = llm.with_structured_output(QAResult, method="json_mode")

    system_prompt = """You are a QA agent for customer support responses.
Score 0.0-1.0 on accuracy, empathy, and completeness.
If score < 0.75, rewrite the response in improved_response.
If score >= 0.75, copy the draft into improved_response unchanged."""

    # Append schema instructions for JSON mode
    groq_prompt = system_prompt + """

You MUST return your response as a valid JSON object matching the following Pydantic schema:
{
  "quality_score": float,
  "is_accurate": boolean,
  "is_empathetic": boolean,
  "is_complete": boolean,
  "issues": list of strings,
  "improved_response": string
}

Ensure all keys are present with the exact spelling. Return raw JSON only."""

    result = structured_llm.invoke([
        SystemMessage(content=groq_prompt),
        HumanMessage(content=f"""
Customer message: {state['user_message']}
Customer sentiment: {state.get('sentiment', 'neutral')}
Draft response: {state['draft_response']}
Tools used: {state.get('tools_called', [])}
        """)
    ])

    if not isinstance(result, QAResult):
        raise TypeError("Expected QAResult from structured LLM")

    state["qa_result"]      = result.model_dump()
    state["quality_score"]  = result.quality_score
    state["final_response"] = result.improved_response
    return state