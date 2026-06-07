from langchain_groq import ChatGroq          # ← changed to ChatGroq
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
    improved_response: str = Field(
        description="Improved version if score < 0.75, else same as input"
    )

def qa_check(state: dict) -> dict:
    """
    QA Agent — checks response quality before sending.
    If quality is below threshold, rewrites the response.
    """
    
    api_key = SecretStr(Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None
    llm = ChatGroq(model=Config.get_model(), api_key=api_key, max_tokens=800)
    structured_llm = llm.with_structured_output(QAResult)
    
    system_prompt = """You are a QA agent for customer support responses.
    
Evaluate the draft response against these criteria:
1. ACCURACY: Does it match the actual data/facts? (0.0-1.0)
2. EMPATHY: Does it acknowledge customer's frustration appropriately?
3. COMPLETENESS: Does it fully address what the customer asked?
4. TONE: Is it professional but warm?

Score 0.0-1.0 overall. If score < 0.75, provide an improved_response.
If score >= 0.75, set improved_response = the same draft response."""
    
    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
Customer message: {state['user_message']}
Customer sentiment: {state.get('sentiment', 'neutral')}
Draft response: {state['draft_response']}
Tools used: {state.get('tools_called', [])}
        """)
    ])
    
    if not isinstance(result, QAResult):
        raise TypeError("Expected QAResult from structured LLM")

    state["qa_result"] = result.model_dump()
    state["quality_score"] = result.quality_score
    
    # Use improved response if QA rewrote it
    state["final_response"] = result.improved_response
    
    return state