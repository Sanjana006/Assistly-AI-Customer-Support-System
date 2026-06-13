from langchain_groq import ChatGroq          # ← changed
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, SecretStr
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

from typing import Optional
import time
from groq import RateLimitError
class QAResult(BaseModel):
    quality_score: float = Field(ge=0.0, le=1.0)
    is_accurate: Optional[bool] = True
    is_empathetic: Optional[bool] = True
    is_complete: Optional[bool] = True
    issues: list[str] = Field(default_factory=list)
    improved_response: str

def extract_order_id_from_state(state: dict) -> Optional[str]:
    # 1. Try pending_order_id
    if state.get("pending_order_id"):
        return state["pending_order_id"]
    
    # 2. Try tools_called args
    for call in state.get("tools_called", []):
        args = call.get("args") or {}
        if "order_id" in args and args["order_id"]:
            return args["order_id"]
            
    # 3. Try classification order_id
    if state.get("classification", {}).get("order_id"):
        return state["classification"]["order_id"]
        
    return None

def safe_invoke(llm, messages, retries: int = 3, backoff: int = 10):
    """Invoke LLM with retry on Groq rate limit errors."""
    attempt = 0
    while True:
        try:
            return llm.invoke(messages)
        except RateLimitError:
            attempt += 1
            if attempt > retries:
                raise
            time.sleep(backoff)
            backoff *= 2

def qa_check(state: dict) -> dict:
    api_key = SecretStr(Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None
    llm = ChatGroq(                          # ← changed
        model=Config.get_model(),
        api_key=api_key,                     # ← changed
        max_tokens=800
    )
    structured_llm = llm.with_structured_output(QAResult, method="json_mode")

    order_id = extract_order_id_from_state(state)
    order_info = None
    if order_id:
        try:
            from tools.order_lookup import get_order_by_id
            order_info = get_order_by_id.invoke({"order_id": order_id})
        except Exception:
            pass

    system_prompt = """You are a QA auditor agent for customer support responses.
Score the draft response 0.0-1.0 on accuracy, empathy, and completeness.

Instructions for 'improved_response':
- If quality_score >= 0.75, you MUST copy the draft response into 'improved_response' completely verbatim, without any modifications, placeholders, or added text.
- If quality_score < 0.75, you must rewrite it in 'improved_response'.
- The 'improved_response' must ONLY contain the final clean message to be sent to the customer. Do NOT include any debugging logs, prompt prefixes, JSON blocks, database fields, or tool calls inside 'improved_response'.

Validation Rules:
1. Verify specific details (like refund IDs, tracking numbers, and amounts) in the draft response against the 'Tools used' list and 'Order Ground Truth'.
2. If a refund or replacement tool was executed successfully in the 'Tools used' list (e.g. 'process_refund' or 'process_replacement'), it is correct for the draft response to say that the refund/replacement has been processed. The status of the order in 'Order Ground Truth' might be 'cancelled' or 'replacement_pending' as a result of that tool execution. This is expected and correct. Do NOT mark it as an error or reject the draft refund/replacement.
3. Verify the order return policy against the return_policy in 'Order Ground Truth'. If the draft response correctly explains that an item is 'non_returnable' (Final Sale) or 'replacement_only' (Replacement-Only) and does not qualify for a refund, that is correct.
4. Offering store credit coupons (e.g., ₹500 credit) or product replacements to appease a customer for transit delays or product damage is ALWAYS valid and allowed, even if the product itself is 'non_returnable' (Final Sale). Do NOT rewrite or block delay/damage appeasement offers.
"""

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

    result = safe_invoke(structured_llm, [
        SystemMessage(content=groq_prompt),
        HumanMessage(content=f"""
Customer message: {state['user_message']}
Customer sentiment: {state.get('sentiment', 'neutral')}
Draft response: {state['draft_response']}
Tools used: {state.get('tools_called', [])}
Order Ground Truth: {order_info if order_info else 'No order selected'}
        """)
    ])


    if not isinstance(result, QAResult):
        raise TypeError("Expected QAResult from structured LLM")

    improved = result.improved_response
    if improved.strip().startswith("{") and improved.strip().endswith("}"):
        try:
            import json
            parsed = json.loads(improved)
            if isinstance(parsed, dict):
                found = False
                for k in ["draft", "draft_response", "improved_response", "response", "message", "content", "final_response"]:
                    if k in parsed:
                        improved = parsed[k]
                        found = True
                        break
                if not found:
                    # Fallback to the longest string in the parsed dict
                    longest_str = ""
                    for v in parsed.values():
                        if isinstance(v, str) and len(v) > len(longest_str):
                            longest_str = v
                    if longest_str:
                        improved = longest_str
        except Exception:
            pass

    state["qa_result"]      = result.model_dump()
    state["quality_score"]  = result.quality_score
    state["final_response"] = improved
    return state