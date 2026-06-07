from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, cast
from loguru import logger
import time

from config import Config

from agents.classifier import classify_message
from agents.resolver import resolve_ticket
from agents.qa_agent import qa_check
from agents.escalation import check_escalation

class SupportState(TypedDict):
    # Input
    user_message: str
    conversation_history: list[dict]

    # After classifier
    classification: dict
    intent: str
    sentiment: str
    frustration_score: float
    urgency: str

    # Confirmation flow
    # When resolver wants to do a destructive action (refund, cancel),
    # it sets this instead of doing it immediately
    pending_action: Optional[str]      # e.g. "refund" | "cancel" | None
    pending_order_id: Optional[str]    # the order the action applies to
    pending_action_reason: Optional[str] # reason for refund/replacement
    awaiting_confirmation: bool        # True = bot is waiting for yes/no

    # After resolver
    draft_response: str
    tools_called: list[dict]

    # After QA
    final_response: str
    quality_score: float
    qa_result: dict

    # After escalation
    needs_escalation: bool
    escalation_reasons: list[str]

    # Metadata
    processing_time_ms: float
    ticket_id: str


def should_escalate(state: SupportState) -> str:
    if state.get("intent") == "human_request":
        return "escalate"
    return "resolve"


def build_graph():
    graph = StateGraph(SupportState)

    graph.add_node("classify",         classify_message)
    graph.add_node("resolve",          resolve_ticket)
    graph.add_node("qa",               qa_check)
    graph.add_node("escalation_check", check_escalation)

    def human_escalation_node(state: SupportState) -> SupportState:
        state["final_response"] = (
            "I completely understand. Connecting you with a human agent now. "
            "Please hold — average wait time is 3-5 minutes."
        )
        state["needs_escalation"]   = True
        state["escalation_reasons"] = ["Customer explicitly requested human"]
        return state

    # Removed stray error handling block (handled later in process_ticket)

    graph.add_node("escalate", human_escalation_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        should_escalate,
        {"escalate": "escalate", "resolve": "resolve"}
    )
    graph.add_edge("resolve",          "qa")
    graph.add_edge("qa",               "escalation_check")
    graph.add_edge("escalation_check", END)
    graph.add_edge("escalate",         END)

    return graph.compile()


support_graph = build_graph()


def process_ticket(
    user_message: str,
    conversation_history: list | None = None,
    # Pass these in when resuming a confirmation flow
    pending_action: str | None = None,
    pending_order_id: str | None = None,
    pending_action_reason: str | None = None,
    active_order_id: str | None = None,
) -> SupportState:
    import uuid

    start_time = time.time()

    initial_state: SupportState = {
        "user_message":          user_message,
        "conversation_history":  conversation_history or [],
        "ticket_id":             str(uuid.uuid4())[:8].upper(),

        "classification":        {"order_id": active_order_id} if active_order_id else {},
        "intent":                "",
        "sentiment":             "",
        "frustration_score":     0.0,
        "urgency":               "",

        # Carry forward any pending action from the previous turn
        "pending_action":        pending_action,
        "pending_order_id":      pending_order_id,
        "pending_action_reason": pending_action_reason,
        "awaiting_confirmation": False,

        "draft_response":        "",
        "tools_called":          [],
        "final_response":        "",
        "quality_score":         0.0,
        "qa_result":             {},
        "needs_escalation":      False,
        "escalation_reasons":    [],
        "processing_time_ms":    0.0,
    }

    logger.info(f"[{initial_state['ticket_id']}] Processing: {user_message[:80]}...")

    # Attempt to invoke the graph with retry logic for rate limiting
    attempts = 0
    while True:
        try:
            final_state = support_graph.invoke(initial_state)
            final_state = cast(SupportState, final_state)  # Ensure static type matches
            break
        except Exception as e:
            # Import RateLimitError for specific handling
            try:
                from groq import RateLimitError
            except ImportError:
                # Fallback to a generic Exception if the specific class is unavailable
                RateLimitError = Exception
            if isinstance(e, RateLimitError):
                attempts += 1
                if attempts >= Config.MAX_RETRIES:
                    logger.error(f"Groq rate limit exceeded after {attempts} attempts: {e}")
                    # Escalate after retries exhausted
                    state: SupportState = initial_state  # preserve TypedDict type
                    state["draft_response"] = (
                        "I encountered a technical issue while processing this request. "
                        "I am immediately connecting you with a human support specialist to assist you further."
                    )
                    state["needs_escalation"] = True
                    state["escalation_reasons"] = [f"LLM RateLimitError after {attempts} retries: {str(e)}"]
                    return cast(SupportState, state)
                backoff = 5 * (2 ** (attempts - 1))
                logger.warning(f"Groq rate limit hit, retrying after {backoff}s (attempt {attempts}/{Config.MAX_RETRIES})")
                time.sleep(backoff)
                continue
            else:
                # Non-rate limit errors trigger normal escalation
                logger.error(f"Error processing ticket: {e}")
                state: SupportState = initial_state  # preserve TypedDict type
                state["draft_response"] = (
                    "I encountered a technical issue while processing this request. "
                    "I am immediately connecting you with a human support specialist to assist you further."
                )
                state["needs_escalation"] = True
                state["escalation_reasons"] = [f"LLM Tool Call Error: {str(e)}"]
                return cast(SupportState, state)

    final_state["processing_time_ms"] = (time.time() - start_time) * 1000
    logger.info(
        f"[{final_state['ticket_id']}] Done in {final_state['processing_time_ms']:.0f}ms | "
        f"Intent: {final_state['intent']} | "
        f"Escalated: {final_state['needs_escalation']}"
    )
    return final_state