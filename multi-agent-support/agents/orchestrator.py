from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from loguru import logger
import time

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
    active_order_id: str | None = None,
) -> dict:
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

    try:
        final_state = support_graph.invoke(initial_state)
        final_state["processing_time_ms"] = (time.time() - start_time) * 1000
        logger.info(
            f"[{final_state['ticket_id']}] Done in {final_state['processing_time_ms']:.0f}ms | "
            f"Intent: {final_state['intent']} | "
            f"Escalated: {final_state['needs_escalation']}"
        )
        return final_state
    except Exception as e:
        logger.error(f"Error processing ticket: {e}")
        raise