from typing import TypedDict, Optional

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
