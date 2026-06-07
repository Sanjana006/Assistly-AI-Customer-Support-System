from config import Config

def check_escalation(state: dict) -> dict:
    reasons = []

    # ✅ Rule 0: If waiting for confirmation → NEVER escalate
    if state.get("awaiting_confirmation"):
        state["needs_escalation"] = False
        state["escalation_reasons"] = []
        return state

    # ✅ Rule 1: Explicit human request → always escalate
    if state.get("intent") == "human_request":
        state["needs_escalation"] = True
        state["escalation_reasons"] = ["Customer requested human agent"]
        return state

    # ✅ Rule 2: NEVER auto-escalate refund requests
    if state.get("intent") == "refund_request":
        state["needs_escalation"] = False
        state["escalation_reasons"] = []
        return state

    # Rule 3: High frustration + poor QA → escalate
    if (
        state.get("frustration_score", 0) >= Config.ESCALATION_FRUSTRATION_THRESHOLD
        and state.get("quality_score", 1) < Config.QA_MIN_SCORE
    ):
        reasons.append(
            f"High frustration ({state['frustration_score']:.2f}) + low QA ({state['quality_score']:.2f})"
        )

    # Rule 4: High urgency + negative sentiment
    if (
        state.get("urgency") == "high"
        and state.get("sentiment") in ["negative", "angry"]
    ):
        reasons.append("High urgency + negative sentiment")

    state["needs_escalation"] = len(reasons) > 0
    state["escalation_reasons"] = reasons

    if state["needs_escalation"]:
        state["final_response"] = (
            "I'm connecting you with a senior support specialist who can better assist you. "
            "Expected wait: 5-10 minutes.\n\n"
            + state.get("final_response", "")
        )

    return state