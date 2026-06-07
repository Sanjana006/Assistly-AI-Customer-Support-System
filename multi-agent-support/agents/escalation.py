from config import Config

def check_escalation(state: dict) -> dict:
    """
    Escalation Agent — decides if human handoff is needed.
    No LLM needed here — pure logic. Fast and cheap.
    """
    
    reasons = []
    
    # Rule 1: Customer explicitly asked for human
    if state.get("intent") == "human_request":
        reasons.append("Customer requested human agent")
    
    # Rule 2: Very high frustration
    if state.get("frustration_score", 0) >= Config.ESCALATION_FRUSTRATION_THRESHOLD:
        reasons.append(f"High frustration score: {state['frustration_score']:.2f}")
    
    # Rule 3: Low quality response after QA
    if state.get("quality_score", 1) < Config.QA_MIN_SCORE:
        reasons.append(f"Low QA score: {state['quality_score']:.2f}")
    
    # Rule 4: High urgency + negative sentiment
    if (state.get("urgency") == "high" and 
        state.get("sentiment") in ["negative", "angry"]):
        reasons.append("High urgency + negative sentiment")
    
    needs_escalation = len(reasons) > 0
    
    state["needs_escalation"] = needs_escalation
    state["escalation_reasons"] = reasons
    
    if needs_escalation:
        # Prepend escalation notice to response
        state["final_response"] = (
            f"I'm connecting you with a senior support specialist who can better assist you. "
            f"Expected wait: 5-10 minutes.\n\n"
            f"In the meantime: {state.get('final_response', '')}"
        )
    
    return state