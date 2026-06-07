from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from pydantic import SecretStr
import json, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from tools.order_lookup import (
    get_order_by_id,
    get_orders_by_customer_email,
    get_order_journey_details,
    get_customer_incident_profile
)
from tools.refund_processor import process_refund, get_refund_status
from tools.kb_search import search_knowledge_base

RESOLVER_TOOLS = [
    get_order_by_id,
    get_orders_by_customer_email,
    get_order_journey_details,
    get_customer_incident_profile,
    process_refund,
    get_refund_status,
    search_knowledge_base,
]

# (UNCHANGED — as requested)
YES_WORDS = {"yes", "yeah", "yep", "confirm", "confirmed", "sure", "proceed",
             "go ahead", "do it", "please", "ok", "okay", "absolutely", "y"}
NO_WORDS  = {"no", "nope", "cancel", "stop", "dont", "don't", "nevermind",
             "never mind", "skip", "n", "nah"}

# ✅ FIXED (without changing your word lists — only logic)
def _is_confirmation(message: str) -> bool:
    cleaned = message.lower().strip().rstrip("!.,")
    return cleaned in YES_WORDS   # exact match only

def _is_rejection(message: str) -> bool:
    cleaned = message.lower().strip().rstrip("!.,")
    return cleaned in NO_WORDS    # exact match only


def resolve_ticket(state: dict) -> dict:
    classification  = state.get("classification", {})
    pending_action  = state.get("pending_action")
    pending_order   = state.get("pending_order_id")
    user_message    = state["user_message"]

    # ⚡ Proactive Delay Check on Greeting
    is_greeting = user_message.lower().strip().rstrip("!.,") in {
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "hi there", "hello there"
    }
    if is_greeting and classification.get("order_id"):
        order_id = classification["order_id"]
        order_info = get_order_by_id.invoke({"order_id": order_id})
        if not order_info.get("error") and order_info.get("status") == "delayed":
            # Proactive intervention!
            state["draft_response"] = (
                f"👋 **Proactive Service Alert** 👋\n\n"
                f"Hello {order_info.get('customer_name', '')}! I noticed you have order **{order_id.upper()}** selected.\n\n"
                f"I wanted to proactively flag that this order is currently **delayed**. "
                f"Our system shows a transit holdup for your package.\n\n"
                f"To make things right immediately, I can:\n"
                f"1. **Initiate an instant refund** for this order.\n"
                f"2. **Escalate** this to our premium logistics team for immediate delivery resolution.\n\n"
                f"How would you like to proceed? Please let me know!\n\n"
                f"— QuickShop Support Team"
            )
            state["tools_called"] = [{"tool": "get_order_by_id", "args": {"order_id": order_id}}]
            state["pending_action"] = None
            state["pending_order_id"] = None
            state["awaiting_confirmation"] = False
            return state

    # ── CONFIRMATION FLOW (UNCHANGED LOGIC) ─────────────────────
    if pending_action and pending_order:
        if _is_confirmation(user_message):
            if pending_action == "refund":
                result = process_refund.invoke({
                    "order_id": pending_order,
                    "reason":   "Customer confirmed refund request"
                })
                if result.get("success"):
                    state["draft_response"] = (
                        f"✅ Done! Your refund has been successfully processed.\n\n"
                        f"**Refund ID:** {result['refund_id']}\n"
                        f"**Amount:** ₹{result['amount']:,.2f}\n"
                        f"**Order:** {result['order_id']} is now cancelled.\n"
                        f"**Inventory:** Stock restored for '{result['product']}'.\n\n"
                        f"The amount will reflect in your original payment method "
                        f"within **3–5 business days**. "
                        f"You'll receive a confirmation email shortly.\n\n"
                        f"Is there anything else I can help you with?\n\n"
                        f"— QuickShop Support Team"
                    )
                    state["tools_called"] = [{"tool": "process_refund", "args": {"order_id": pending_order}}]
                else:
                    state["draft_response"] = (
                        f"I'm sorry, I wasn't able to process the refund. "
                        f"Reason: {result.get('error', 'Unknown error')}. "
                        f"Please contact our support team for assistance.\n\n"
                        f"— QuickShop Support Team"
                    )

            state["pending_action"]        = None
            state["pending_order_id"]      = None
            state["awaiting_confirmation"] = False
            return state

        elif _is_rejection(user_message):
            state["draft_response"] = (
                f"No problem at all! The refund for order {pending_order} has **not** been processed. "
                f"Your order remains unchanged.\n\n"
                f"Is there anything else I can help you with?\n\n"
                f"— QuickShop Support Team"
            )
            state["pending_action"]        = None
            state["pending_order_id"]      = None
            state["awaiting_confirmation"] = False
            return state

        else:
            state["draft_response"] = (
                f"I just want to make sure — do you want me to go ahead and process "
                f"the **full refund** for order **{pending_order}**?\n\n"
                f"Please reply **Yes** to confirm or **No** to cancel.\n\n"
                f"— QuickShop Support Team"
            )
            state["awaiting_confirmation"] = True
            return state


    # ==========================================================
    # ✅ NEW: FORCED REFUND FLOW (CRITICAL FIX)
    # ==========================================================
    if (
        classification.get("intent") == "refund_request"
        and classification.get("order_id")
    ):
        order_id = classification["order_id"]
        order_info = get_order_by_id.invoke({"order_id": order_id})

        if order_info.get("error"):
            state["draft_response"] = f"Sorry, I couldn't find order {order_id}."
            return state

        amount  = order_info.get("amount", 0)
        product = order_info.get("product_name", "your item")
        email   = order_info.get("email", "")

        # 🧠 Adaptive Memory Check
        profile = get_customer_incident_profile.invoke({"email": email})
        incident_count = profile.get("incident_count", 0)

        # If high incident count or highly frustrated customer, bypass confirmation!
        if incident_count >= 2 or state.get("frustration_score", 0.0) >= Config.ESCALATION_FRUSTRATION_THRESHOLD:
            result = process_refund.invoke({
                "order_id": order_id,
                "reason": f"System Auto-Bypass (Incidents: {incident_count}, Frustration: {state.get('frustration_score', 0.0):.2f})"
            })
            if result.get("success"):
                state["draft_response"] = (
                    f"🚨 **VIP Frictionless Service Applied** 🚨\n\n"
                    f"Dear {profile.get('customer_name', 'Customer')},\n"
                    f"I see that you have experienced multiple delivery issues in the past (total incidents: {incident_count}). "
                    f"We sincerely apologize for this repeated inconvenience.\n\n"
                    f"To make things right immediately, I have **bypassed our standard verification steps** and processed a full refund for you:\n\n"
                    f"• **Order:** {order_id.upper()}\n"
                    f"• **Refund ID:** {result['refund_id']}\n"
                    f"• **Refund Amount:** ₹{result['amount']:,.2f}\n\n"
                    f"The funds will reflect in your original payment method in **3-5 business days**.\n\n"
                    f"Is there anything else I can do to restore your trust?\n\n"
                    f"— QuickShop Support Team"
                )
                state["tools_called"] = [
                    {"tool": "get_customer_incident_profile", "args": {"email": email}},
                    {"tool": "process_refund", "args": {"order_id": order_id}}
                ]
            else:
                state["draft_response"] = (
                    f"I'm sorry, I wasn't able to process the refund. "
                    f"Reason: {result.get('error', 'Unknown error')}. "
                    f"Please contact our support team for assistance.\n\n"
                    f"— QuickShop Support Team"
                )
            state["pending_action"]        = None
            state["pending_order_id"]      = None
            state["awaiting_confirmation"] = False
            return state

        # Otherwise, standard confirmation flow
        state["draft_response"] = (
            f"I completely understand your frustration, and I sincerely apologize "
            f"for the delay with your order.\n\n"
            f"I can process a **full refund** for you:\n\n"
            f"• **Order:** {order_id.upper()}\n"
            f"• **Product:** {product}\n"
            f"• **Refund amount:** ₹{float(amount):,.2f}\n\n"
            f"Would you like me to go ahead? "
            f"Please reply **Yes** to confirm or **No** to cancel.\n\n"
            f"— QuickShop Support Team"
        )

        state["pending_action"]        = "refund"
        state["pending_order_id"]      = order_id.upper()
        state["awaiting_confirmation"] = True
        state["tools_called"]          = [{"tool": "get_order_by_id", "args": {"order_id": order_id}}]

        return state


    from datetime import datetime
    current_date_str = datetime.now().strftime("%B %d, %Y")

    # ── PRE-FLIGHT: Fetch real order data from DB before LLM runs ────────────
    # This injects the EXACT database values into the system prompt so the LLM
    # can NEVER hallucinate product names, amounts, or dates.
    order_id_in_context = classification.get("order_id")
    preflight_order_block = ""
    if order_id_in_context:
        preflight_data = get_order_by_id.invoke({"order_id": order_id_in_context})
        if not preflight_data.get("error"):
            preflight_order_block = f"""
⚠️  GROUND TRUTH — EXACT DATABASE VALUES FOR {order_id_in_context.upper()} ⚠️
You MUST use these exact values. Never guess or substitute any field:
  • order_id        : {preflight_data.get('order_id')}
  • product_name    : {preflight_data.get('product_name')}
  • amount          : ₹{preflight_data.get('amount')}
  • status          : {preflight_data.get('status')}
  • created_at      : {preflight_data.get('created_at')}
  • expected_delivery: {preflight_data.get('expected_delivery')}
  • delivered_at    : {preflight_data.get('delivered_at', 'Not yet delivered')}
  • customer_name   : {preflight_data.get('customer_name')}
  • customer_email  : {preflight_data.get('email')}
  • customer_tier   : {preflight_data.get('tier')}
  • tracking_number : {preflight_data.get('tracking_number')}
"""
        else:
            preflight_order_block = f"\nNote: Order {order_id_in_context} was not found in the database. Inform the customer politely.\n"

    # ── PRE-FLIGHT: Also look up any existing refund for this order ───────────
    # Prevents the LLM from ever guessing/constructing a refund ID.
    preflight_refund_block = ""
    if order_id_in_context:
        refund_data = get_refund_status.invoke({"order_id": order_id_in_context})
        if refund_data.get("refund_exists"):
            preflight_refund_block = f"""
⚠️  EXISTING REFUND RECORD FOR {order_id_in_context.upper()} ⚠️
A refund already exists. Use ONLY these exact values — NEVER construct or guess a refund ID:
  • refund_id       : {refund_data.get('refund_id')}
  • refund_amount   : ₹{refund_data.get('amount')}
  • refund_status   : {refund_data.get('status')}
  • refund_reason   : {refund_data.get('reason')}
  • refund_date     : {refund_data.get('created_at')}
"""
        else:
            preflight_refund_block = f"\n  • refund_status   : No refund on record for {order_id_in_context.upper()}.\n"

    # ── NORMAL FLOW ────────────────────────────────────────────────────────────
    system_prompt = f"""You are a helpful customer support agent for QuickShop India, an e-commerce platform.

Current Date: {current_date_str}

Customer context:
- Intent: {classification.get('intent', 'unknown')}
- Sentiment: {classification.get('sentiment', 'neutral')}
- Urgency: {classification.get('urgency', 'medium')}
- Order ID mentioned: {classification.get('order_id', 'None')}
{preflight_order_block}{preflight_refund_block}
CRITICAL RULES — follow these exactly:
1. If GROUND TRUTH data is provided above, use ONLY those values for the order — do NOT call get_order_by_id again for the same order; the data is already fetched.
2. If EXISTING REFUND RECORD is provided above, use ONLY those values for the refund — do NOT invent or construct a refund ID. The refund_id field MUST come verbatim from the EXISTING REFUND RECORD block.
3. If the order status is "delayed" or "shipped", call get_order_journey_details to retrieve warehouse, failure event, and transit delay reason.
4. NEVER hallucinate or guess any order fields (product name, amount, dates) or refund fields (refund_id, amount). Use ONLY the GROUND TRUTH and EXISTING REFUND RECORD blocks above.
5. Dates are in ISO format (e.g., "2026-06-05T12:00:00"). Convert to readable format like "June 5, 2026" keeping the exact day/month/year. NEVER change the year.
6. For refund or cancellation requests:
   - ASK FOR CONFIRMATION before calling process_refund
   - Use this exact format: "I can process a full refund of ₹[amount] for order [order_id] ([product_name]). Would you like me to go ahead? Please reply Yes to confirm or No to cancel."
   - DO NOT call process_refund until the customer confirms
7. For angry customers — acknowledge their frustration FIRST, then solve
8. Keep responses concise and clear
9. Sign off as "QuickShop Support Team"
10. If the customer has multiple previous incidents, mention them and use a highly apologetic tone.
"""

    api_key = SecretStr(Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None
    llm = ChatGroq(
        model=Config.get_model(),
        api_key=api_key,
        max_tokens=1000
    )

    llm_with_tools = llm.bind_tools(RESOLVER_TOOLS)

    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    for msg in state.get("conversation_history", []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_message))

    tool_registry = {t.name: t for t in RESOLVER_TOOLS}
    tools_called  = []

    response = AIMessage(content="")
    for _ in range(Config.MAX_RETRIES):
        res = llm_with_tools.invoke(messages)
        if not isinstance(res, AIMessage):
            raise TypeError("Expected AIMessage from LLM")
        response = res
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]

            if tool_name == "process_refund":
                order_id = tool_call["args"].get("order_id", "")

                order_info = get_order_by_id.invoke({"order_id": order_id})
                email = order_info.get("email", "")
                
                # Check incident history for bypass
                profile = get_customer_incident_profile.invoke({"email": email})
                incident_count = profile.get("incident_count", 0)

                if incident_count >= 2 or state.get("frustration_score", 0.0) >= Config.ESCALATION_FRUSTRATION_THRESHOLD:
                    result = process_refund.invoke({
                        "order_id": order_id,
                        "reason": f"System Auto-Bypass (Incidents: {incident_count}, Frustration: {state.get('frustration_score', 0.0):.2f})"
                    })
                    if result.get("success"):
                        state["draft_response"] = (
                            f"🚨 **VIP Frictionless Service Applied** 🚨\n\n"
                            f"Dear {profile.get('customer_name', 'Customer')},\n"
                            f"I see that you have experienced multiple delivery issues in the past (total incidents: {incident_count}). "
                            f"We sincerely apologize for this repeated inconvenience.\n\n"
                            f"To make things right immediately, I have **bypassed our standard verification steps** and processed a full refund for you:\n\n"
                            f"• **Order:** {order_id.upper()}\n"
                            f"• **Refund ID:** {result['refund_id']}\n"
                            f"• **Refund Amount:** ₹{result['amount']:,.2f}\n\n"
                            f"The funds will reflect in your original payment method in **3-5 business days**.\n\n"
                            f"Is there anything else I can do to restore your trust?\n\n"
                            f"— QuickShop Support Team"
                        )
                        state["tools_called"] = [
                            {"tool": "get_customer_incident_profile", "args": {"email": email}},
                            {"tool": "process_refund", "args": {"order_id": order_id}}
                        ]
                    else:
                        state["draft_response"] = (
                            f"I'm sorry, I wasn't able to process the refund. "
                            f"Reason: {result.get('error', 'Unknown error')}. "
                            f"Please contact our support team for assistance.\n\n"
                            f"— QuickShop Support Team"
                        )
                    state["pending_action"]        = None
                    state["pending_order_id"]      = None
                    state["awaiting_confirmation"] = False
                    return state

                # Otherwise, standard confirmation flow
                amount  = order_info.get("amount", "?")
                product = order_info.get("product_name", "your item")

                state["draft_response"] = (
                    f"I completely understand your frustration, and I sincerely apologize "
                    f"for the delay with your order.\n\n"
                    f"I can process a **full refund** for you:\n\n"
                    f"• **Order:** {order_id.upper()}\n"
                    f"• **Product:** {product}\n"
                    f"• **Refund amount:** ₹{float(amount):,.2f}\n\n"
                    f"Would you like me to go ahead? "
                    f"Please reply **Yes** to confirm or **No** to cancel.\n\n"
                    f"— QuickShop Support Team"
                )
                state["pending_action"]        = "refund"
                state["pending_order_id"]      = order_id.upper()
                state["awaiting_confirmation"] = True
                state["tools_called"]          = tools_called
                return state

            tool_result = (
                tool_registry[tool_name].invoke(tool_call["args"])
                if tool_name in tool_registry
                else {"error": f"Tool {tool_name} not found"}
            )
            tools_called.append({"tool": tool_name, "args": tool_call["args"]})
            messages.append(ToolMessage(
                content=json.dumps(tool_result, default=str),
                tool_call_id=tool_call["id"]
            ))

    state["draft_response"]   = response.content
    state["tools_called"]     = tools_called
    state["pending_action"]   = None
    state["pending_order_id"] = None
    return state