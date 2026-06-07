from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from pydantic import SecretStr
import json, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from tools.order_lookup import get_order_by_id, get_orders_by_customer_email
from tools.refund_processor import process_refund, get_refund_status
from tools.kb_search import search_knowledge_base

RESOLVER_TOOLS = [
    get_order_by_id,
    get_orders_by_customer_email,
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


    # ── NORMAL FLOW (UNCHANGED) ─────────────────────────────────
    system_prompt = f"""You are a helpful customer support agent for QuickShop India, an e-commerce platform.

Customer context:
- Intent: {classification.get('intent', 'unknown')}
- Sentiment: {classification.get('sentiment', 'neutral')}
- Urgency: {classification.get('urgency', 'medium')}
- Order ID mentioned: {classification.get('order_id', 'None')}

CRITICAL RULES — follow these exactly:
1. ALWAYS fetch order data with get_order_by_id before mentioning any order details
2. For refund or cancellation requests:
   - First fetch the order to confirm it exists
   - Then ASK FOR CONFIRMATION before calling process_refund
   - Use this exact format to ask:
     "I can process a full refund of ₹[amount] for order [order_id] ([product_name]).
      Would you like me to go ahead? Please reply Yes to confirm or No to cancel."
   - DO NOT call process_refund until the customer confirms
3. For angry customers — acknowledge their frustration FIRST, then solve
4. Never make up order details — only use data from tools
5. Keep responses concise and clear
6. Sign off as "QuickShop Support Team"
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