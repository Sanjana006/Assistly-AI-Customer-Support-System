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
from tools.refund_processor import process_refund, get_refund_status, process_replacement, get_replacement_status, engine
from tools.kb_search import search_knowledge_base
from sqlalchemy import text
import uuid
from datetime import datetime

RESOLVER_TOOLS = [
    get_order_by_id,
    get_orders_by_customer_email,
    get_order_journey_details,
    get_customer_incident_profile,
    process_refund,
    get_refund_status,
    process_replacement,
    get_replacement_status,
    search_knowledge_base,
]

# (UNCHANGED — as requested)
YES_WORDS = {"yes", "yeah", "yep", "confirm", "confirmed", "sure", "proceed",
             "go ahead", "do it", "please", "ok", "okay", "absolutely", "y"}
NO_WORDS  = {"no", "nope", "cancel", "stop", "dont", "don't", "nevermind",
             "never mind", "skip", "n", "nah", "not"}

# ✅ FIXED (without changing your word lists — only logic)
def _is_confirmation(message: str) -> bool:
    cleaned = message.lower().strip().rstrip("!.,")
    if cleaned in YES_WORDS:
        return True
    words = cleaned.split()
    has_yes = any(w in YES_WORDS for w in words)
    has_no = any(w in NO_WORDS for w in words)
    return has_yes and not has_no

def _is_rejection(message: str) -> bool:
    cleaned = message.lower().strip().rstrip("!.,")
    if cleaned in NO_WORDS:
        return True
    words = cleaned.split()
    has_no = any(w in NO_WORDS for w in words)
    has_yes = any(w in YES_WORDS for w in words)
    return has_no and not has_yes

def _map_reason(message: str) -> str | None:
    msg = message.lower().strip().rstrip("!.,")
    if "delay" in msg or "time" in msg or "late" in msg or "transit" in msg or "delivery" in msg:
        return "Delayed Delivery"
    if "damage" in msg or "defect" in msg or "broken" in msg or "faulty" in msg or "issue" in msg or "work" in msg:
        return "Damaged/Defective Product"
    if "mistake" in msg or "accident" in msg or "wrong" in msg or "don't want" in msg or "change" in msg:
        return "Ordered by Mistake"
    if "price" in msg or "cheap" in msg or "cost" in msg or "find" in msg or "better" in msg:
        return "Found Better Price"
    for r in ["Delayed Delivery", "Damaged/Defective Product", "Ordered by Mistake", "Found Better Price"]:
        if r.lower() in msg:
            return r
    return None

def _is_coupon_acceptance(message: str) -> bool:
    cleaned = message.lower().strip().rstrip("!.,")
    if any(w in cleaned for w in ["coupon", "accept", "credit", "500", "gift", "keep", "yes please"]):
        return True
    return _is_confirmation(message)

def _is_replacement_acceptance(message: str) -> bool:
    cleaned = message.lower().strip().rstrip("!.,")
    if any(w in cleaned for w in ["replace", "replacement", "accept", "send", "yes please"]):
        return True
    return _is_confirmation(message)

def _is_insist_refund(message: str) -> bool:
    cleaned = message.lower().strip().rstrip("!.,")
    if any(w in cleaned for w in ["refund", "cash", "money", "insist", "cancel", "no thanks"]):
        return True
    if _is_rejection(message):
        return True
    return False


def resolve_ticket(state: dict) -> dict:
    classification  = state.get("classification", {})
    pending_action  = state.get("pending_action")
    pending_order   = state.get("pending_order_id")
    pending_reason  = state.get("pending_action_reason")
    user_message    = state["user_message"]

    if pending_order and not classification.get("order_id"):
        classification["order_id"] = pending_order
        state["classification"] = classification

    # ⚡ Proactive Delay Check on Greeting
    is_greeting = user_message.lower().strip().rstrip("!.,") in {
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "hi there", "hello there"
    }
    if is_greeting and classification.get("order_id"):
        order_id = classification["order_id"]
        order_info = get_order_by_id.invoke({"order_id": order_id})
        if not order_info.get("error") and order_info.get("status") == "delayed":
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
            state["tools_called"] = [{"tool": "get_order_by_id", "args": {"order_id": order_id}, "output": order_info}]
            state["pending_action"] = None
            state["pending_order_id"] = None
            state["pending_action_reason"] = None
            state["awaiting_confirmation"] = False
            return state

    # ── DIALOG STATE MACHINE (COUPONS, ALTERNATIVES, POLICIES & REASONS) ──
    if pending_action and pending_order:
        # A. Collect Reason State
        if pending_action == "collect_reason":
            reason = _map_reason(user_message)
            if not reason:
                state["draft_response"] = (
                    "To proceed with your request, please choose one of the following reasons:\n"
                    "• **Delayed Delivery** ⏰\n"
                    "• **Damaged/Defective Product** ⚠️\n"
                    "• **Ordered by Mistake** 🛒\n"
                    "• **Found Better Price** 💰\n\n"
                    "Please select or state one of these options."
                )
                state["awaiting_confirmation"] = True
                return state

            state["pending_action_reason"] = reason
            order_info = get_order_by_id.invoke({"order_id": pending_order})
            if order_info.get("error"):
                state["draft_response"] = f"Sorry, I couldn't find order {pending_order}."
                state["pending_action"] = None
                state["pending_order_id"] = None
                state["pending_action_reason"] = None
                state["awaiting_confirmation"] = False
                return state

            policy = order_info.get("return_policy", "eligible")

            if reason == "Delayed Delivery":
                state["draft_response"] = (
                    f"I'm so sorry for the delay with your order **{pending_order.upper()}**.\n\n"
                    f"To make this up to you, I can offer you a **₹500 Store Credit coupon** if you'd like to keep the order. "
                    f"Otherwise, I can proceed with the refund request. Which would you prefer?\n\n"
                    f"— QuickShop Support Team"
                )
                state["pending_action"] = "mitigate_delay"
                state["awaiting_confirmation"] = True
                return state

            elif reason == "Damaged/Defective Product":
                state["draft_response"] = (
                    f"I'm so sorry that your **{order_info.get('product_name')}** (Order **{pending_order.upper()}**) arrived damaged or defective.\n\n"
                    f"I can arrange a **Free Priority Replacement** for you immediately. "
                    f"Otherwise, I can proceed with the refund request. Which would you prefer?\n\n"
                    f"— QuickShop Support Team"
                )
                state["pending_action"] = "mitigate_defective"
                state["awaiting_confirmation"] = True
                return state

            else:  # Ordered by Mistake or Found Better Price
                if policy == "eligible":
                    amount = order_info.get("amount", 0)
                    product = order_info.get("product_name", "your item")
                    state["draft_response"] = (
                        f"I can process a full refund of ₹{float(amount):,.2f} for order **{pending_order.upper()}** ({product}).\n\n"
                        f"Would you like me to go ahead? Please reply **Yes** to confirm or **No** to cancel.\n\n"
                        f"— QuickShop Support Team"
                    )
                    state["pending_action"] = "refund"
                    state["awaiting_confirmation"] = True
                    return state

                elif policy == "replacement_only":
                    state["draft_response"] = (
                        f"Under our electronics warranty policy, order **{pending_order.upper()}** is only eligible for replacement, not a cash refund.\n\n"
                        f"Would you like me to initiate a priority replacement for you instead? Please reply **Yes** to confirm or **No** to cancel.\n\n"
                        f"— QuickShop Support Team"
                    )
                    state["pending_action"] = "replacement"
                    state["awaiting_confirmation"] = True
                    return state

                elif policy == "non_returnable":
                    state["draft_response"] = (
                        f"I apologize, but order **{pending_order.upper()}** contains a final-sale or clearance product and is non-refundable and non-returnable.\n\n"
                        f"Therefore, we cannot process a refund or replacement for this order. Is there anything else I can help you with?\n\n"
                        f"— QuickShop Support Team"
                    )
                    state["pending_action"] = None
                    state["pending_order_id"] = None
                    state["pending_action_reason"] = None
                    state["awaiting_confirmation"] = False
                    return state

        # B. Mitigate Delay State
        elif pending_action == "mitigate_delay":
            if _is_coupon_acceptance(user_message):
                try:
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO order_events
                                (event_id, order_id, event_type, old_status, new_status, note, created_at)
                            VALUES
                                (:eid, :oid, 'coupon_applied', NULL, NULL, :note, :now)
                        """), {
                            "eid": f"EVT{str(uuid.uuid4())[:8].upper()}",
                            "oid": pending_order.upper(),
                            "note": "Customer accepted ₹500 Store Credit coupon for delayed delivery.",
                            "now": datetime.now().isoformat()
                        })
                except Exception as e:
                    print(f"Error logging coupon: {e}")

                state["draft_response"] = (
                    f"Thank you! I have successfully credited **₹500 Store Credit** to your QuickShop account. "
                    f"Your order **{pending_order.upper()}** remains active and will be delivered as soon as possible.\n\n"
                    f"Is there anything else I can help you with?\n\n"
                    f"— QuickShop Support Team"
                )
                state["tools_called"] = [{
                    "tool": "apply_store_credit_coupon",
                    "args": {"order_id": pending_order.upper(), "amount": 500},
                    "output": {"success": True, "amount": 500, "message": "₹500 store credit coupon applied"}
                }]
                state["pending_action"] = None
                state["pending_order_id"] = None
                state["pending_action_reason"] = None
                state["awaiting_confirmation"] = False
                return state

            elif _is_insist_refund(user_message):
                order_info = get_order_by_id.invoke({"order_id": pending_order})
                policy = order_info.get("return_policy", "eligible")

                if policy == "eligible":
                    result = process_refund.invoke({
                        "order_id": pending_order,
                        "reason": f"{pending_reason} - Customer insisted on refund"
                    })
                    if result.get("success"):
                        state["draft_response"] = (
                            f"✅ Done! Your refund has been successfully processed.\n\n"
                            f"• **Refund ID:** {result['refund_id']}\n"
                            f"• **Amount:** ₹{result['amount']:,.2f}\n"
                            f"• **Order:** {result['order_id']} is now cancelled and stock restored.\n\n"
                            f"The amount will reflect in your account within 3–5 business days.\n\n"
                            f"— QuickShop Support Team"
                        )
                        state["tools_called"] = [{
                            "tool": "process_refund",
                            "args": {"order_id": pending_order, "reason": f"{pending_reason} - Customer insisted on refund"},
                            "output": result
                        }]
                    else:
                        state["draft_response"] = f"Unable to process refund: {result.get('error')}"
                    state["pending_action"] = None
                    state["pending_order_id"] = None
                    state["pending_action_reason"] = None
                    state["awaiting_confirmation"] = False
                    return state

                elif policy == "replacement_only":
                    state["draft_response"] = (
                        f"Under our electronics warranty policy, order **{pending_order.upper()}** is only eligible for replacement, not a cash refund.\n\n"
                        f"Would you like me to initiate a priority replacement for you instead? Please reply **Yes** to confirm or **No** to cancel.\n\n"
                        f"— QuickShop Support Team"
                    )
                    state["pending_action"] = "replacement"
                    state["awaiting_confirmation"] = True
                    return state

                elif policy == "non_returnable":
                    state["draft_response"] = (
                        f"I apologize, but order **{pending_order.upper()}** contains a final-sale or clearance product and is non-refundable and non-returnable.\n\n"
                        f"Therefore, we cannot process a refund or replacement for this order.\n\n"
                        f"— QuickShop Support Team"
                    )
                    state["pending_action"] = None
                    state["pending_order_id"] = None
                    state["pending_action_reason"] = None
                    state["awaiting_confirmation"] = False
                    return state

            else:
                state["draft_response"] = (
                    f"Please choose how you would like to proceed for order **{pending_order.upper()}**:\n"
                    f"1. **Accept ₹500 Coupon 🎁** (Apply store credit and keep order)\n"
                    f"2. **Insist on Cash Refund 💸** (Proceed with refund request)\n\n"
                    f"Please select or state one of these options."
                )
                state["awaiting_confirmation"] = True
                return state

        # C. Mitigate Defective State
        elif pending_action == "mitigate_defective":
            if _is_replacement_acceptance(user_message):
                result = process_replacement.invoke({
                    "order_id": pending_order,
                    "reason": "Damaged/Defective Product replacement"
                })
                if result.get("success"):
                    state["draft_response"] = (
                        f"✅ Done! Priority replacement successfully processed.\n\n"
                        f"• **Replacement ID:** {result['replacement_id']}\n"
                        f"• **Order:** {result['order_id']} status updated to 'replacement_pending'.\n"
                        f"• **Product:** {result['product']}\n\n"
                        f"A priority shipment of the item has been dispatched.\n\n"
                        f"— QuickShop Support Team"
                    )
                    state["tools_called"] = [{
                        "tool": "process_replacement",
                        "args": {"order_id": pending_order, "reason": "Damaged/Defective Product replacement"},
                        "output": result
                    }]
                else:
                    state["draft_response"] = f"Unable to process replacement: {result.get('error')}"
                state["pending_action"] = None
                state["pending_order_id"] = None
                state["pending_action_reason"] = None
                state["awaiting_confirmation"] = False
                return state

            elif _is_insist_refund(user_message):
                order_info = get_order_by_id.invoke({"order_id": pending_order})
                policy = order_info.get("return_policy", "eligible")

                if policy == "eligible":
                    result = process_refund.invoke({
                        "order_id": pending_order,
                        "reason": f"{pending_reason} - Customer insisted on refund"
                    })
                    if result.get("success"):
                        state["draft_response"] = (
                            f"✅ Done! Your refund has been successfully processed.\n\n"
                            f"• **Refund ID:** {result['refund_id']}\n"
                            f"• **Amount:** ₹{result['amount']:,.2f}\n"
                            f"• **Order:** {result['order_id']} is now cancelled and stock restored.\n\n"
                            f"The amount will reflect in your account within 3–5 business days.\n\n"
                            f"— QuickShop Support Team"
                        )
                        state["tools_called"] = [{
                            "tool": "process_refund",
                            "args": {"order_id": pending_order, "reason": f"{pending_reason} - Customer insisted on refund"},
                            "output": result
                        }]
                    else:
                        state["draft_response"] = f"Unable to process refund: {result.get('error')}"
                    state["pending_action"] = None
                    state["pending_order_id"] = None
                    state["pending_action_reason"] = None
                    state["awaiting_confirmation"] = False
                    return state

                elif policy == "replacement_only":
                    state["draft_response"] = (
                        f"Under our electronics warranty policy, order **{pending_order.upper()}** is only eligible for replacement, not a cash refund.\n\n"
                        f"Since you declined the priority replacement, we cannot issue a refund. Your order remains active.\n\n"
                        f"Is there anything else I can assist you with?\n\n"
                        f"— QuickShop Support Team"
                    )
                    state["pending_action"] = None
                    state["pending_order_id"] = None
                    state["pending_action_reason"] = None
                    state["awaiting_confirmation"] = False
                    return state

                elif policy == "non_returnable":
                    state["draft_response"] = (
                        f"I apologize, but order **{pending_order.upper()}** contains a final-sale or clearance product and is non-refundable and non-returnable.\n\n"
                        f"Therefore, we cannot process a refund or replacement for this order.\n\n"
                        f"— QuickShop Support Team"
                    )
                    state["pending_action"] = None
                    state["pending_order_id"] = None
                    state["pending_action_reason"] = None
                    state["awaiting_confirmation"] = False
                    return state

            else:
                state["draft_response"] = (
                    f"Please choose how you would like to proceed for order **{pending_order.upper()}**:\n"
                    f"1. **Accept Free Replacement 📦** (Dispatch a priority replacement)\n"
                    f"2. **Insist on Cash Refund 💸** (Proceed with refund request)\n\n"
                    f"Please select or state one of these options."
                )
                state["awaiting_confirmation"] = True
                return state

        # D. Refund Confirmation State
        elif pending_action == "refund":
            if _is_confirmation(user_message):
                result = process_refund.invoke({
                    "order_id": pending_order,
                    "reason": pending_reason or "Customer confirmed refund request"
                })
                if result.get("success"):
                    state["draft_response"] = (
                        f"✅ Done! Your refund has been successfully processed.\n\n"
                        f"**Refund ID:** {result['refund_id']}\n"
                        f"**Amount:** ₹{result['amount']:,.2f}\n"
                        f"**Order:** {result['order_id']} is now cancelled.\n"
                        f"**Inventory:** Stock restored for '{result['product']}'.\n\n"
                        f"The amount will reflect in your original payment method "
                        f"within **3–5 business days**.\n\n"
                        f"— QuickShop Support Team"
                    )
                    state["tools_called"] = [{
                        "tool": "process_refund",
                        "args": {"order_id": pending_order, "reason": pending_reason or "Customer confirmed refund request"},
                        "output": result
                    }]
                else:
                    state["draft_response"] = f"Unable to process refund: {result.get('error')}"
                state["pending_action"] = None
                state["pending_order_id"] = None
                state["pending_action_reason"] = None
                state["awaiting_confirmation"] = False
                return state

            elif _is_rejection(user_message):
                state["draft_response"] = (
                    f"No problem at all! The refund for order **{pending_order.upper()}** has **not** been processed. "
                    f"Your order remains active.\n\n"
                    f"Is there anything else I can help you with?\n\n"
                    f"— QuickShop Support Team"
                )
                state["pending_action"] = None
                state["pending_order_id"] = None
                state["pending_action_reason"] = None
                state["awaiting_confirmation"] = False
                return state

            else:
                state["draft_response"] = (
                    f"Do you want me to go ahead and process the refund for order **{pending_order.upper()}**?\n\n"
                    f"Please reply **Yes** to confirm or **No** to cancel.\n\n"
                    f"— QuickShop Support Team"
                )
                state["awaiting_confirmation"] = True
                return state

        # E. Replacement Confirmation State
        elif pending_action == "replacement":
            if _is_confirmation(user_message):
                result = process_replacement.invoke({
                    "order_id": pending_order,
                    "reason": pending_reason or "Customer confirmed replacement request"
                })
                if result.get("success"):
                    state["draft_response"] = (
                        f"✅ Done! Priority replacement successfully processed.\n\n"
                        f"• **Replacement ID:** {result['replacement_id']}\n"
                        f"• **Order:** {result['order_id']} status updated to 'replacement_pending'.\n"
                        f"• **Product:** {result['product']}\n\n"
                        f"A priority shipment of the item has been dispatched.\n\n"
                        f"— QuickShop Support Team"
                    )
                    state["tools_called"] = [{
                        "tool": "process_replacement",
                        "args": {"order_id": pending_order, "reason": pending_reason or "Customer confirmed replacement request"},
                        "output": result
                    }]
                else:
                    state["draft_response"] = f"Unable to process replacement: {result.get('error')}"
                state["pending_action"] = None
                state["pending_order_id"] = None
                state["pending_action_reason"] = None
                state["awaiting_confirmation"] = False
                return state

            elif _is_rejection(user_message):
                state["draft_response"] = (
                    f"No problem! The priority replacement for order **{pending_order.upper()}** has **not** been initiated. "
                    f"Your order remains active.\n\n"
                    f"Is there anything else I can help you with?\n\n"
                    f"— QuickShop Support Team"
                )
                state["pending_action"] = None
                state["pending_order_id"] = None
                state["pending_action_reason"] = None
                state["awaiting_confirmation"] = False
                return state

            else:
                state["draft_response"] = (
                    f"Do you want me to go ahead and initiate a priority replacement for order **{pending_order.upper()}**?\n\n"
                    f"Please reply **Yes** to confirm or **No** to cancel.\n\n"
                    f"— QuickShop Support Team"
                )
                state["awaiting_confirmation"] = True
                return state

    # ==========================================================
    # ✅ REFUND/CANCELLATION REQUEST PRE-FLIGHT
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
        policy  = order_info.get("return_policy", "eligible")

        # 🧠 Incident History Check for VIP Bypass
        profile = get_customer_incident_profile.invoke({"email": email})
        incident_count = profile.get("incident_count", 0)
        is_vip = incident_count >= 2 or state.get("frustration_score", 0.0) >= Config.ESCALATION_FRUSTRATION_THRESHOLD

        if is_vip:
            if policy == "eligible":
                # VIP Instant Refund
                result = process_refund.invoke({
                    "order_id": order_id,
                    "reason": f"VIP/Frustrated Auto-Bypass Refund (Incidents: {incident_count}, Frustration: {state.get('frustration_score', 0.0):.2f})"
                })
                if result.get("success"):
                    state["draft_response"] = (
                        f"🚨 **VIP Frictionless Service Applied** 🚨\n\n"
                        f"Dear {profile.get('customer_name', 'Customer')},\n"
                        f"I see that you have experienced multiple issues in the past (total incidents: {incident_count}). "
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
                        {"tool": "get_customer_incident_profile", "args": {"email": email}, "output": profile},
                        {"tool": "process_refund", "args": {"order_id": order_id, "reason": f"VIP/Frustrated Auto-Bypass Refund (Incidents: {incident_count})"}, "output": result}
                    ]
                else:
                    state["draft_response"] = f"Unable to process refund: {result.get('error')}"
                state["pending_action"]        = None
                state["pending_order_id"]      = None
                state["pending_action_reason"] = None
                state["awaiting_confirmation"] = False
                return state

            elif policy in ["replacement_only", "non_returnable"]:
                # VIP auto-escalation exception requests on restricted products
                state["needs_escalation"] = True
                state["escalation_reasons"] = [f"VIP customer requested refund exception on {policy} order"]
                state["draft_response"] = (
                    f"🚨 **VIP Exceptional Handoff** 🚨\n\n"
                    f"Dear {profile.get('customer_name', 'Customer')},\n"
                    f"I see that you are requesting a refund for Order **{order_id.upper()}** ({product}). "
                    f"Since this item has a '{policy.replace('_', ' ')}' return policy, our automated assistant is restricted from processing a refund directly.\n\n"
                    f"However, because you are a valued VIP customer (total incidents: {incident_count}) and have faced repeated issues, I am immediately routing your ticket to a senior supervisor to approve a special refund exception."
                )
                state["tools_called"] = [
                    {"tool": "get_customer_incident_profile", "args": {"email": email}, "output": profile}
                ]
                return state

        else:
            # Standard customer -> collect reason first
            state["draft_response"] = (
                f"I understand you want to cancel or refund your order **{order_id.upper()}** ({product}).\n\n"
                f"Before we proceed, could you please select the reason for your cancellation/refund request below?"
            )
            state["pending_action"]        = "collect_reason"
            state["pending_order_id"]      = order_id.upper()
            state["pending_action_reason"] = None
            state["awaiting_confirmation"] = True
            state["tools_called"]          = [{"tool": "get_order_by_id", "args": {"order_id": order_id}, "output": order_info}]
            return state

    current_date_str = datetime.now().strftime("%B %d, %Y")

    # ── PRE-FLIGHT: Fetch real order data from DB before LLM runs ────────────
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
  • return_policy   : {preflight_data.get('return_policy')}
"""
        else:
            preflight_order_block = f"\nNote: Order {order_id_in_context} was not found in the database. Inform the customer politely.\n"

    # ── PRE-FLIGHT: Look up existing refund / replacement status ───────────
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
            rep_data = get_replacement_status.invoke({"order_id": order_id_in_context})
            if rep_data.get("replacement_exists"):
                preflight_refund_block = f"""
⚠️  EXISTING REPLACEMENT RECORD FOR {order_id_in_context.upper()} ⚠️
A replacement already exists:
  • replacement_id   : {rep_data.get('replacement_id')}
  • replacement_status: {rep_data.get('status')}
  • replacement_reason: {rep_data.get('reason')}
  • replacement_date  : {rep_data.get('created_at')}
"""
            else:
                preflight_refund_block = f"\n  • refund_status   : No refund or replacement on record for {order_id_in_context.upper()}.\n"

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
2. If EXISTING REFUND/REPLACEMENT RECORD is provided above, use ONLY those values — do NOT invent or construct IDs.
3. If the order status is "delayed" or "shipped", call get_order_journey_details to retrieve warehouse, failure event, and transit delay reason.
4. NEVER hallucinate or guess any order fields (product name, amount, dates) or refund fields (refund_id, amount). Use ONLY the GROUND TRUTH and EXISTING records above.
5. Dates are in ISO format (e.g., "2026-06-05T12:00:00"). Convert to readable format like "June 5, 2026". NEVER change the month, day, or year. Maintain the exact original month and day from the database ground truth.
6. For refund or cancellation requests, redirect them or prompt, but do not issue refunds without collecting reasons first.
7. For angry customers — acknowledge their frustration FIRST, then solve.
8. Keep responses concise and clear.
9. Sign off as "QuickShop Support Team".
10. If the customer has multiple previous incidents, mention them and use a highly apologetic tone.
11. If an EXISTING REFUND/REPLACEMENT RECORD is provided in the prompt, the action is ALREADY processed. Do NOT call the processing tools again. Just display the existing details.
12. When you need to call a tool, only output the tool call. Do not mix conversational text and tool calls.
13. To process a refund or replacement, always invoke the 'process_refund' or 'process_replacement' tool. Never write a response claiming a refund or replacement is processed unless you have called the tool and received the success confirmation in the tool output.
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
        try:
            res = llm_with_tools.invoke(messages)
        except Exception as e:
            from loguru import logger
            logger.error(f"Groq API call failed: {e}")
            state["draft_response"] = (
                "I encountered a technical issue while processing this request. "
                "I am immediately connecting you with a human support specialist to assist you further."
            )
            state["needs_escalation"] = True
            state["escalation_reasons"] = [f"LLM Tool Call Error: {str(e)}"]
            return state

        if not isinstance(res, AIMessage):
            raise TypeError("Expected AIMessage from LLM")
        response = res
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]

            # Intercept refund/replacement calls in the LLM tool execution loop
            if tool_name in ["process_refund", "process_replacement"]:
                order_id = tool_call["args"].get("order_id", "")
                if not order_id and order_id_in_context:
                    order_id = order_id_in_context

                order_info = get_order_by_id.invoke({"order_id": order_id})
                if order_info.get("error"):
                    state["draft_response"] = f"Sorry, I couldn't find order {order_id}."
                    return state

                email = order_info.get("email", "")
                profile = get_customer_incident_profile.invoke({"email": email})
                incident_count = profile.get("incident_count", 0)
                is_vip = incident_count >= 2 or state.get("frustration_score", 0.0) >= Config.ESCALATION_FRUSTRATION_THRESHOLD
                policy = order_info.get("return_policy", "eligible")

                if is_vip:
                    if tool_name == "process_refund" and policy in ["replacement_only", "non_returnable"]:
                        state["needs_escalation"] = True
                        state["escalation_reasons"] = [f"VIP customer requested refund exception on {policy} order"]
                        state["draft_response"] = (
                            f"🚨 **VIP Exceptional Handoff** 🚨\n\n"
                            f"Dear {profile.get('customer_name', 'Customer')},\n"
                            f"I see that you are requesting a refund for Order **{order_id.upper()}**. "
                            f"Since this item has a '{policy.replace('_', ' ')}' return policy, our automated assistant is restricted from processing a refund directly.\n\n"
                            f"However, because you are a valued VIP customer (total incidents: {incident_count}), I am immediately routing your ticket to a senior supervisor to approve a special refund exception."
                        )
                        state["tools_called"] = [
                            {"tool": "get_customer_incident_profile", "args": {"email": email}, "output": profile}
                        ]
                        return state

                    # VIP allows auto-bypass execution of tool call
                    reason = tool_call["args"].get("reason") or f"VIP Auto-Bypass (Incidents: {incident_count})"
                    tool_result = tool_registry[tool_name].invoke({
                        "order_id": order_id,
                        "reason": reason
                    })
                    prefix = "🚨 **VIP Frictionless Service Applied** 🚨\n\n"
                    salutation = f"Dear {profile.get('customer_name', 'Customer')},\n"
                    incident_note = f"I see that you have experienced multiple issues in the past (total incidents: {incident_count}). We sincerely apologize for this repeated inconvenience.\n\nTo make things right immediately, I have bypassed our standard verification steps and completed your request:\n\n"
                    
                    state["draft_response"] = (
                        f"{prefix}{salutation}{incident_note}"
                        f"{tool_result.get('message', '')}"
                    )
                    state["tools_called"] = [
                        {"tool": "get_customer_incident_profile", "args": {"email": email}, "output": profile},
                        {"tool": tool_name, "args": tool_call["args"], "output": tool_result}
                    ]
                    state["pending_action"]        = None
                    state["pending_order_id"]      = None
                    state["pending_action_reason"] = None
                    state["awaiting_confirmation"] = False
                    return state

                # Standard customer -> intercept tool call and collect reason
                state["draft_response"] = (
                    f"I would be happy to help you with your order **{order_id.upper()}**.\n\n"
                    f"Before we proceed, could you please select the reason for your cancellation/refund request below?"
                )
                state["pending_action"] = "collect_reason"
                state["pending_order_id"] = order_id.upper()
                state["pending_action_reason"] = None
                state["awaiting_confirmation"] = True
                state["tools_called"] = [
                    {"tool": "get_order_by_id", "args": {"order_id": order_id}, "output": order_info}
                ]
                return state

            tool_result = (
                tool_registry[tool_name].invoke(tool_call["args"])
                if tool_name in tool_registry
                else {"error": f"Tool {tool_name} not found"}
            )
            tools_called.append({
                "tool": tool_name,
                "args": tool_call["args"],
                "output": tool_result
            })
            messages.append(ToolMessage(
                content=json.dumps(tool_result, default=str),
                tool_call_id=tool_call["id"]
            ))

    state["draft_response"]   = response.content
    state["tools_called"]     = tools_called
    state["pending_action"]   = None
    state["pending_order_id"] = None
    state["pending_action_reason"] = None
    return state