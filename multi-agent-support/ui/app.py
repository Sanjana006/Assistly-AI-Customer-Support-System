import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.orchestrator import process_ticket
from config import Config

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "active_order_id" not in st.session_state:
    st.session_state.active_order_id = None
if "active_order_id_prev" not in st.session_state:
    st.session_state.active_order_id_prev = None
if "pending_action" not in st.session_state:
    st.session_state.pending_action = None
if "pending_order_id" not in st.session_state:
    st.session_state.pending_order_id = None

st.set_page_config(
    page_title="QuickShop Support AI", 
    page_icon="🤖", 
    layout="wide"
)

# Load dynamic test scenarios from database for easy verification
@st.cache_data
def get_test_scenarios():
    from sqlalchemy import create_engine, text
    engine = create_engine(Config.DATABASE_URL)
    scenarios = {}
    try:
        with engine.connect() as conn:
            # 1. Delayed order scenario
            delayed = conn.execute(text("""
                SELECT o.order_id, o.product_name, c.email
                FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                WHERE o.status = 'delayed'
                LIMIT 1
            """)).fetchone()
            if delayed:
                scenarios["delayed"] = {"order_id": delayed.order_id, "product": delayed.product_name, "email": delayed.email}
                
            # 2. VIP high-incident scenario (>= 2 delays/cancellations)
            vip = conn.execute(text("""
                SELECT c.email, c.name, COUNT(o.order_id) as incident_count,
                       (SELECT order_id FROM orders WHERE customer_id = c.customer_id LIMIT 1) as order_id
                FROM customers c
                JOIN orders o ON o.customer_id = c.customer_id
                WHERE o.status IN ('delayed', 'cancelled')
                GROUP BY c.customer_id
                HAVING incident_count >= 2
                LIMIT 1
            """)).fetchone()
            if vip:
                scenarios["vip"] = {"email": vip.email, "name": vip.name, "incident_count": vip.incident_count, "order_id": vip.order_id}
                
            # 3. Clean scenario (0 delays/cancellations)
            clean = conn.execute(text("""
                SELECT o.order_id, o.product_name, c.email, c.name
                FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                WHERE c.customer_id NOT IN (
                    SELECT DISTINCT customer_id FROM orders WHERE status IN ('delayed', 'cancelled')
                )
                LIMIT 1
            """)).fetchone()
            if clean:
                scenarios["clean"] = {"order_id": clean.order_id, "product": clean.product_name, "email": clean.email, "name": clean.name}
    except Exception as e:
        print(f"Error loading test scenarios: {e}")
    return scenarios

scenarios = get_test_scenarios()

# Custom Professional CSS Injection (Clean White Theme & Cards)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], .main, .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #FAFCFF !important;
    color: #1E293B !important;
}

h1 {
    font-weight: 700 !important;
    color: #0F172A !important;
    font-size: 2.25rem !important;
    margin-bottom: 0.25rem !important;
}
.stCaption {
    color: #64748B !important;
    font-size: 0.95rem !important;
    margin-bottom: 1.5rem !important;
}

/* Pinned active order context banner */
.pinned-order-container {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-left: 5px solid #3B82F6 !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
    margin-bottom: 1.25rem !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.04) !important;
}

.pinned-order-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 700;
    font-size: 1.05rem;
    color: #0F172A;
    margin-bottom: 0.75rem;
}

.pinned-order-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.75rem;
    font-size: 0.85rem;
}

.grid-item {
    display: flex;
    flex-direction: column;
}

.grid-item-label {
    color: #64748B;
    font-weight: 500;
    margin-bottom: 0.15rem;
}

.grid-item-value {
    color: #1E293B;
    font-weight: 600;
}

/* Sidebar styling overrides */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}

/* Custom metrics block style */
.custom-metric-row {
    display: flex;
    justify-content: space-between;
    background-color: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.75rem !important;
    margin-top: 0.5rem !important;
    margin-bottom: 1rem !important;
}
.custom-metric-row * {
    color: #1E293B !important;
}
.custom-metric {
    text-align: center;
    flex: 1;
    border-right: 1px solid #E2E8F0;
}
.custom-metric:last-child {
    border-right: none;
}
.metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    color: #64748B !important;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 0.15rem;
}
.metric-value {
    font-size: 0.85rem;
    font-weight: 700;
    color: #0F172A !important;
}

/* Beautiful custom tags for status */
.status-badge {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 9999px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
}
.status-delivered { background-color: #DEF7EC !important; color: #03543F !important; }
.status-shipped { background-color: #E1EFFE !important; color: #1E429F !important; }
.status-processing { background-color: #FDF6B2 !important; color: #723B13 !important; }
.status-delayed { background-color: #FDE8E8 !important; color: #9B1C1C !important; }
.status-cancelled { background-color: #EDEBFE !important; color: #5521B5 !important; }

/* Customizing Chat bubbles & forcing dark text colors for readability */
[data-testid="stChatMessage"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin-bottom: 0.75rem !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03) !important;
}
[data-testid="stChatMessage"] * {
    color: #1E293B !important;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span, [data-testid="stChatMessage"] div {
    color: #1E293B !important;
}

/* Alert styles for VIP auto-refund */
.vip-alert-banner {
    background-color: #FFF5F5 !important;
    border: 1px solid #FEB2B2 !important;
    color: #9B1C1C !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.75rem !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    margin-top: 0.75rem !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Reset / Clear Button override */
div.stButton > button.unpin-button {
    background-color: #FFFFFF !important;
    color: #EF4444 !important;
    border: 1px solid #FCA5A5 !important;
    border-radius: 6px !important;
    padding: 0.25rem 0.75rem !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
}
div.stButton > button.unpin-button:hover {
    background-color: #FEF2F2 !important;
    border-color: #EF4444 !important;
}
</style>
""", unsafe_allow_html=True)

# Main Title Section
st.title("🤖 Multi-Agent Customer Support System")
st.caption("Active Order Copilot · Hybrid PEFT Architecture · LangGraph + ChatGroq")

# Sidebar Configuration and Scenarios
with st.sidebar:
    st.header("Model Engine Settings")
    
    # Toggle local PEFT classifier model
    local_classifier = st.toggle("Local PEFT Classifier", value=False)
    if local_classifier != Config.USE_LOCAL_CLASSIFIER:
        Config.USE_LOCAL_CLASSIFIER = local_classifier
        st.rerun()

    if local_classifier:
        import os
        weights_exist = (
            os.path.exists(Config.LOCAL_MODEL_PATH)
            and len(os.listdir(Config.LOCAL_MODEL_PATH)) > 0
        )
        if weights_exist:
            st.success("✅ Fine-tuned LoRA weights found. Local PEFT Classifier is active.")
        else:
            st.warning(
                "⚠️ No fine-tuned weights found at `" + Config.LOCAL_MODEL_PATH + "`.\n\n"
                "The system will **fall back to Cloud ChatGroq** automatically.\n\n"
                "Run `scripts/finetune.py` first to train and save your LoRA adapters."
            )
    else:
        st.success("🟢 Cloud ChatGroq Llama-3.1 API (active)")
        st.caption("Toggle ON only after running `scripts/finetune.py` to train your own classifier.")
        
    st.divider()
    
    st.header("System Architecture")
    st.markdown("""
    **Active Agents:**
    - 🔍 **Classifier** (Intent + Sentiment)
    - 🛠️ **Resolver** (DB + KB + Refunds)
    - ✅ **QA Auditor** (Accuracy Score)
    - 🚨 **Escalation** (Human Handoff)
    """)
    
    # Render scenarios dynamically for evaluation
    if scenarios:
        st.divider()
        st.header("Test Scenarios")
        st.markdown("Use these order numbers/emails to test different agent logic:")
        
        if "delayed" in scenarios:
            st.subheader("1. Proactive Alert & 'Why' Reason")
            st.markdown(f"- **Order ID:** `{scenarios['delayed']['order_id']}`")
            st.markdown("- **Action:** Pin this order and type **'hello'** to trigger the proactive apology. Or ask **'why is my order delayed?'** to test structured reasoning.")
            
        if "vip" in scenarios:
            st.subheader("2. VIP Instant Refund (Bypass)")
            st.markdown(f"- **Email:** `{scenarios['vip']['email']}`")
            st.markdown(f"- **Order ID:** `{scenarios['vip']['order_id']}`")
            st.markdown(f"- **Context:** Customer has `{scenarios['vip']['incident_count']}` past incidents. Refund triggers auto-bypass of Yes/No checks.")
            
        if "clean" in scenarios:
            st.subheader("3. Standard Refund (Confirmation)")
            st.markdown(f"- **Email:** `{scenarios['clean']['email']}`")
            st.markdown(f"- **Order ID:** `{scenarios['clean']['order_id']}`")
            st.markdown("- **Context:** Customer has 0 incidents. Asking for a refund will trigger standard Yes/No confirmation check.")

# Pinned Active Order Card Header
if st.session_state.active_order_id:
    from tools.order_lookup import get_order_by_id, get_customer_incident_profile
    order_details = get_order_by_id.invoke({"order_id": st.session_state.active_order_id})
    
    if "error" in order_details:
        # Invalid order ID, reset
        st.session_state.active_order_id = None
    else:
        email = order_details.get("email", "")
        profile = get_customer_incident_profile.invoke({"email": email})
        incident_count = profile.get("incident_count", 0)
        
        status = order_details.get("status", "processing")
        status_class = f"status-{status}"
        
        vip_banner = ""
        if incident_count >= 2:
            vip_banner = f"""
            <div class="vip-alert-banner">
                <span>🚨</span>
                <span>VIP AUTO-BYPASS ACTIVE: Customer experienced {incident_count} past incidents. Confirmation friction is disabled for this chat context.</span>
            </div>
            """
            
        st.markdown(f"""
        <div class="pinned-order-container">
            <div class="pinned-order-header">
                <span>📌 Pinned Order Context: {order_details['order_id']} ({order_details['product_name']})</span>
                <span class="status-badge {status_class}">{status}</span>
            </div>
            <div class="pinned-order-grid">
                <div class="grid-item">
                    <span class="grid-item-label">Customer Name</span>
                    <span class="grid-item-value">{order_details['customer_name']}</span>
                </div>
                <div class="grid-item">
                    <span class="grid-item-label">Amount</span>
                    <span class="grid-item-value">₹{order_details['amount']:,.2f}</span>
                </div>
                <div class="grid-item">
                    <span class="grid-item-label">Tier / Email</span>
                    <span class="grid-item-value">{order_details['tier'].upper()} / {order_details['email']}</span>
                </div>
                <div class="grid-item">
                    <span class="grid-item-label">Tracking Number</span>
                    <span class="grid-item-value">{order_details['tracking_number']}</span>
                </div>
                <div class="grid-item">
                    <span class="grid-item-label">Expected Delivery</span>
                    <span class="grid-item-value">{order_details['expected_delivery'].split('T')[0]}</span>
                </div>
            </div>
            {vip_banner}
        </div>
        """, unsafe_allow_html=True)
        
        # Clear/Unpin button
        col1, col2 = st.columns([1, 10])
        with col1:
            if st.button("❌ Clear Pinned Context", key="unpin_context_btn", type="secondary"):
                st.session_state.active_order_id = None
                st.session_state.active_order_id_prev = None
                st.session_state.pending_action = None
                st.session_state.pending_order_id = None
                st.rerun()

# Display Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg["role"] == "assistant" and "metadata" in msg:
            result = msg["metadata"]
            needs_esc = "Yes" if result.get("needs_escalation") else "No"
            esc_color = "color: #EF4444;" if result.get("needs_escalation") else ""
            metrics_html = f"""
            <div class="custom-metric-row">
                <div class="custom-metric">
                    <div class="metric-label">Intent</div>
                    <div class="metric-value">{result.get('intent', 'unknown')}</div>
                </div>
                <div class="custom-metric">
                    <div class="metric-label">Sentiment</div>
                    <div class="metric-value">{result.get('sentiment', 'neutral')}</div>
                </div>
                <div class="custom-metric">
                    <div class="metric-label">QA Score</div>
                    <div class="metric-value">{result.get('quality_score', 0.0):.2f}</div>
                </div>
                <div class="custom-metric">
                    <div class="metric-label">Escalated</div>
                    <div class="metric-value" style="{esc_color}">{needs_esc}</div>
                </div>
                <div class="custom-metric">
                    <div class="metric-label">Latency</div>
                    <div class="metric-value">{result.get('processing_time_ms', 0.0):.0f}ms</div>
                </div>
            </div>
            """
            st.markdown(metrics_html, unsafe_allow_html=True)

# User Chat Input
if prompt := st.chat_input("Type your support query here..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Agents working..."):
            result = process_ticket(
                user_message=prompt,
                conversation_history=st.session_state.history,
                pending_action=st.session_state.pending_action,
                pending_order_id=st.session_state.pending_order_id,
                active_order_id=st.session_state.active_order_id
            )
            
        # Extract order ID to auto-pin context if found
        extracted_order_id = result.get("classification", {}).get("order_id")
        if extracted_order_id:
            st.session_state.active_order_id = extracted_order_id
            
        # Update confirmation flow settings
        st.session_state.pending_action = result.get("pending_action")
        st.session_state.pending_order_id = result.get("pending_order_id")
        
        # Display response — use markdown so HTML in agent responses renders correctly
        st.markdown(result["final_response"], unsafe_allow_html=True)
        
        # Render metrics row
        needs_esc = "Yes" if result.get("needs_escalation") else "No"
        esc_color = "color: #EF4444;" if result.get("needs_escalation") else ""
        metrics_html = f"""
        <div class="custom-metric-row">
            <div class="custom-metric">
                <div class="metric-label">Intent</div>
                <div class="metric-value">{result.get('intent', 'unknown')}</div>
            </div>
            <div class="custom-metric">
                <div class="metric-label">Sentiment</div>
                <div class="metric-value">{result.get('sentiment', 'neutral')}</div>
            </div>
            <div class="custom-metric">
                <div class="metric-label">QA Score</div>
                <div class="metric-value">{result.get('quality_score', 0.0):.2f}</div>
            </div>
            <div class="custom-metric">
                <div class="metric-label">Escalated</div>
                <div class="metric-value" style="{esc_color}">{needs_esc}</div>
            </div>
            <div class="custom-metric">
                <div class="metric-label">Latency</div>
                <div class="metric-value">{result.get('processing_time_ms', 0.0):.0f}ms</div>
            </div>
        </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)
        
        # Append history and save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["final_response"],
            "metadata": result
        })
        st.session_state.history.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": result["final_response"]}
        ])
        
        # Force rerun if context is newly pinned
        if extracted_order_id and extracted_order_id != st.session_state.active_order_id_prev:
            st.session_state.active_order_id_prev = extracted_order_id
            st.rerun()