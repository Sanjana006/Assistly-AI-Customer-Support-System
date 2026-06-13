import streamlit as st
import sys, os

st.set_page_config(
    page_title="Assistly",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph.graph import process_ticket
from config import Config

# ── Auto-seed database on first run (Streamlit Cloud has no committed .db) ────
def _ensure_database():
    """Create and seed the SQLite database if it doesn't exist or is empty."""
    try:
        from sqlalchemy import create_engine, inspect, text
        engine = create_engine(Config.DATABASE_URL)
        inspector = inspect(engine)
        has_data = False
        if "customers" in inspector.get_table_names():
            with engine.connect() as conn:
                count = conn.execute(text("SELECT COUNT(*) FROM customers")).scalar()
                if count and count > 0:
                    has_data = True
        
        if not has_data:
            from data.seed_database import create_database
            create_database()
            
        # Ensure additional tables (refunds, replacements, inventory, order_events) exist
        from tools.refund_processor import _ensure_tables_exist
        _ensure_tables_exist()
        
        return {"success": True, "error": None}
    except Exception as e:
        import traceback
        return {"success": False, "error": f"{e}\n{traceback.format_exc()}"}

db_status = _ensure_database()


# ── Session State Init ────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("history", []),
    ("active_order_id", None),
    ("active_order_id_prev", None),
    ("pending_action", None),
    ("pending_order_id", None),
    ("pending_action_reason", None),
    ("is_streaming", False),
    ("current_streaming_text", ""),
    ("streaming_progress", ""),
    ("stream_result", None),
    ("last_user_prompt", ""),
    ("pending_user_input", None),   # Holds prompt between phase-1 and phase-2 reruns
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Load test scenarios ───────────────────────────────────────────────────────
def get_test_scenarios():
    from sqlalchemy import create_engine, text
    engine = create_engine(Config.DATABASE_URL)
    scenarios = {}
    try:
        with engine.connect() as conn:
            delayed = conn.execute(text("""
                SELECT o.order_id, o.product_name, c.email
                FROM orders o JOIN customers c ON o.customer_id = c.customer_id
                WHERE o.status = 'delayed' LIMIT 1
            """)).fetchone()
            if delayed:
                scenarios["delayed"] = {"order_id": delayed.order_id, "product": delayed.product_name, "email": delayed.email}

            vip = conn.execute(text("""
                SELECT c.email, c.name, COUNT(o.order_id) as incident_count,
                       (SELECT order_id FROM orders WHERE customer_id = c.customer_id LIMIT 1) as order_id
                FROM customers c JOIN orders o ON o.customer_id = c.customer_id
                WHERE o.status IN ('delayed','cancelled')
                GROUP BY c.customer_id HAVING incident_count >= 2 LIMIT 1
            """)).fetchone()
            if vip:
                scenarios["vip"] = {"email": vip.email, "name": vip.name, "incident_count": vip.incident_count, "order_id": vip.order_id}

            clean = conn.execute(text("""
                SELECT o.order_id, o.product_name, c.email, c.name
                FROM orders o JOIN customers c ON o.customer_id = c.customer_id
                WHERE c.customer_id NOT IN (
                    SELECT DISTINCT customer_id FROM orders WHERE status IN ('delayed','cancelled')
                ) LIMIT 1
            """)).fetchone()
            if clean:
                scenarios["clean"] = {"order_id": clean.order_id, "product": clean.product_name, "email": clean.email, "name": clean.name}

            # New: Replacement-Only
            repl_only = conn.execute(text("""
                SELECT o.order_id, o.product_name, c.email
                FROM orders o JOIN customers c ON o.customer_id = c.customer_id
                WHERE o.return_policy = 'replacement_only' LIMIT 1
            """)).fetchone()
            if repl_only:
                scenarios["replacement_only"] = {"order_id": repl_only.order_id, "product": repl_only.product_name, "email": repl_only.email}

            # New: Final Sale
            final_sale = conn.execute(text("""
                SELECT o.order_id, o.product_name, c.email
                FROM orders o JOIN customers c ON o.customer_id = c.customer_id
                WHERE o.return_policy = 'non_returnable' LIMIT 1
            """)).fetchone()
            if final_sale:
                scenarios["final_sale"] = {"order_id": final_sale.order_id, "product": final_sale.product_name, "email": final_sale.email}
    except Exception as e:
        print(f"Scenarios error: {e}")
    return scenarios

scenarios = get_test_scenarios()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── GLOBAL RESET ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #080C14 !important;
    color: #E2E8F0 !important;
}

/* Remove Streamlit header/footer chrome but keep sidebar toggle visible */
#MainMenu, footer { visibility: hidden !important; }
header {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    pointer-events: none !important;
}
/* Hide sidebar expand/collapse buttons to lock the sidebar open */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}
[data-testid="stDeployButton"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0F1520; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366F1; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #0B0F19 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F1F5F9 !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stMarkdown p { color: #94A3B8 !important; font-size:0.85rem !important; }

/* Sidebar toggle */
[data-testid="stSidebar"] [data-testid="stToggle"] label span {
    color: #E2E8F0 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
[data-testid="stSidebar"] .stToggle > label { color: #E2E8F0 !important; }

/* Sidebar success/warning alerts */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    border-radius: 10px !important;
}
.stAlert[data-baseweb="notification"] { border-radius: 10px !important; }

/* Divider */
[data-testid="stSidebar"] hr {
    border-color: rgba(99,102,241,0.15) !important;
    margin: 1rem 0 !important;
}

/* ── MAIN CONTENT AREA ── */
[data-testid="stMain"] { padding: 0 !important; }
.block-container {
    padding: 2rem 2.5rem 5rem !important;
    max-width: 900px !important;
}

/* ── HEADER ── */
.assistly-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem 0 0.5rem;
    margin-bottom: 0.25rem;
}
.assistly-logo {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 0 24px rgba(99,102,241,0.4);
    flex-shrink: 0;
}
.assistly-title {
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #C7D2FE, #A5B4FC, #818CF8);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -0.02em;
}
.assistly-subtitle {
    font-size: 0.8rem;
    color: #64748B !important;
    font-weight: 500;
    margin-top: 0.1rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── PINNED ORDER CARD ── */
.pinned-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.05));
    border: 1px solid rgba(99,102,241,0.25);
    border-left: 4px solid #6366F1;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}
.pinned-card::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(99,102,241,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.pinned-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}
.pinned-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #A5B4FC !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.pinned-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 1rem;
}
.pinned-field label {
    display: block;
    font-size: 0.7rem;
    font-weight: 600;
    color: #64748B !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
}
.pinned-field span {
    font-size: 0.88rem;
    font-weight: 600;
    color: #E2E8F0 !important;
}

/* Status badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-delivered  { background: rgba(16,185,129,0.15);  color: #34D399 !important; border: 1px solid rgba(16,185,129,0.3); }
.badge-shipped    { background: rgba(59,130,246,0.15);  color: #60A5FA !important; border: 1px solid rgba(59,130,246,0.3); }
.badge-processing { background: rgba(245,158,11,0.15);  color: #FCD34D !important; border: 1px solid rgba(245,158,11,0.3); }
.badge-delayed    { background: rgba(239,68,68,0.15);   color: #F87171 !important; border: 1px solid rgba(239,68,68,0.3); }
.badge-cancelled  { background: rgba(168,85,247,0.15);  color: #C084FC !important; border: 1px solid rgba(168,85,247,0.3); }
.badge-refundable { background: rgba(16,185,129,0.15);  color: #34D399 !important; border: 1px solid rgba(16,185,129,0.3); }
.badge-replacement { background: rgba(245,158,11,0.15); color: #FCD34D !important; border: 1px solid rgba(245,158,11,0.3); }
.badge-final      { background: rgba(239,68,68,0.15);   color: #F87171 !important; border: 1px solid rgba(239,68,68,0.3); }

/* VIP alert */
.vip-banner {
    margin-top: 1rem;
    padding: 0.6rem 1rem;
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 10px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #FCA5A5 !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
    margin-bottom: 0.5rem !important;
    box-shadow: none !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"][data-role="user"] {
    background: transparent !important;
}

/* Override all chat content wrappers */
[data-testid="stChatMessageContent"] {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 16px !important;
    padding: 1rem 1.25rem !important;
    backdrop-filter: blur(8px) !important;
}
[data-testid="stChatMessageContent"] * {
    color: #E2E8F0 !important;
}
[data-testid="stChatMessageContent"] p {
    color: #CBD5E1 !important;
    line-height: 1.7 !important;
}
[data-testid="stChatMessageContent"] strong { color: #F1F5F9 !important; }

/* ── METRICS ROW ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.5rem;
    margin-top: 0.75rem;
}
.metric-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 10px;
    padding: 0.5rem 0.6rem;
    text-align: center;
}
.metric-label {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #64748B !important;
    margin-bottom: 0.2rem;
}
.metric-value {
    font-size: 0.82rem;
    font-weight: 700;
    color: #A5B4FC !important;
}
.metric-value.escalated-yes { color: #F87171 !important; }
.metric-value.escalated-no  { color: #34D399 !important; }

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: rgba(15,23,42,0.9) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #475569 !important; }
[data-testid="stChatInputSubmitButton"] {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
}
[data-testid="stChatInputSubmitButton"]:disabled {
    background: rgba(99, 102, 241, 0.15) !important;
    color: rgba(255, 255, 255, 0.25) !important;
}

/* ── BUTTONS ── */
[data-testid="stButton"] button {
    background: rgba(99,102,241,0.1) !important;
    color: #A5B4FC !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] button:hover {
    background: rgba(99,102,241,0.2) !important;
    border-color: rgba(99,102,241,0.5) !important;
    color: #C7D2FE !important;
}

/* Sidebar success box */
[data-testid="stSidebar"] [data-testid="stAlert"][data-type="success"] {
    background: rgba(16,185,129,0.1) !important;
    border: 1px solid rgba(16,185,129,0.25) !important;
    border-radius: 10px !important;
    color: #6EE7B7 !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"][data-type="warning"] {
    background: rgba(245,158,11,0.1) !important;
    border: 1px solid rgba(245,158,11,0.25) !important;
    border-radius: 10px !important;
    color: #FDE68A !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] p { color: inherit !important; }

/* ── SCENARIO CARDS in sidebar ── */
.scenario-card {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
}
.scenario-card h4 {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    color: #A5B4FC !important;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.scenario-card p {
    font-size: 0.78rem !important;
    color: #94A3B8 !important;
    margin: 0.15rem 0 !important;
}
.scenario-card code {
    background: rgba(99,102,241,0.15) !important;
    color: #C7D2FE !important;
    border-radius: 4px;
    padding: 0.1rem 0.35rem;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] { color: #A5B4FC !important; }
[data-testid="stSpinner"] p { color: #94A3B8 !important; }

/* ── GENERAL TEXT OVERRIDES ── */
p, span, div, li { color: #CBD5E1; }
h1 { color: #F1F5F9 !important; }
h2 { color: #E2E8F0 !important; font-weight: 700 !important; }
h3 { color: #CBD5E1 !important; font-weight: 600 !important; }
code { background: rgba(99,102,241,0.15) !important; color: #C7D2FE !important; }

/* Empty chat placeholder */
.empty-chat {
    text-align: center;
    padding: 4rem 2rem;
    color: #334155;
}
.empty-chat .icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.4; }
.empty-chat p { font-size: 0.9rem; color: #475569 !important; }

/* Custom Stop Button Overlay */
#custom-stop-btn-wrapper {
    position: fixed;
    bottom: 38px; /* Aligns inside the chat input box */
    right: calc(50% - 450px + 30px); /* Centers relative to max-width 900px */
    z-index: 9999999;
}
#custom-stop-btn-wrapper button {
    background: #EF4444 !important; /* Red background */
    color: white !important;
    border: none !important;
    border-radius: 50% !important; /* Circle shape */
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    min-height: 32px !important;
    max-width: 32px !important;
    max-height: 32px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 0.75rem !important;
    font-weight: bold !important;
    box-shadow: 0 0 12px rgba(239, 68, 68, 0.5) !important;
    cursor: pointer !important;
    transition: transform 0.1s ease, background-color 0.2s !important;
}
#custom-stop-btn-wrapper button:hover {
    background: #DC2626 !important;
    transform: scale(1.05);
}
#custom-stop-btn-wrapper button:active {
    transform: scale(0.95);
}

@media (max-width: 950px) {
    #custom-stop-btn-wrapper {
        right: 48px; /* Aligns inside padding on responsive screens */
    }
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="assistly-header">
    <div class="assistly-logo">⚡</div>
    <div>
        <div class="assistly-title">Assistly Support AI</div>
        <div class="assistly-subtitle">Multi-Agent · LangGraph · Hybrid PEFT Architecture</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not db_status["success"]:
    st.error(f"⚠️ Database Setup Error:\n\n{db_status['error']}")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Engine")

    local_classifier = st.toggle("🧠 Local PEFT Classifier", value=Config.USE_LOCAL_CLASSIFIER, disabled=st.session_state.get("is_streaming", False))
    if local_classifier != Config.USE_LOCAL_CLASSIFIER:
        Config.USE_LOCAL_CLASSIFIER = local_classifier
        st.rerun()

    if local_classifier:
        weights_exist = (
            os.path.exists(Config.LOCAL_MODEL_PATH)
            and len(os.listdir(Config.LOCAL_MODEL_PATH)) > 0
        )
        if weights_exist:
            st.success("✅ LoRA weights found — Local PEFT active")
        else:
            st.warning("⚠️ No weights found. Run `scripts/finetune.py` first.\n\nFalling back to Cloud API.")
    else:
        st.success("🟢 Cloud · Groq Llama-3.1 (fast mode)")
        st.caption("Enable toggle only after running `scripts/finetune.py`")

    st.markdown("---")
    st.markdown("## 🏗️ Active Agents")
    st.markdown("""
- 🔍 **Classifier** — Intent & Sentiment
- 🛠️ **Resolver** — DB, KB & Refunds
- ✅ **QA Auditor** — Accuracy Score
- 🚨 **Escalation** — Human Handoff
""")

    if scenarios:
        st.markdown("---")
        st.markdown("## 🧪 Test Scenarios")

        if "delayed" in scenarios:
            s = scenarios["delayed"]
            st.markdown(f"""
<div class="scenario-card">
    <h4>📦 Proactive Delay Alert</h4>
    <p>Order ID: <code>{s['order_id']}</code></p>
    <p>Pin order, type <code>hello</code> to trigger apology or ask <em>"why is my order delayed?"</em></p>
</div>
""", unsafe_allow_html=True)

        if "vip" in scenarios:
            s = scenarios["vip"]
            st.markdown(f"""
<div class="scenario-card">
    <h4>🚨 VIP Instant Refund</h4>
    <p>Email: <code>{s['email']}</code></p>
    <p>Order: <code>{s['order_id']}</code></p>
    <p>{s['incident_count']} past incidents → auto-bypass confirmation</p>
</div>
""", unsafe_allow_html=True)

        if "clean" in scenarios:
            s = scenarios["clean"]
            st.markdown(f"""
<div class="scenario-card">
    <h4>✅ Standard Refund</h4>
    <p>Email: <code>{s['email']}</code></p>
    <p>Order: <code>{s['order_id']}</code></p>
    <p>0 incidents → Yes/No confirmation flow</p>
</div>
""", unsafe_allow_html=True)

        if "replacement_only" in scenarios:
            s = scenarios["replacement_only"]
            st.markdown(f"""
<div class="scenario-card">
    <h4>🔄 Replacement-Only Order</h4>
    <p>Order: <code>{s['order_id']}</code></p>
    <p>Item: <code>{s['product']}</code></p>
    <p>Electronics warranty policy → Replacement offered first; VIP gets supervisor escalation.</p>
</div>
""", unsafe_allow_html=True)

        if "final_sale" in scenarios:
            s = scenarios["final_sale"]
            st.markdown(f"""
<div class="scenario-card">
    <h4>❌ Final Sale / No Return</h4>
    <p>Order: <code>{s['order_id']}</code></p>
    <p>Item: <code>{s['product']}</code></p>
    <p>Clearance / Hygiene policy → standard customer rejected; VIP gets supervisor escalation.</p>
</div>
""", unsafe_allow_html=True)

# ── PINNED ORDER CONTEXT CARD ─────────────────────────────────────────────────
if st.session_state.active_order_id:
    from tools.order_lookup import get_order_by_id, get_customer_incident_profile
    order_details = get_order_by_id.invoke({"order_id": st.session_state.active_order_id})

    if "error" in order_details:
        st.session_state.active_order_id = None
    else:
        email = order_details.get("email", "")
        profile = get_customer_incident_profile.invoke({"email": email})
        incident_count = profile.get("incident_count", 0)
        status = order_details.get("status", "processing")
        badge_class = f"badge-{status}"

        policy = order_details.get("return_policy", "eligible")
        policy_label = {
            "eligible": "Refundable",
            "replacement_only": "Replacement-Only",
            "non_returnable": "Final Sale",
        }.get(policy, "Refundable")
        policy_class = {
            "eligible": "badge-refundable",
            "replacement_only": "badge-replacement",
            "non_returnable": "badge-final",
        }.get(policy, "badge-refundable")

        is_vip = incident_count >= 2

        delivery_date = order_details.get('expected_delivery', '')
        if delivery_date and 'T' in delivery_date:
            delivery_date = delivery_date.split('T')[0]

        st.markdown(f"""
<div class="pinned-card">
    <div class="pinned-header">
        <div class="pinned-title">📌 Pinned Order Context</div>
        <div style="display: flex; gap: 0.5rem;">
            <span class="badge {policy_class}">{policy_label}</span>
            <span class="badge {badge_class}">{status}</span>
        </div>
    </div>
    <div class="pinned-grid">
        <div class="pinned-field">
            <label>Order ID</label>
            <span>{order_details['order_id']}</span>
        </div>
        <div class="pinned-field">
            <label>Product</label>
            <span>{order_details['product_name']}</span>
        </div>
        <div class="pinned-field">
            <label>Customer</label>
            <span>{order_details['customer_name']}</span>
        </div>
        <div class="pinned-field">
            <label>Amount</label>
            <span>₹{order_details['amount']:,.2f}</span>
        </div>
        <div class="pinned-field">
            <label>Tier</label>
            <span>{order_details['tier'].upper()}</span>
        </div>
        <div class="pinned-field">
            <label>Tracking</label>
            <span>{order_details['tracking_number']}</span>
        </div>
        <div class="pinned-field">
            <label>Expected Delivery</label>
            <span>{delivery_date}</span>
        </div>
        <div class="pinned-field">
            <label>Email</label>
            <span>{order_details['email']}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
        if is_vip:
            st.markdown(f'<div class="vip-banner">🚨 VIP AUTO-BYPASS ACTIVE &mdash; {incident_count} past incidents. Confirmation friction disabled.</div>', unsafe_allow_html=True)

        if st.button("✕  Clear Pinned Context", key="unpin_context_btn", disabled=st.session_state.get("is_streaming", False)):
            st.session_state.active_order_id = None
            st.session_state.active_order_id_prev = None
            st.session_state.pending_action = None
            st.session_state.pending_order_id = None
            st.session_state.pending_action_reason = None
            st.rerun()

# ── HELPERS ───────────────────────────────────────────────────────────────────
def render_metrics(result: dict):
    needs_esc = result.get("needs_escalation", False)
    esc_class = "escalated-yes" if needs_esc else "escalated-no"
    esc_label = "Yes ⚠️" if needs_esc else "No"
    intent    = result.get("intent", "—")
    sentiment = result.get("sentiment", "—")
    qa        = result.get("quality_score", 0.0)
    latency   = result.get("processing_time_ms", 0.0)

    # Sentiment colouring
    sent_color = {"positive": "#34D399", "neutral": "#A5B4FC", "negative": "#F87171", "angry": "#EF4444"}.get(sentiment, "#A5B4FC")

    st.markdown(f"""
<div class="metrics-row">
    <div class="metric-card">
        <div class="metric-label">Intent</div>
        <div class="metric-value">{intent.replace("_"," ").title()}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Sentiment</div>
        <div class="metric-value" style="color:{sent_color} !important">{sentiment.title()}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">QA Score</div>
        <div class="metric-value">{qa:.2f}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Escalated</div>
        <div class="metric-value {esc_class}">{esc_label}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Latency</div>
        <div class="metric-value">{latency:.0f} ms</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── HELPER FOR BUTTON CLICK ───────────────────────────────────────────────────
def handle_button_click(choice_text: str):
    st.session_state.messages.append({"role": "user", "content": choice_text})
    st.session_state["last_user_prompt"] = choice_text

    result = process_ticket(
        user_message=choice_text,
        conversation_history=st.session_state.history,
        pending_action=st.session_state.pending_action,
        pending_order_id=st.session_state.pending_order_id,
        pending_action_reason=st.session_state.get("pending_action_reason"),
        active_order_id=st.session_state.active_order_id,
    )

    extracted_order_id = result.get("classification", {}).get("order_id")
    if extracted_order_id:
        st.session_state.active_order_id = extracted_order_id

    st.session_state.pending_action        = result.get("pending_action")
    st.session_state.pending_order_id      = result.get("pending_order_id")
    st.session_state.pending_action_reason = result.get("pending_action_reason")

    # Set up streaming state
    st.session_state["is_streaming"] = True
    st.session_state["current_streaming_text"] = result["final_response"]
    st.session_state["streaming_progress"] = ""
    st.session_state["stream_result"] = result

    if extracted_order_id and extracted_order_id != st.session_state.active_order_id_prev:
        st.session_state.active_order_id_prev = extracted_order_id

    st.rerun()

# ── HELPER FOR REGENERATION ───────────────────────────────────────────────────
def handle_regenerate():
    if st.session_state.messages:
        # Find the last user prompt and the prior history
        # Let's clean the last assistant response
        if st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages.pop()
        if st.session_state.history and st.session_state.history[-1]["role"] == "assistant":
            st.session_state.history.pop()

        last_prompt = st.session_state.get("last_user_prompt")
        # Fallback to last user message if last_user_prompt is empty
        if not last_prompt:
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "user":
                    last_prompt = msg["content"]
                    break

        if last_prompt:
            with st.spinner("Regenerating response..."):
                result = process_ticket(
                    user_message=last_prompt,
                    conversation_history=st.session_state.history,
                    pending_action=st.session_state.pending_action,
                    pending_order_id=st.session_state.pending_order_id,
                    pending_action_reason=st.session_state.get("pending_action_reason"),
                    active_order_id=st.session_state.active_order_id,
                )

            extracted_order_id = result.get("classification", {}).get("order_id")
            if extracted_order_id:
                st.session_state.active_order_id = extracted_order_id

            st.session_state.pending_action        = result.get("pending_action")
            st.session_state.pending_order_id      = result.get("pending_order_id")
            st.session_state.pending_action_reason = result.get("pending_action_reason")

            # Set up streaming state
            st.session_state["is_streaming"] = True
            st.session_state["current_streaming_text"] = result["final_response"]
            st.session_state["streaming_progress"] = ""
            st.session_state["stream_result"] = result

            if extracted_order_id and extracted_order_id != st.session_state.active_order_id_prev:
                st.session_state.active_order_id_prev = extracted_order_id

            st.rerun()

# ── CONVERSATION HISTORY ──────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
<div class="empty-chat">
    <div class="icon">💬</div>
    <p>Send a message to start a support conversation.</p>
    <p style="margin-top:0.5rem; font-size:0.78rem;">Try: <em>"Where is my order ORD00042?"</em> or <em>"I need a refund."</em></p>
</div>
""", unsafe_allow_html=True)
else:
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if msg["role"] == "assistant" and "metadata" in msg:
                render_metrics(msg["metadata"])

                # Show Regenerate button only under the last assistant message
                if idx == len(st.session_state.messages) - 1 and not st.session_state.get("is_streaming"):
                    st.markdown("""
<style>
.regenerate-container {
    margin-top: 0.6rem;
    display: flex;
    justify-content: flex-start;
}
.regenerate-container button {
    background: rgba(99,102,241,0.06) !important;
    border: 1px dashed rgba(99,102,241,0.25) !important;
    color: #A5B4FC !important;
    font-size: 0.72rem !important;
    padding: 0.25rem 0.6rem !important;
    border-radius: 8px !important;
    cursor: pointer !important;
}
.regenerate-container button:hover {
    background: rgba(99,102,241,0.12) !important;
    border-color: rgba(99,102,241,0.4) !important;
    color: #C7D2FE !important;
}
</style>
<div class="regenerate-container">
                    """, unsafe_allow_html=True)
                    if st.button("🔄 Regenerate Response", key="btn_regenerate"):
                        handle_regenerate()
                    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("is_streaming"):
        with st.chat_message("assistant"):
            placeholder = st.empty()

            # Perform typewriter streaming effect
            import time
            full_text = st.session_state["current_streaming_text"]
            words = full_text.split(" ")
            current_progress = ""
            for i, word in enumerate(words):
                current_progress += (word + " ")
                st.session_state["streaming_progress"] = current_progress
                placeholder.markdown(current_progress + "▌", unsafe_allow_html=True)
                time.sleep(0.04)

            # Finalize full stream completion
            st.session_state["is_streaming"] = False
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_text,
                "metadata": st.session_state["stream_result"],
            })
            st.session_state.history.extend([
                {"role": "user", "content": st.session_state["last_user_prompt"]},
                {"role": "assistant", "content": full_text}
            ])

            # Clear streaming state
            st.session_state["current_streaming_text"] = ""
            st.session_state["streaming_progress"] = ""
            st.rerun()

# ── INTERACTIVE OPTIONS ───────────────────────────────────────────────────────
if st.session_state.pending_action:
    st.markdown("<div style='margin-bottom: 0.6rem; font-size: 0.85rem; font-weight: 700; color: #A5B4FC;'>⚡ Quick Actions:</div>", unsafe_allow_html=True)
    if st.session_state.pending_action == "collect_reason":
        cols = st.columns(4)
        reasons = [
            ("⏰ Delayed Delivery", "Delayed Delivery"),
            ("⚠️ Damaged/Defective", "Damaged/Defective Product"),
            ("🛒 Ordered by Mistake", "Ordered by Mistake"),
            ("💰 Found Better Price", "Found Better Price"),
        ]
        for i, (label, val) in enumerate(reasons):
            with cols[i]:
                if st.button(label, key=f"btn_reason_{i}", use_container_width=True, disabled=st.session_state.get("is_streaming", False)):
                    handle_button_click(val)

    elif st.session_state.pending_action == "mitigate_delay":
        cols = st.columns(2)
        with cols[0]:
            if st.button("Wait for Delivery 🚚", key="btn_wait_delivery", use_container_width=True, disabled=st.session_state.get("is_streaming", False)):
                handle_button_click("Wait for Delivery 🚚")
        with cols[1]:
            if st.button("Request Refund 💸", key="btn_insist_refund_delay", use_container_width=True, disabled=st.session_state.get("is_streaming", False)):
                handle_button_click("Request Refund 💸")

    elif st.session_state.pending_action == "mitigate_defective":
        policy = "eligible"
        if st.session_state.active_order_id:
            try:
                from tools.order_lookup import get_order_by_id
                order_details = get_order_by_id.invoke({"order_id": st.session_state.active_order_id})
                policy = order_details.get("return_policy", "eligible")
            except Exception:
                pass
        
        if policy == "non_returnable":
            cols = st.columns(2)
            with cols[0]:
                if st.button("Accept ₹500 Coupon 🎁", key="btn_accept_coupon_def", use_container_width=True, disabled=st.session_state.get("is_streaming", False)):
                    handle_button_click("Yes")
            with cols[1]:
                if st.button("Decline Coupon ✕", key="btn_decline_coupon_def", use_container_width=True, disabled=st.session_state.get("is_streaming", False)):
                    handle_button_click("No")
        else:
            cols = st.columns(2)
            with cols[0]:
                if st.button("Accept Free Replacement 📦", key="btn_accept_rep", use_container_width=True, disabled=st.session_state.get("is_streaming", False)):
                    handle_button_click("Accept Free Replacement 📦")
            with cols[1]:
                if st.button("Insist on Cash Refund 💸", key="btn_insist_refund_def", use_container_width=True, disabled=st.session_state.get("is_streaming", False)):
                    handle_button_click("Insist on Cash Refund 💸")

    elif st.session_state.pending_action in ["refund", "replacement"]:
        cols = st.columns(2)
        act_label = "refund" if st.session_state.pending_action == "refund" else "replacement"
        with cols[0]:
            if st.button(f"Yes, confirm {act_label} ✅", key="btn_confirm_action", use_container_width=True, disabled=st.session_state.get("is_streaming", False)):
                handle_button_click("Yes")
        with cols[1]:
            if st.button("No, cancel request ✕", key="btn_cancel_action", use_container_width=True, disabled=st.session_state.get("is_streaming", False)):
                handle_button_click("No")

# ── CHAT INPUT ────────────────────────────────────────────────────────────────
# Phase 2: Process the pending input (runs on the rerun AFTER user bubble is shown)
if st.session_state.get("pending_user_input"):
    prompt = st.session_state["pending_user_input"]
    st.session_state["pending_user_input"] = None   # consume immediately

    with st.spinner("Agents working…"):
        result = process_ticket(
            user_message=prompt,
            conversation_history=st.session_state.history,
            pending_action=st.session_state.pending_action,
            pending_order_id=st.session_state.pending_order_id,
            pending_action_reason=st.session_state.get("pending_action_reason"),
            active_order_id=st.session_state.active_order_id,
        )

    extracted_order_id = result.get("classification", {}).get("order_id")
    if extracted_order_id:
        st.session_state.active_order_id = extracted_order_id

    st.session_state.pending_action        = result.get("pending_action")
    st.session_state.pending_order_id      = result.get("pending_order_id")
    st.session_state.pending_action_reason = result.get("pending_action_reason")

    # Set up streaming state
    st.session_state["is_streaming"] = True
    st.session_state["current_streaming_text"] = result["final_response"]
    st.session_state["streaming_progress"] = ""
    st.session_state["stream_result"] = result

    if extracted_order_id and extracted_order_id != st.session_state.active_order_id_prev:
        st.session_state.active_order_id_prev = extracted_order_id

    st.rerun()

# Phase 1: User submits message → show it immediately, queue processing for next rerun
if prompt := st.chat_input("Ask anything about your order, refund, or account...", disabled=st.session_state.get("is_streaming", False)):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state["last_user_prompt"] = prompt
    st.session_state["pending_user_input"] = prompt   # triggers Phase 2 on next rerun
    st.rerun()   # ← immediately rerender to show user bubble

# Overlay Stop button over the chat input submit button when active
if st.session_state.get("is_streaming"):
    st.markdown("<div id='custom-stop-btn-wrapper'>", unsafe_allow_html=True)
    stop_clicked = st.button("■", key="stop_generation_btn_overlay")
    st.markdown("</div>", unsafe_allow_html=True)

    if stop_clicked:
        st.session_state["is_streaming"] = False
        truncated_text = st.session_state.get("streaming_progress", "") + "\n\n*[Generation stopped by user]*"

        # Append to messages and history
        st.session_state.messages.append({
            "role": "assistant",
            "content": truncated_text,
            "metadata": st.session_state["stream_result"],
        })
        st.session_state.history.extend([
            {"role": "user", "content": st.session_state["last_user_prompt"]},
            {"role": "assistant", "content": truncated_text}
        ])

        # Clear streaming state
        st.session_state["current_streaming_text"] = ""
        st.session_state["streaming_progress"] = ""
        st.rerun()