import streamlit as st
import pandas as pd
import sqlite3
from config import Config

st.title("🗄️ Live Database Viewer")

# Extract the SQLite database path from DATABASE_URL
db_path = Config.DATABASE_URL
if db_path.startswith("sqlite:///"):
    db_path = db_path.replace("sqlite:///", "")
elif db_path.startswith("sqlite://"):
    db_path = db_path.replace("sqlite://", "")

tab1, tab2, tab3, tab4 = st.tabs(["Orders", "Refunds", "Replacements", "Audit Log"])

with sqlite3.connect(db_path) as conn:
    with tab1:
        df = pd.read_sql("SELECT * FROM orders ORDER BY created_at DESC", conn)
        st.dataframe(df)

    with tab2:
        df = pd.read_sql("SELECT * FROM refunds ORDER BY created_at DESC", conn)
        st.dataframe(df)

    with tab3:
        df = pd.read_sql("SELECT * FROM replacements ORDER BY created_at DESC", conn)
        st.dataframe(df)

    with tab4:
        df = pd.read_sql("SELECT * FROM order_events ORDER BY created_at DESC", conn)
        st.dataframe(df)