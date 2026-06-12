import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from config import Config

st.title("🗄️ Live Database Viewer")

engine = create_engine(Config.DATABASE_URL)

tab1, tab2, tab3, tab4 = st.tabs(["Orders", "Refunds", "Replacements", "Audit Log"])

with engine.connect() as conn:
    with tab1:
        df = pd.read_sql("SELECT * FROM orders ORDER BY created_at DESC", conn)  # type: ignore
        st.dataframe(df)

    with tab2:
        df = pd.read_sql("SELECT * FROM refunds ORDER BY created_at DESC", conn)  # type: ignore
        st.dataframe(df)

    with tab3:
        df = pd.read_sql("SELECT * FROM replacements ORDER BY created_at DESC", conn)  # type: ignore
        st.dataframe(df)

    with tab4:
        df = pd.read_sql("SELECT * FROM order_events ORDER BY created_at DESC", conn)  # type: ignore
        st.dataframe(df)