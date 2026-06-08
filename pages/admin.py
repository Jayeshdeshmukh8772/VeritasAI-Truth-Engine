"""
VeritasAI Admin Dashboard.
Password-gated analytics page showing query logs, model health, trust score trends,
and usage metrics. Access via /admin in Streamlit multipage navigation.
"""

import os
import sqlite3
import streamlit as st
import pandas as pd
import bcrypt
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="VeritasAI Admin",
    page_icon="🔒",
    layout="wide",
)

from styles import inject_styles, inject_admin_styles
inject_styles()
inject_admin_styles()


DB_PATH = "logs/admin.db"


def verify_admin_credentials() -> bool:
    """
    Verify admin password against bcrypt hash stored in environment or st.secrets.

    Returns:
        True if authenticated (or already authenticated this session), False otherwise
    """
    if st.session_state.get("admin_authenticated"):
        return True

    st.markdown(
        """
        <div style='max-width:400px; margin:4rem auto; text-align:center;'>
            <h2>🔒 Admin Dashboard</h2>
            <p style='color:#888;'>Enter the admin password to access system analytics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 2, 1])
    with col:
        password_input = st.text_input(
            "Admin Password", type="password", key="admin_pw_input"
        )
        if st.button("Unlock Dashboard", use_container_width=True, type="primary"):
            # Try st.secrets first (Streamlit Cloud), then env var, then default hash
            stored_hash: bytes
            try:
                stored_hash = st.secrets["ADMIN_PASSWORD_HASH"].encode("utf-8")
            except Exception:
                env_hash = os.getenv("ADMIN_PASSWORD_HASH", "")
                if env_hash:
                    stored_hash = env_hash.encode("utf-8")
                else:
                    # Default fallback for local dev: password is "admin123"
                    stored_hash = b"$2b$12$P8StVgtXwPLSWuYx9j7pGu96QMh5jaWwh4rGzuQGleappH/38QGEi"

            try:
                if bcrypt.checkpw(password_input.encode("utf-8"), stored_hash):
                    st.session_state.admin_authenticated = True
                    st.success("✅ Access granted.")
                    st.rerun()
                else:
                    st.error("❌ Invalid password.")
            except Exception as e:
                st.error(f"Authentication error: {e}")

    return False


def load_logs() -> pd.DataFrame:
    """
    Load the last 500 log entries from SQLite.

    Returns:
        DataFrame of log records, or empty DataFrame on error
    """
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT ts, level, event, model, latency_ms, trust_score,
                   consensus_ratio, error_type, component, hallucination_flagged, cache_hit
            FROM logs
            ORDER BY ts DESC
            LIMIT 500
            """,
            conn,
        )
        conn.close()
        return df
    except Exception as e:
        st.error(f"Failed to load logs: {e}")
        return pd.DataFrame()


def load_queries() -> pd.DataFrame:
    """Load query records from SQLite."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT ts, query_type, consensus_ratio, total_latency_ms,
                   models_used, models_trusted, models_flagged
            FROM queries
            ORDER BY ts DESC
            LIMIT 200
            """,
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_model_trust_history() -> pd.DataFrame:
    """Load per-model trust scores for the last 50 events."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT ts, model, trust_score
            FROM logs
            WHERE trust_score IS NOT NULL AND model IS NOT NULL
            ORDER BY ts DESC
            LIMIT 200
            """,
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_feedback() -> pd.DataFrame:
    """Load user feedback records."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT ts, session_id, vote, comment FROM feedback ORDER BY ts DESC LIMIT 100",
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ─── Main dashboard render ────────────────────────────────────────────────────

if not verify_admin_credentials():
    st.stop()

st.title("📊 VeritasAI — System Analytics Dashboard")
st.caption(f"Last refreshed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

if st.button("🔄 Refresh Data"):
    st.rerun()

st.write("---")

# Load all data
logs_df = load_logs()
queries_df = load_queries()
trust_history_df = load_model_trust_history()
feedback_df = load_feedback()

# ── Metric cards ──────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

total_queries = len(queries_df) if not queries_df.empty else 0
m1.metric("🔍 Total Queries", total_queries)

if not queries_df.empty and "consensus_ratio" in queries_df.columns:
    avg_confidence = queries_df["consensus_ratio"].dropna().mean()
    m2.metric("📊 Avg Confidence", f"{avg_confidence * 100:.1f}%")
else:
    m2.metric("📊 Avg Confidence", "—")

if not logs_df.empty and "cache_hit" in logs_df.columns:
    cache_hits = logs_df["cache_hit"].sum()
    cache_rate = cache_hits / max(len(logs_df), 1) * 100
    m3.metric("💾 Cache Hit Rate", f"{cache_rate:.1f}%")
else:
    m3.metric("💾 Cache Hit Rate", "—")

if not logs_df.empty:
    error_count = len(logs_df[logs_df["level"] == "ERROR"])
    error_rate = error_count / max(len(logs_df), 1) * 100
    m4.metric("❌ Error Rate", f"{error_rate:.1f}%", delta=f"{error_count} errors")
else:
    m4.metric("❌ Error Rate", "—")

st.write("---")

# ── Trust score chart ─────────────────────────────────────────────────────────
st.markdown("### 📈 Per-Model Trust Score Over Time")
if not trust_history_df.empty:
    pivot = trust_history_df.pivot_table(
        index="ts", columns="model", values="trust_score"
    ).reset_index()
    st.line_chart(pivot.set_index("ts"), use_container_width=True, height=300)
else:
    st.caption("No trust score data yet — run some queries to populate this chart.")

st.write("---")

# ── Query history table ───────────────────────────────────────────────────────
st.markdown("### 📋 Query History")
if not queries_df.empty:
    st.dataframe(queries_df, use_container_width=True, hide_index=True)
else:
    st.caption("No queries logged yet.")

st.write("---")

# ── Filterable log table ──────────────────────────────────────────────────────
st.markdown("### 📜 Event Log")
if not logs_df.empty:
    # Filter controls
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        level_filter = st.multiselect(
            "Level", options=logs_df["level"].unique().tolist(), default=[]
        )
    with filter_col2:
        event_filter = st.text_input("Filter by event name", "")
    with filter_col3:
        model_filter = st.text_input("Filter by model", "")

    filtered = logs_df.copy()
    if level_filter:
        filtered = filtered[filtered["level"].isin(level_filter)]
    if event_filter:
        filtered = filtered[filtered["event"].str.contains(event_filter, case=False, na=False)]
    if model_filter:
        filtered = filtered[filtered["model"].str.contains(model_filter, case=False, na=False)]

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(filtered)} of {len(logs_df)} log entries")
else:
    st.caption("No log data yet.")

st.write("---")

# ── Feedback ──────────────────────────────────────────────────────────────────
st.markdown("### 👍 User Feedback")
if not feedback_df.empty:
    up_count = len(feedback_df[feedback_df["vote"] == "up"])
    down_count = len(feedback_df[feedback_df["vote"] == "down"])
    total_fb = len(feedback_df)
    f1, f2, f3 = st.columns(3)
    f1.metric("Total Feedback", total_fb)
    f2.metric("👍 Positive", up_count)
    f3.metric("👎 Negative", down_count)
    st.dataframe(feedback_df, use_container_width=True, hide_index=True)
else:
    st.caption("No feedback recorded yet.")

st.write("---")

# Logout button
if st.button("🔒 Logout"):
    st.session_state.admin_authenticated = False
    st.rerun()