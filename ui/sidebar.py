"""
VeritasAI Sidebar Control Panel.
Query history, fast/deep mode toggle, model health status, and configuration controls.
"""

import streamlit as st
from typing import List


class SidebarComponent:
    """Renders the full sidebar with navigation, history, and settings."""

    @staticmethod
    def render_controls() -> None:
        """
        Render the complete sidebar UI with all controls and query history.
        Sets session state values that other components read.
        """
        with st.sidebar:
            # --- Branding ---
            st.markdown(
                "<h2 style='margin-bottom:0;'>⚖️ VeritasAI</h2>"
                "<p style='color:#888; font-size:12px; margin-top:2px;'>Multi-LLM Truth Engine v1.0</p>",
                unsafe_allow_html=True,
            )
            st.write("---")

            # --- Query Mode ---
            st.markdown("#### ⚙️ Query Mode")
            fast_mode = st.toggle(
                "⚡ Fast Mode (3 models, no peer review)",
                value=st.session_state.get("fast_mode", False),
                key="fast_mode",
                help="Fast: Groq + Gemini + Cerebras only. Deep: All 7 models + peer review.",
            )

            st.write("---")

            # --- Feature Toggles ---
            st.markdown("#### 🔧 Pipeline Features")
            st.toggle("🌐 Live Web Search", value=True, key="feat_search",
                      help="Query the live web for real-time grounding context (RAG)")
            st.toggle("✨ Query Enhancement", value=True, key="feat_enhancer",
                      help="Rewrite query for maximum clarity before dispatching")
            st.toggle("💾 Response Cache", value=True, key="feat_cache",
                      help="Cache identical queries to save API calls")
            st.toggle("🔬 Peer Review Stage", value=True, key="feat_peer",
                      help="Models anonymously critique each other's responses")
            st.toggle("🗣️ Debate Mode", value=False, key="debate_mode",
                      help="Show raw peer review critique text in model cards")

            st.write("---")

            # --- Weight Sliders ---
            st.markdown("#### ⚖️ Trust Score Weights")
            sem_weight = st.slider(
                "Semantic Weight (Math)", 0.0, 1.0, 0.6, step=0.05,
                key="sem_weight",
                help="Weight given to DBSCAN semantic clustering signal",
            )
            peer_weight = round(1.0 - sem_weight, 2)
            st.caption(f"Peer Review Weight auto-set to: **{peer_weight}**")

            st.write("---")

            # --- Model Health Status ---
            st.markdown("#### 🏥 Model Health")
            model_health = st.session_state.get("model_health", {})
            if model_health:
                for model_name, is_healthy in model_health.items():
                    icon = "🟢" if is_healthy else "🔴"
                    st.markdown(f"{icon} `{model_name}`")
            else:
                st.caption("Run a query to see model health status")

            st.write("---")

            # --- Query History ---
            st.markdown("#### 📜 Query History")
            history: List[str] = st.session_state.get("query_history", [])
            if not history:
                st.caption("No queries yet. Ask something!")
            else:
                for i, hist_query in enumerate(reversed(history[-20:])):
                    truncated = hist_query[:50] + "..." if len(hist_query) > 50 else hist_query
                    if st.button(
                        f"🔁 {truncated}",
                        key=f"hist_{i}",
                        use_container_width=True,
                    ):
                        st.session_state["reload_query"] = hist_query
                        st.rerun()

            st.write("---")

            # --- Session Info ---
            sess_id = st.session_state.get("session_id", "unknown")
            st.caption(f"Session: `{sess_id[:12]}...`")
            st.markdown("🔒 [Admin Dashboard](/admin)")
            st.markdown("[📘 GitHub](https://github.com/veritas-ai) | [📖 Docs](#)")

    @staticmethod
    def add_to_history(query: str) -> None:
        """
        Add a query to the session history list (max 20 entries).

        Args:
            query: The query string to record in history
        """
        history: List[str] = st.session_state.get("query_history", [])
        # Avoid duplicates
        if query not in history:
            history.append(query)
        if len(history) > 20:
            history = history[-20:]
        st.session_state["query_history"] = history