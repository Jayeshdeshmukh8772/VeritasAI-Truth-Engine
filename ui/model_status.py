"""
VeritasAI Asynchronous Node Process Tracker Bar.
Renders status indicators for each active LLM tracking stream.
"""

import streamlit as st
from core.result import LLMResult, LLMStatus

class ModelStatusComponent:
    @staticmethod
    def render_tracker_row(active_results: list[LLMResult]):
        """Generates dynamic indicator badges for active worker processes."""
        st.write("---")
        st.markdown("### Operational Dispatch Tracker")
        
        if not active_results:
            st.caption("Initializing distributed dispatch worker matrices...")
            return

        cols = st.columns(len(active_results))
        for idx, res in enumerate(active_results):
            with cols[idx]:
                if res.status == LLMStatus.SUCCESS:
                    st.markdown(f"✅ **{res.model}**\n`{res.latency_ms}ms`", unsafe_allow_html=True)
                elif res.status == LLMStatus.FAILED:
                    st.markdown(f"❌ **{res.model}**\n`{res.error_type or 'Error'}`", unsafe_allow_html=True)
                elif res.status == LLMStatus.SKIPPED:
                    st.markdown(f"⚪ **{res.model}**\n`Skipped`", unsafe_allow_html=True)