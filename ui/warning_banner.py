"""
VeritasAI Pipeline Safety Intercept Card.
Renders highly visible warnings if semantic consensus metrics diverge.
"""

import streamlit as st

class WarningBannerComponent:
    @staticmethod
    def check_and_render(low_consensus_triggered: bool, flagged_outliers: list[str]):
        """Displays a visibility banner warning if anomaly trends exceed boundaries."""
        if low_consensus_triggered:
            st.markdown(
                "<div style='background-color: #fff3e0; border-left: 5px solid #ff9800; "
                "padding: 16px; margin: 16px 0; border-radius: 4px; color: #e65100;'>"
                "⚠️ **Critical Hallucination Risk Warning:** The model council's output responses "
                "have dropped below the semantic consistency bounds. Review individual model cards "
                "carefully to extract isolated variance statements."
                "</div>",
                unsafe_allow_html=True
            )
            
        if flagged_outliers:
            st.markdown(
                f"<div style='background-color: #ffebee; border-left: 5px solid #f44336; "
                f"padding: 12px; margin: 8px 0; border-radius: 4px; color: #c62828; font-size: 13px;'>"
                f"<strong>Anomalies Isolated:</strong> Mathematical vectors flagged internal hallucination traces "
                f"inside: {', '.join(flagged_outliers)}."
                f"</div>",
                unsafe_allow_html=True
            )