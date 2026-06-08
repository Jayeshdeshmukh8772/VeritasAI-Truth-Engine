"""
VeritasAI Per-Model Result Card.
Expandable card for each model showing trust bar, peer ranking, latency,
full response, and optionally peer review critique text.
"""

import streamlit as st
from typing import List
from core.result import LLMResult, LLMStatus


class ModelCardComponent:
    """Renders a grid of per-model expandable result cards."""

    @staticmethod
    def render_grid(all_results: List[LLMResult], show_debate: bool = False) -> None:
        """
        Render expandable cards for every model in the pipeline results.

        Args:
            all_results: List of LLMResult objects from all models
            show_debate: If True, show the raw peer review ranking text
        """
        if not all_results:
            return

        st.write("---")
        st.markdown("### 🤖 Individual Model Responses")

        for res in all_results:
            # Build expander label
            if res.status == LLMStatus.SUCCESS:
                if res.is_outlier:
                    icon = "🛑"
                    suffix = f"Flagged — Trust: {res.trust_score:.3f}"
                elif res.trust_score >= 0.65:
                    icon = "✅"
                    suffix = f"Trusted — {res.trust_score:.3f}"
                else:
                    icon = "⚠️"
                    suffix = f"Borderline — {res.trust_score:.3f}"
            elif res.status == LLMStatus.SKIPPED:
                icon = "⚪"
                suffix = f"Skipped — {res.error_msg or 'No API key configured'}"
            else:
                icon = "❌"
                suffix = f"Failed — {res.error_type or 'Error'}: {res.error_msg or 'No details'}"

            label = f"{icon} {res.model}  ·  {suffix}  ·  {res.latency_ms}ms"

            with st.expander(label, expanded=False):
                if res.status == LLMStatus.SUCCESS:
                    # Metrics row
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Trust Score", f"{res.trust_score:.3f}")
                    m2.metric("Peer Ranking", f"{res.peer_rank_score:.3f}")
                    m3.metric("Latency", f"{res.latency_ms}ms")
                    m4.metric("Tokens", str(res.tokens_used) if res.tokens_used else "—")

                    # Trust bar
                    trust_color = (
                        "🟢" if res.trust_score >= 0.65
                        else "🟡" if res.trust_score >= 0.45
                        else "🔴"
                    )
                    trust_pct = min(100, int(res.trust_score * 100))
                    st.markdown(
                        f"""
                        <div style='margin: 8px 0;'>
                            <span style='font-size:12px; color:#666;'>Trust Level: {trust_color}</span>
                            <div style='background:#eee; border-radius:4px; height:8px; margin-top:4px;'>
                                <div style='
                                    background: {"#4caf50" if res.trust_score >= 0.65 else "#ff9800" if res.trust_score >= 0.45 else "#f44336"};
                                    width:{trust_pct}%; height:8px; border-radius:4px;
                                '></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.write("")
                    st.markdown("**Response:**")
                    st.write(res.response)

                    if res.is_outlier:
                        st.warning(
                            "⚠️ This response was mathematically flagged as a potential hallucination "
                            "by the DBSCAN semantic clustering engine and peer review scoring."
                        )

                else:
                    if res.status == LLMStatus.SKIPPED:
                        st.info(
                            f"This model was skipped. "
                            f"Reason: {res.error_msg or 'API key not configured or quota exceeded.'}"
                        )
                    else:
                        st.error(
                            f"**Error Type:** `{res.error_type}`  \n"
                            f"**Message:** {res.error_msg}"
                        )