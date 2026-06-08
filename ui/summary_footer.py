"""
VeritasAI Query Summary Footer.
Displays a concise statistics bar after results are shown:
"N models · M consensus · X flagged · Ys total"
"""

import streamlit as st
from core.result import FinalOutput, LLMStatus


class SummaryFooterComponent:
    """Renders the compact statistics footer at the bottom of the results page."""

    @staticmethod
    def render(output: FinalOutput) -> None:
        """
        Display a summary footer with key pipeline metrics.

        Args:
            output: The FinalOutput dataclass from the completed pipeline run
        """
        total_models = len(output.all_results)
        successful = sum(1 for r in output.all_results if r.status == LLMStatus.SUCCESS)
        consensus_count = len(output.all_results) - len(output.hallucination_flags)
        flagged_count = len(output.hallucination_flags)
        total_secs = round(output.total_latency_ms / 1000, 1)

        flagged_span = (
            f"<span style='margin: 0 12px;'>🛑 <strong>{flagged_count}</strong> flagged</span>"
            if flagged_count > 0
            else ""
        )

        st.write("---")
        st.markdown(
            f"<div style='text-align:center; color:#888; font-size:13px; padding:8px 0;'>"
            f"<span style='margin: 0 12px;'>🤖 <strong>{successful}</strong> / {total_models} models responded</span>"
            f"<span style='margin: 0 12px;'>✅ <strong>{consensus_count}</strong> in consensus</span>"
            f"{flagged_span}"
            f"<span style='margin: 0 12px;'>⏱️ <strong>{total_secs}s</strong> total</span>"
            f"<span style='margin: 0 12px;'>📊 <strong>{int(output.consensus_ratio * 100)}%</strong> consensus</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
