"""
VeritasAI Trust Score Visualization.
Renders a horizontal bar chart of per-model trust scores with color coding.
Green = trusted, Amber = borderline, Red = flagged outlier.
"""

import streamlit as st
import pandas as pd
from core.result import FinalOutput


class TrustChartComponent:
    """Renders the per-model trust score horizontal bar chart."""

    @staticmethod
    def render(output: FinalOutput) -> None:
        """
        Display a color-coded horizontal bar chart of trust scores.

        Args:
            output: FinalOutput containing trust_scores and hallucination_flags
        """
        if not output.trust_scores:
            return

        st.write("---")
        st.markdown("### 📊 Model Trust Score Analysis")

        flagged = set(output.hallucination_flags)
        rows = []
        for model, score in output.trust_scores.items():
            if score >= 0.65:
                status = "✅ Trusted"
                color = "#4caf50"
            elif score >= 0.45:
                status = "⚠️ Borderline"
                color = "#ff9800"
            else:
                status = "🛑 Flagged"
                color = "#f44336"

            rows.append({
                "Model": model,
                "Trust Score": round(score, 3),
                "Status": status,
            })

        df = pd.DataFrame(rows).sort_values("Trust Score", ascending=False)
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Trust Score": st.column_config.ProgressColumn(
                    "Trust Score",
                    help="Dual-signal trust: 60% semantic alignment + 40% peer ranking",
                    format="%.3f",
                    min_value=0,
                    max_value=1,
                ),
            },
            hide_index=True,
        )

        # Also show a bar chart
        chart_df = pd.DataFrame(
            {"Trust Score": output.trust_scores}
        )
        st.bar_chart(chart_df, use_container_width=True, height=200)
