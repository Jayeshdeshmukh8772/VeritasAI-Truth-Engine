"""
VeritasAI Consensus Answer Card.
Prominent hero card displaying the synthesized answer with confidence badge,
copy button, TTS trigger, and follow-up question suggestions.
"""

import streamlit as st
from typing import List, Optional
from ui.tts import TTSComponent


class ConsensusCardComponent:
    """Renders the primary consensus answer card — the most prominent UI element."""

    @staticmethod
    def render(
        answer: Optional[str],
        consensus_ratio: float,
        follow_up_questions: Optional[List[str]] = None,
        on_followup_click=None,
    ) -> None:
        """
        Display the synthesized answer with confidence badge, TTS, and copy button.

        Args:
            answer: The synthesized answer text (None if low consensus)
            consensus_ratio: Float 0-1 representing consensus strength
            follow_up_questions: Optional list of 3 follow-up suggestions
            on_followup_click: Optional callback when a follow-up is clicked
        """
        st.write("---")

        if not answer:
            st.markdown(
                """
                <div style='background: linear-gradient(135deg, #fff3e0, #fff8e1);
                    border-left: 5px solid #ff9800; border-radius: 12px;
                    padding: 24px; margin: 16px 0;'>
                    <h3 style='color: #e65100; margin-top:0;'>⚠️ Low Consensus — No Synthesis</h3>
                    <p style='color:#bf360c;'>
                        The models disagree significantly. Review individual responses below
                        to form your own judgment. No synthesized answer is provided
                        to avoid misleading you.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        # Badge color & label based on confidence
        if consensus_ratio >= 0.75:
            badge_color = "#2e7d32"
            badge_bg = "#e8f5e9"
            badge_text = "HIGH CONFIDENCE"
            border_color = "#4caf50"
        elif consensus_ratio >= 0.50:
            badge_color = "#e65100"
            badge_bg = "#fff3e0"
            badge_text = "MODERATE CONFIDENCE"
            border_color = "#ff9800"
        else:
            badge_color = "#b71c1c"
            badge_bg = "#ffebee"
            badge_text = "LOW CONFIDENCE"
            border_color = "#f44336"

        confidence_pct = int(consensus_ratio * 100)

        st.markdown(
            f"""
            <div style='
                background: {badge_bg};
                border: 1px solid {border_color};
                border-left: 5px solid {border_color};
                border-radius: 12px;
                padding: 24px;
                margin: 16px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            '>
                <div style='display:flex; align-items:center; margin-bottom:12px; gap:10px;'>
                    <span style='
                        background:{badge_color}; color:white;
                        padding:4px 14px; border-radius:20px;
                        font-size:11px; font-weight:700; letter-spacing:0.5px;
                    '>{badge_text} · {confidence_pct}%</span>
                    <span style='color:#888; font-size:12px;'>Consensus Answer</span>
                </div>
                <h3 style='margin-top:0; margin-bottom:16px; color:#1a1a1a;'>
                    🧠 VeritasAI Synthesis
                </h3>
                <div style='
                    font-size:16px; line-height:1.75; color:#222;
                    white-space: pre-wrap;
                '>{answer}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Action buttons row
        action_col1, action_col2, action_col3 = st.columns([2, 2, 4])
        with action_col1:
            TTSComponent.render_button(answer, "🔊 Read Aloud")
        with action_col2:
            # Copy button using clipboard JS
            st.markdown(
                f"""
                <button onclick="navigator.clipboard.writeText(`{answer[:500].replace('`', '')}`).then(() => {{
                    this.textContent='✅ Copied!'; setTimeout(() => this.textContent='📋 Copy', 1500);
                }})"
                style='
                    background:#f5f5f5; border:1px solid #ddd; border-radius:6px;
                    padding:8px 16px; cursor:pointer; font-size:13px;
                    transition: background 0.2s;
                '>📋 Copy Answer</button>
                """,
                unsafe_allow_html=True,
            )

        # Follow-up questions
        if follow_up_questions:
            st.write("")
            st.markdown("**💡 Follow-up Questions:**")
            for i, q in enumerate(follow_up_questions[:3]):
                if st.button(f"→ {q}", key=f"followup_{i}", use_container_width=False):
                    if on_followup_click:
                        on_followup_click(q)
                    else:
                        st.session_state["followup_query"] = q
                        st.rerun()