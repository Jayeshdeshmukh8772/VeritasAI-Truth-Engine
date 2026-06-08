"""
VeritasAI Enhanced Query Diff Display.
Shows the original vs AI-rewritten query so users understand the enhancement.
"""

import streamlit as st


class EnhancedQueryComponent:
    """Renders the query enhancement diff card between original and enhanced query."""

    @staticmethod
    def render(original: str, enhanced: str) -> None:
        """
        Display the original query alongside the enhanced version with visual diff styling.

        Args:
            original: The raw user input string
            enhanced: The AI-rewritten, enriched query string
        """
        if not original or not enhanced:
            return

        # Don't show if enhancement didn't change anything meaningful
        if original.strip().lower() == enhanced.strip().lower():
            return

        with st.expander("🔍 Query Enhancement — See how your question was refined", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Query**")
                st.markdown(
                    f"<div style='background:#f5f5f5; padding:12px; border-radius:8px; "
                    f"color:#666; font-size:14px; line-height:1.5;'>{original}</div>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown("**Enhanced Query** ✨")
                st.markdown(
                    f"<div style='background:#e8f5e9; padding:12px; border-radius:8px; "
                    f"color:#1b5e20; font-size:14px; line-height:1.5; border-left:3px solid #4caf50;'>"
                    f"{enhanced}</div>",
                    unsafe_allow_html=True,
                )
