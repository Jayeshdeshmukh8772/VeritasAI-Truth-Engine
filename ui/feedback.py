"""
VeritasAI User Feedback Component.
Thumbs up / thumbs down feedback buttons with optional comment.
Writes votes to SQLite feedback table via logger.
"""

import streamlit as st
from typing import Optional


class FeedbackComponent:
    """Renders user feedback controls and persists votes to database."""

    @staticmethod
    def render(
        query_hash: str,
        session_id: str,
        logger=None,
    ) -> None:
        """
        Display thumbs-up/down feedback buttons below the consensus answer.

        Args:
            query_hash: SHA-256 hash of the current query (used as primary key in DB)
            session_id: Current Streamlit session identifier
            logger: VeritasLogger instance for persisting feedback to SQLite
        """
        st.write("---")
        st.markdown("**Was this answer helpful?**")

        feedback_key = f"feedback_{query_hash[:8]}"
        if st.session_state.get(feedback_key):
            vote = st.session_state[feedback_key]
            emoji = "👍" if vote == "up" else "👎"
            st.success(f"{emoji} Thank you for your feedback!")
            return

        col1, col2, col3 = st.columns([1, 1, 4])

        with col1:
            if st.button("👍 Helpful", key=f"up_{query_hash[:8]}", use_container_width=True):
                st.session_state[feedback_key] = "up"
                if logger:
                    logger.log_feedback(session_id, query_hash, "up")
                st.rerun()

        with col2:
            if st.button("👎 Not Helpful", key=f"down_{query_hash[:8]}", use_container_width=True):
                st.session_state[feedback_key] = "down"
                if logger:
                    logger.log_feedback(session_id, query_hash, "down")
                st.rerun()
