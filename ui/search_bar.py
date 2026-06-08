"""
VeritasAI Unified Input Controller.
Handles text input with character counter, voice recording, image upload, and webcam.
Supports all input modalities defined in the spec.
"""

import base64
import streamlit as st
from typing import Optional, Tuple


class SearchBarComponent:
    """Multi-modal input controller for the VeritasAI search interface."""

    @staticmethod
    def render() -> Tuple[Optional[str], Optional[str]]:
        """
        Render the full multi-modal input interface.

        Returns:
            Tuple of (query_text: str | None, image_b64: str | None)
            query_text is None if user hasn't submitted yet.
        """
        # --- Hero Header ---
        st.markdown(
            """
            <div style='text-align:center; padding: 2rem 0 1rem 0;'>
                <h1 style='font-size:2.8rem; font-weight:800; margin-bottom:0.2rem;'>
                    🔍 VeritasAI
                </h1>
                <p style='color:#888; font-size:1.1rem; margin-top:0;'>
                    Ask anything. We ask everyone. Truth by consensus.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _, center_col, _ = st.columns([1, 5, 1])
        image_b64: Optional[str] = None

        with center_col:
            # --- Input mode selector ---
            mode = st.radio(
                "Input mode",
                ["📝 Text", "🎤 Voice", "🖼️ Image", "📷 Webcam"],
                horizontal=True,
                key="input_mode",
                label_visibility="collapsed",
            )

            user_input: Optional[str] = None

            # --- Text input ---
            if mode == "📝 Text":
                user_input = st.text_area(
                    label="Your question",
                    placeholder=(
                        "e.g. What are the key differences between transformer and "
                        "mamba architectures for sequence modeling?"
                    ),
                    max_chars=2000,
                    height=120,
                    key="primary_search_box",
                    label_visibility="collapsed",
                )
                if user_input:
                    char_count = len(user_input)
                    color = "#f44336" if char_count > 1800 else "#888"
                    st.markdown(
                        f"<p style='text-align:right; font-size:12px; color:{color};'>"
                        f"{char_count} / 2000 characters</p>",
                        unsafe_allow_html=True,
                    )

            # --- Voice input ---
            elif mode == "🎤 Voice":
                st.info(
                    "🎤 Voice input requires `streamlit-audio-recorder` and a Groq Whisper API key. "
                    "Record your question and it will be transcribed automatically."
                )
                try:
                    from audiorecorder import audiorecorder
                    audio = audiorecorder("🎙️ Click to Record", "⏹ Stop Recording", key="voice_recorder")
                    if audio and len(audio) > 0:
                        groq_key = st.session_state.get("groq_key") or __import__("os").getenv("GROQ_API_KEY", "")
                        if groq_key:
                            with st.spinner("Transcribing audio..."):
                                import io
                                from groq import Groq
                                audio_bytes = audio.export(format="wav").read()
                                client = Groq(api_key=groq_key)
                                transcription = client.audio.transcriptions.create(
                                    model="whisper-large-v3",
                                    file=("audio.wav", io.BytesIO(audio_bytes), "audio/wav"),
                                )
                                user_input = transcription.text
                                if user_input:
                                    st.success(f"Transcribed: *{user_input}*")
                        else:
                            st.warning("Voice transcription requires GROQ_API_KEY to be set.")
                except ImportError:
                    st.warning("Install `streamlit-audio-recorder` to enable voice input.")

            # --- Image upload ---
            elif mode == "🖼️ Image":
                uploaded = st.file_uploader(
                    "Upload an image (JPG/PNG) — analyzed by Gemini",
                    type=["jpg", "jpeg", "png"],
                    key="image_uploader",
                )
                if uploaded:
                    image_b64 = base64.b64encode(uploaded.read()).decode("utf-8")
                    st.image(uploaded, caption="Uploaded image", width=300)
                user_input = st.text_input(
                    "Question about the image:",
                    placeholder="What's happening in this image?",
                    key="image_question",
                )

            # --- Webcam capture ---
            elif mode == "📷 Webcam":
                camera_image = st.camera_input("Take a photo", key="webcam_input")
                if camera_image:
                    image_b64 = base64.b64encode(camera_image.read()).decode("utf-8")
                user_input = st.text_input(
                    "Question about the photo:",
                    placeholder="Describe what you want to know...",
                    key="webcam_question",
                )

            st.write("")

            # --- Action buttons ---
            fast_mode = st.session_state.get("fast_mode", False)
            btn_label = "⚡ Quick Query (3 models)" if fast_mode else "🚀 Full Council Query (7 models)"

            btn_col1, btn_col2 = st.columns([3, 1])
            with btn_col1:
                submit = st.button(btn_label, use_container_width=True, type="primary")
            with btn_col2:
                if st.button("🧹 Clear", use_container_width=True):
                    for key in ["primary_search_box", "image_question", "webcam_question"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

            if submit and user_input and user_input.strip():
                return user_input.strip(), image_b64

        return None, None