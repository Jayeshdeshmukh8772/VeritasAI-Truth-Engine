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
        # --- Check for query params first to pre-fill the search box ---
        if "q" in st.query_params:
            q_val = st.query_params["q"]
            st.session_state["primary_search_box"] = q_val
            # Clear parameter to prevent loop
            del st.query_params["q"]
            st.rerun()

        # --- Top Header Bar ---
        st.markdown(
            """
            <div class="vai-top-bar">
                <div class="vai-logo-container">
                    <span class="vai-purple-dot"></span>
                    <span class="vai-logo-text">VeritasAI <span class="vai-version">v2.1</span></span>
                </div>
                <div class="vai-badge-container">
                    <span class="vai-header-badge">DECISION INTELLIGENCE PLATFORM</span>
                </div>
            </div>
            <div class="vai-section-header">QUERY INPUT</div>
            """,
            unsafe_allow_html=True,
        )

        image_b64: Optional[str] = None
        user_input: Optional[str] = None

        # --- Open consolidated query card container ---
        st.markdown("<div class='vai-query-container'>", unsafe_allow_html=True)

        # --- Input mode selector styled like a tab bar ---
        mode = st.radio(
            "Input mode",
            ["📝 Text", "🎤 Voice", "🖼️ Image", "📷 Webcam"],
            horizontal=True,
            key="input_mode",
            label_visibility="collapsed",
        )

        # --- Text input ---
        if mode == "📝 Text":
            user_input = st.text_area(
                label="Your question",
                placeholder="Ask any factual question...",
                max_chars=2000,
                height=120,
                key="primary_search_box",
                label_visibility="collapsed",
            )
            if user_input:
                char_count = len(user_input)
                color = "#f44336" if char_count > 1800 else "#888"
                st.markdown(
                    f"<p style='text-align:right; font-size:12px; color:{color}; margin-top:-10px; margin-bottom:5px;'>"
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

        # --- Action buttons inside columns to prevent wrapping ---
        col1, col2, col3, _ = st.columns([2.6, 1.8, 1.2, 2.4])
        with col1:
            submit_enhance = st.button("⚡ Enhance & Dispatch", use_container_width=True, type="primary")
        with col2:
            submit_quick = st.button("⚡ Quick Run", use_container_width=True)
        with col3:
            submit_clear = st.button("🧹 Clear", use_container_width=True)

        if submit_clear:
            for key in ["primary_search_box", "image_question", "webcam_question", "image_uploader", "webcam_input"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        # --- Interactive Clickable Chips ---
        st.markdown(
            """
            <div class="vai-chips-row">
                <a class="vai-chip" href="?q=petrol+price+Pune" target="_self">petrol price Pune</a>
                <a class="vai-chip" href="?q=repo+rate+India" target="_self">repo rate India</a>
                <a class="vai-chip" href="?q=gold+price+Mumbai" target="_self">gold price Mumbai</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Close query container ---
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Handle submissions ---
        if user_input and user_input.strip():
            if submit_enhance:
                st.session_state["bypass_enhancer"] = False
                return user_input.strip(), image_b64
            elif submit_quick:
                st.session_state["bypass_enhancer"] = True
                return user_input.strip(), image_b64

        return None, None