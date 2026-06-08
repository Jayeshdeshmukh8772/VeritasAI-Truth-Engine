"""
VeritasAI Text-to-Speech Component.
Uses the browser's native SpeechSynthesis Web API via st.components.v1.html.
No external API required — works fully client-side.
"""

import streamlit as st
import streamlit.components.v1 as components


class TTSComponent:
    """Browser-native TTS component using SpeechSynthesis API."""

    @staticmethod
    def render_button(text: str, button_label: str = "🔊 Read Aloud") -> None:
        """
        Render a button that reads the given text aloud using browser SpeechSynthesis.

        Args:
            text: The text content to be spoken aloud
            button_label: Display label for the TTS button
        """
        if not text:
            return

        # Escape the text for safe JavaScript injection
        safe_text = (
            text.replace("\\", "\\\\")
                .replace("`", "\\`")
                .replace("$", "\\$")
                .replace('"', '\\"')
                .replace("\n", " ")
        )

        tts_html = f"""
        <div style="margin: 4px 0;">
            <button
                id="tts-btn"
                onclick="speakText()"
                style="
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 13px;
                    font-weight: 500;
                    transition: opacity 0.2s;
                "
                onmouseover="this.style.opacity='0.85'"
                onmouseout="this.style.opacity='1'"
            >
                {button_label}
            </button>
            <button
                id="tts-stop-btn"
                onclick="stopSpeech()"
                style="
                    background: #666;
                    color: white;
                    border: none;
                    padding: 8px 14px;
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 13px;
                    margin-left: 8px;
                    display: none;
                "
            >
                ⏹ Stop
            </button>
        </div>
        <script>
            const textToSpeak = `{safe_text}`;
            let currentUtterance = null;

            function speakText() {{
                if ('speechSynthesis' in window) {{
                    window.speechSynthesis.cancel();
                    currentUtterance = new SpeechSynthesisUtterance(textToSpeak);
                    currentUtterance.rate = 0.95;
                    currentUtterance.pitch = 1.0;
                    currentUtterance.volume = 1.0;
                    currentUtterance.onstart = function() {{
                        document.getElementById('tts-stop-btn').style.display = 'inline-block';
                    }};
                    currentUtterance.onend = function() {{
                        document.getElementById('tts-stop-btn').style.display = 'none';
                    }};
                    window.speechSynthesis.speak(currentUtterance);
                }} else {{
                    alert('Text-to-speech is not supported in your browser.');
                }}
            }}

            function stopSpeech() {{
                window.speechSynthesis.cancel();
                document.getElementById('tts-stop-btn').style.display = 'none';
            }}
        </script>
        """
        components.html(tts_html, height=60)
