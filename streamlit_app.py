# ui/streamlit_app.py

import streamlit as st
import sys, os

# Add parent path to import services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import numpy as np
import av
import tempfile
from scipy.io.wavfile import write

from services.ai_service import AIService
from services.translator import TranslationService
from prompts import CONVO_PROMPT

# --- Init services ---
ai_service = AIService()
translator = TranslationService()

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": CONVO_PROMPT}]
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Text"

st.title("🇪🇸 Spanish Conversation Assistant")
st.markdown("Talk to the AI in Spanish — with voice or text input.")

# --- Input mode toggle ---
st.radio("Input Mode", ["Text", "Voice"], key="input_mode", horizontal=True)

user_input = None

# --- Text Input Mode ---
if st.session_state.input_mode == "Text":
    with st.form("text_input_form", clear_on_submit=True):
        user_input = st.text_input("Tu mensaje en español:")
        submitted = st.form_submit_button("Send")
        if not submitted:
            user_input = None

# --- Voice Input Mode (via WebRTC) ---
else:
    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.frames = []

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            self.frames.append(frame.to_ndarray())
            return frame

    ctx = webrtc_streamer(
        key="webrtc",
        mode="SENDRECV",
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
        audio_processor_factory=AudioProcessor,
    )

    if ctx and ctx.state.playing and ctx.audio_processor:
        st.info("Recording... Press STOP above when done.")
    elif ctx and ctx.audio_processor and ctx.audio_processor.frames:
        if st.button("Transcribe"):
            audio_data = np.concatenate(ctx.audio_processor.frames, axis=0)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                write(f.name, 48000, audio_data.astype(np.int16))
                transcription = ai_service.transcribe_audio(f.name)
                user_input = transcription
                st.markdown(f"**You said:** {transcription}")
                os.remove(f.name)

# --- Process User Input ---
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    translation = translator.translate(user_input)
    if translation:
        st.markdown(f"**English translation:** {translation}")

    with st.spinner("Thinking..."):
        response = ai_service.get_text_completion(st.session_state.chat_history)

    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # TTS
    speech_path = ai_service.text_to_speech(response)
    if speech_path:
        st.audio(speech_path, format="audio/mp3")
        os.remove(speech_path)

# --- Chat Display ---
st.divider()
st.subheader("🗨️ Conversation")
for msg in st.session_state.chat_history[1:]:  # skip system message
    speaker = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
    st.markdown(f"**{speaker}:** {msg['content']}")
