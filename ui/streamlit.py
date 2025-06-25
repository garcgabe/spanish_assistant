import streamlit as st
from services.ai_service import AIService
from services.translator import TranslationService
from services.audio_recorder import record_audio
from prompts import CONVO_PROMPT
import os
import sys

# --- Initialize services ---
translator = TranslationService()
ai_service = AIService()

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": CONVO_PROMPT}]
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Text"
if "last_audio" in st.session_state:
    try:
        os.remove(st.session_state.last_audio)
    except:
        pass
    del st.session_state.last_audio

# --- Title ---
st.title("🇪🇸 Spanish Conversation Assistant")

st.markdown("Talk to the assistant in Spanish and get voice + text responses.")

# --- Input Mode Toggle ---
st.radio("Input Mode:", ["Text", "Voice"], horizontal=True, key="input_mode")

user_input = None

# --- Get User Input ---
if st.session_state.input_mode == "Text":
    with st.form("text_input_form", clear_on_submit=True):
        user_input = st.text_input("Tu mensaje en español:", key="text_input")
        submitted = st.form_submit_button("Send")
        if not submitted:
            user_input = None
else:
    if st.button("🎤 Record Voice"):
        audio_path = record_audio()
        user_input = ai_service.transcribe_audio(audio_path)
        st.write(f"**You said:** {user_input}")
        os.remove(audio_path)

# --- Handle Input & Response ---
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Translate (optional for learner)
    translation = translator.translate(user_input)
    if translation:
        st.markdown(f"**English translation:** {translation}")

    # Get AI response
    with st.spinner("Thinking..."):
        response = ai_service.get_text_completion(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Convert response to speech
    speech_file = ai_service.text_to_speech(response)
    if speech_file:
        st.audio(speech_file, format="audio/mp3")
        st.session_state.last_audio = speech_file  # For cleanup

# --- Show Conversation ---
st.markdown("---")
st.subheader("🗨️ Conversation")
for message in st.session_state.chat_history[1:]:  # Skip system prompt
    if message["role"] == "user":
        st.markdown(f"**🧑 You:** {message['content']}")
    elif message["role"] == "assistant":
        st.markdown(f"**🤖 Assistant:** {message['content']}")
