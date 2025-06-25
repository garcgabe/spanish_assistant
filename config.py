import os
import streamlit as st

DEEPL_ACCESS_KEY = os.getenv("DEEPL_ACCESS_KEY", st.secrets["DEEPL_ACCESS_KEY"])
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", st.secrets["OPENAI_API_KEY"])
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY", st.secrets["ELEVEN_LABS_API_KEY"])

DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"
OPENAI_TEXT_MODEL = "gpt-4o-mini"
OPENAI_AUDIO_MODEL = "whisper-1"
SAMPLE_RATE = 44100
MAX_RECORD_TIME = 60
