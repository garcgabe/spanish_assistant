import os
import tempfile
import subprocess
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from config import OPENAI_API_KEY, OPENAI_TEXT_MODEL, OPENAI_AUDIO_MODEL, ELEVEN_LABS_API_KEY

class AIService:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.elevenlabs = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)

    def get_text_completion(self, context, max_tokens=1500):
        res = self.client.chat.completions.create(
            model=OPENAI_TEXT_MODEL,
            messages=context,
            max_tokens=max_tokens,
        )
        return res.choices[0].message.content

    def transcribe_audio(self, file_path: str, lang="es") -> str:
        with open(file_path, "rb") as f:
            res = self.client.audio.transcriptions.create(
                model=OPENAI_AUDIO_MODEL,
                file=f,
                language=lang
            )
        os.remove(file_path)
        return res.text

    def text_to_speech(self, text: str) -> str:
        response = self.elevenlabs.text_to_speech.convert(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.15,
                speed=0.75
            ),
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            for chunk in response:
                f.write(chunk)
            return f.name

    def play_audio(self, path: str):
        subprocess.call(["afplay", path])  # macOS only
