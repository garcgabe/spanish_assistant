# services/audio_recorder.py

import sounddevice as sd
import numpy as np
import tempfile
import time
from scipy.io.wavfile import write
from config import SAMPLE_RATE, MAX_RECORD_TIME


def record_audio_cli() -> str:
    """Record audio from the command line using sounddevice"""
    print("Press ENTER to start recording...")
    input()
    print("Recording... Press ENTER to stop (max 60s)")
    recording = sd.rec(
        int(MAX_RECORD_TIME * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )

    start_time = time.time()
    input()
    sd.stop()
    elapsed = time.time() - start_time
    print("Recording stopped. Transcribing...")

    trimmed = recording[:int(elapsed * SAMPLE_RATE)]
    trimmed = np.int16(trimmed * 32767)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        write(f.name, SAMPLE_RATE, trimmed)
        return f.name
