import sounddevice as sd
import numpy as np
import tempfile
from scipy.io.wavfile import write
import time
from config import SAMPLE_RATE, MAX_RECORD_TIME

def record_audio() -> str:
    duration = MAX_RECORD_TIME
    print("Press ENTER to start recording...")
    input()
    print("Recording... Press ENTER to stop.")

    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')

    start = time.time()
    input()
    sd.stop()
    elapsed = time.time() - start

    trimmed = recording[:int(elapsed * SAMPLE_RATE)]
    trimmed = np.int16(trimmed * 32767)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        write(f.name, SAMPLE_RATE, trimmed)
        return f.name
