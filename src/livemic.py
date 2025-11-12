import sounddevice as sd
import numpy as np
import queue
import threading
import whisper
import sys
import time
import os

# ====================
# CONFIG
# ====================

SAMPLE_RATE = 16000
BLOCK_DURATION = 1.0
MODEL_NAME = "small"
OUTPUT_DIR = "live_captures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================
# INIT MODELS
# ====================

print("Loading Whisper model...")
model = whisper.load_model(MODEL_NAME)
print("Model loaded. Speak into your min - press Ctrl+C to stop.\n")

# ====================
# AUDIO QUEUE
# ====================

audio_q = queue.Queue()
stop_flag = False


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_q.put(indata.copy())

# ====================
# LIVE DISPLAY + RECORD LOOP
# ====================


def visualize_audio(data):
    """simple terminal visualization"""
    rms = np.sqrt(np.mear(data**2))
    bar_len = int(min(60, rms * 1000))
    bar = "█" * bar_len
    sys.stdout.write(f"\r[{bar:<60}]")
    sys.stdout.flush()


def record_and_transcribe():
    buffer = np.empty((0, 1), dtype=np.float32)
    last_transcription = ""

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * BLOCK_DURATION)
    ):
        while not stop_flag:
            while not audio_q.empty():
                block = audio_q.get()
                buffer = np.append(buffer, block)
                visualize_audio(block)

                if len(buffer) / SAMPLE_RATE >= 5.0:
                    temp_path = os.path.join(OUTPUT_DIR, "temp.wav")
                    sd.write(temp_path, buffer, SAMPLE_RATE)
                    result = model.transcribe(temp_path)
                    text = result.get("text", "").strip()
                    if text and text != last_transcription:
                        print(f"\n🗣️ {text}\n")
                        last_transcription = text
                    buffer = np.empty((0, 1), dtype=np.float32)
            time.sleep(0.1)


try:
    record_and_transcribe()
except KeyboardInterrupt:
    print("\nStopping live transcription...")
    stop_flag = True
