import sounddevice as sd
import numpy as np
import whisper
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ====================
# CONFIG
# ====================

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.2
TRANSCRIBE_INTERVAL = 5.0
MODEL_NAME = "tiny"
CHANNELS = 1

# ====================
# INIT
# ====================

print("Loading Whisper model (tiny)...")
model = whisper.load_model(MODEL_NAME)
print("✅ Model loaded. GUI starting...")

# ====================
# AUDIO QUEUE
# ====================

audio_q = queue.Queue()
buffer = np.empty((0, CHANNELS), dtype=np.float32)
stop_flag = threading.Event()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_q.put(indata.copy())

# ====================
# GUI SETUP
# ====================


class LiveMicGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PulseSpeech Live Mic")
        self.root.configure(bg="black")

        # figure for waveform
        self.fig = Figure(figsize=(8, 3), facecolor="black")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("black")
        self.ax.tick_params(left=False, labelleft=False,
                            bottom=False, labelbottom=False)
        self.line, = self.ax.plot([], [], color="#00BFFF", linewidth=2)

        self.ax.set_ylim([-1, 1])
        self.ax.set_xlim([0, SAMPLE_RATE * 2])

        # embed matplotlib figure into tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # transcription label
        self.transcript_label = tk.Label(
            self.root, text="Listening...", fg="white", bg="black",
            font=("Helvetica", 14)
        )
        self.transcript_label.pack(pady=10)

        # stop button
        self.stop_button = ttk.Button(
            self.root, text="Stop", command=self.stop)
        self.stop_button.pack(pady=10)

        # start waveform and listener threads
        self.update_waveform()
        threading.Thread(target=self.listen_and_transcribe,
                         daemon=True).start()

    # stop everything
    def stop(self):
        stop_flag.set()
        self.root.destroy()

    # update waveform plot in real time
    def update_waveform(self):
        global buffer
        if len(buffer) > 0:
            samples_to_show = buffer[-int(SAMPLE_RATE * 2):, 0]
            x = np.arange(len(samples_to_show))
            self.line.set_data(x, samples_to_show)
            self.ax.set_xlim([0, len(samples_to_show)])
            self.canvas.draw()
        if not stop_flag.is_set():
            self.root.after(50, self.update_waveform)

    # background listening + transcription
    def listen_and_transcribe(self):
        global buffer
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
        ):
            last_transcript = ""
            start_time = time.time()
            while not stop_flag.is_set():
                while not audio_q.empty():
                    block = audio_q.get()
                    buffer = np.append(buffer, block, axis=0)

                if time.time() - start_time >= TRANSCRIBE_INTERVAL and len(buffer) > 0:
                    audio_copy = np.copy(buffer)
                    start_time = time.time()

                    # normalize audio
                    audio_copy = np.clip(audio_copy, -1, 1)

                    # convert to mono float32 array
                    mono = audio_copy.flatten().astype(np.float32)

                    # transcribe with whisper
                    result = model.transcribe(mono, fp16=False)
                    text = result.get("text", "").strip()

                    if text and text != last_transcript:
                        self.transcript_label.config(text=f"“{text}”")
                        last_transcript = text

                time.sleep(0.1)


# ====================
# MAIN
# ====================

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveMicGUI(root)
    root.mainloop()
