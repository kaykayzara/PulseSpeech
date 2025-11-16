import os
import json
import time
import queue
import threading

import numpy as np
import sounddevice as sd
import librosa
import parselmouth
import pandas as pd
import whisper
import joblib
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ====================
# CONFIG
# ====================

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.2          # seconds per audio block
TRANSCRIBE_INTERVAL = 2.0     # seconds between transcribe/predict calls
CHANNELS = 1
WHISPER_MODEL_NAME = "tiny"

# How much history we keep / analyze (seconds)
BUFFER_SECONDS = 6.0          # rolling buffer length
ANALYSIS_WINDOW_SECONDS = 4.0  # last N seconds used for Whisper + features

# Emotion model files (created by pulsespeech.py)
MODEL_PATH = "rf_emotion_model.joblib"
SCALER_PATH = "rf_scaler.joblib"
FEATURE_COLS_PATH = "rf_feature_cols.json"

# ====================
# GLOBALS
# ====================

audio_q = queue.Queue()
buffer = np.empty((0, CHANNELS), dtype=np.float32)
stop_flag = threading.Event()

# Will be loaded at startup
emotion_model = None
scaler = None
feature_cols = None
whisper_model = None

# ====================
# AUDIO CALLBACK
# ====================


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_q.put(indata.copy())


# ====================
# FEATURE EXTRACTION (array)
# ====================


def extract_acoustic_features_from_array(y, sr):
    """
    Mirror the features used in pulsespeech.py but operate on a
    raw waveform array instead of loading from a file.
    """

    # Ensure mono 1D
    if y.ndim > 1:
        y = y.flatten()
    y = y.astype(np.float32)

    duration = len(y) / float(sr)

    # RMS + ZCR
    rms = float(librosa.feature.rms(y=y).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())

    # Pitch via Parselmouth
    snd = parselmouth.Sound(y, sr)
    pitch = snd.to_pitch().selected_array["frequency"]
    pitch_vals = pitch[pitch > 0]
    pitch_mean = float(pitch_vals.mean()) if len(pitch_vals) else 0.0
    pitch_std = float(pitch_vals.std()) if len(pitch_vals) else 0.0

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfcc.mean(axis=1)
    mfcc_stds = mfcc.std(axis=1)

    feats = {
        "duration": duration,
        "rms": rms,
        "zcr": zcr,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
    }

    for i, v in enumerate(mfcc_means):
        feats[f"mfcc_mean_{i}"] = float(v)
    for i, v in enumerate(mfcc_stds):
        feats[f"mfcc_std_{i}"] = float(v)

    return feats


def predict_emotion_from_audio(y, sr):
    """
    Take a waveform, extract features, scale them, and return:
      - predicted emotion string
      - dict of key acoustic features to display
    """
    if emotion_model is None or scaler is None or feature_cols is None:
        return "N/A", {}

    feats = extract_acoustic_features_from_array(y, sr)

    # If almost silence, don't trust the prediction
    if feats["rms"] < 1e-4:
        return "UNKNOWN", {
            "Pitch mean": feats.get("pitch_mean", 0.0),
            "Pitch std": feats.get("pitch_std", 0.0),
            "RMS": feats.get("rms", 0.0),
            "ZCR": feats.get("zcr", 0.0),
        }

    # Build feature vector in the exact same order as training
    row = {col: feats.get(col, 0.0) for col in feature_cols}
    df = pd.DataFrame([row])

    X_scaled = scaler.transform(df)
    pred = emotion_model.predict(X_scaled)[0]

    # Choose a few "headline" features for the UI
    display_feats = {
        "Pitch mean": feats.get("pitch_mean", 0.0),
        "Pitch std": feats.get("pitch_std", 0.0),
        "RMS": feats.get("rms", 0.0),
        "ZCR": feats.get("zcr", 0.0),
    }

    return pred, display_feats


# ====================
# GUI
# ====================


class LiveMicGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PulseSpeech Live Demo")
        self.root.configure(bg="#05070B")

        # ---------- TOP FRAME: Emotion + Acoustic Features ----------
        top_frame = tk.Frame(self.root, bg="#05070B")
        top_frame.pack(fill=tk.X, padx=20, pady=(20, 10))

        # Emotion panel (left)
        emotion_frame = tk.Frame(top_frame, bg="#0A0E18", bd=0, relief=tk.FLAT)
        emotion_frame.pack(side=tk.LEFT, fill=tk.BOTH,
                           expand=True, padx=(0, 10))

        emotion_title = tk.Label(
            emotion_frame, text="Emotion", fg="#FFFFFF", bg="#0A0E18",
            font=("Helvetica", 14, "bold")
        )
        emotion_title.pack(anchor="w", padx=15, pady=(15, 5))

        self.emotion_value = tk.Label(
            emotion_frame, text="Listening...", fg="#E0E0E0", bg="#0A0E18",
            font=("Helvetica", 32, "bold")
        )
        self.emotion_value.pack(anchor="w", padx=15, pady=(0, 20))

        # Acoustic features panel (right)
        features_frame = tk.Frame(
            top_frame, bg="#0A0E18", bd=0, relief=tk.FLAT)
        features_frame.pack(side=tk.LEFT, fill=tk.BOTH,
                            expand=True, padx=(10, 0))

        features_title = tk.Label(
            features_frame, text="Acoustic Features", fg="#FFFFFF",
            bg="#0A0E18", font=("Helvetica", 14, "bold")
        )
        features_title.pack(anchor="w", padx=15, pady=(15, 5))

        self.feature_labels = {}
        for name in ["Pitch mean", "Pitch std", "RMS", "ZCR"]:
            row_frame = tk.Frame(features_frame, bg="#0A0E18")
            row_frame.pack(anchor="w", padx=15, pady=2, fill=tk.X)

            lbl_name = tk.Label(
                row_frame, text=name, fg="#CCCCCC", bg="#0A0E18",
                font=("Helvetica", 12)
            )
            lbl_name.pack(side=tk.LEFT)

            lbl_val = tk.Label(
                row_frame, text="--", fg="#FFFFFF", bg="#0A0E18",
                font=("Helvetica", 12, "bold")
            )
            lbl_val.pack(side=tk.RIGHT)
            self.feature_labels[name] = lbl_val

        # ---------- WAVEFORM ----------
        self.fig = Figure(figsize=(8, 3), facecolor="#05070B")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#05070B")
        self.ax.tick_params(left=False, labelleft=False,
                            bottom=False, labelbottom=False)
        self.line, = self.ax.plot([], [], color="#00BFFF", linewidth=2)
        self.ax.set_ylim([-1, 1])
        self.ax.set_xlim([0, SAMPLE_RATE * 2])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 10))

        # ---------- TRANSCRIPTION -----------
        transcript_frame = tk.Frame(self.root, bg="#05070B")
        transcript_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

        self.transcript_label = tk.Label(
            transcript_frame, text="“Listening...”", fg="#E0E0E0", bg="#05070B",
            font=("Helvetica", 16, "italic"), anchor="w", justify="left"
        )
        self.transcript_label.pack(fill=tk.X)

        # ---------- CONTROL BUTTONS ----------
        buttons_frame = tk.Frame(self.root, bg="#05070B")
        buttons_frame.pack(pady=(0, 20))

        self.stop_button = ttk.Button(
            buttons_frame, text="Stop", command=self.stop
        )
        self.stop_button.pack()

        # Start waveform updater + background listener thread
        self.update_waveform()
        threading.Thread(
            target=self.listen_transcribe_and_predict, daemon=True
        ).start()

    # stop everything
    def stop(self):
        stop_flag.set()
        self.root.destroy()

    # update waveform plot in real time
    def update_waveform(self):
        global buffer
        if len(buffer) > 0:
            samples_to_show = buffer[-int(SAMPLE_RATE * 0.4):, 0]
            x = np.arange(len(samples_to_show))
            self.line.set_data(x, samples_to_show)
            self.ax.set_xlim([0, len(samples_to_show)])
            self.canvas.draw_idle()
        if not stop_flag.is_set():
            self.root.after(15, self.update_waveform)

    # background audio listener + Whisper + Emotion prediction
    def listen_transcribe_and_predict(self):
        global buffer
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
        ):
            last_transcription = ""
            start_time = time.time()
            while not stop_flag.is_set():
                # pull audio blocks into buffer
                while not audio_q.empty():
                    block = audio_q.get()
                    buffer = np.append(buffer, block, axis=0)

                    # keep only the last BUFFER_SECONDS
                    max_len = int(SAMPLE_RATE * BUFFER_SECONDS)
                    if len(buffer) > max_len:
                        buffer = buffer[-max_len:]

                # every TRANSCRIBE_INTERVAL seconds, run Whisper + Emotion
                if time.time() - start_time >= TRANSCRIBE_INTERVAL and len(buffer) > 0:
                    start_time = time.time()

                    # copy buffer so it doesn't change mid-analysis
                    audio_copy = np.copy(buffer)
                    audio_copy = np.clip(audio_copy, -1, 1)
                    mono = audio_copy.flatten().astype(np.float32)

                    # only use last ANALYSIS_WINDOW_SECONDS to speed things up
                    max_window = int(SAMPLE_RATE * ANALYSIS_WINDOW_SECONDS)
                    if len(mono) > max_window:
                        mono = mono[-max_window:]

                    # Run Whisper + Emotion in a background thread
                    threading.Thread(
                        target=self.process_audio_in_background,
                        args=(mono, last_transcription),
                        daemon=True,
                    ).start()

                time.sleep(0.05)

    def process_audio_in_background(self, mono, last_transcription):
        # ----- TRANSCRIPTION -----
        text = ""
        try:
            result = whisper_model.transcribe(
                mono,
                fp16=False,
                language="en",
                condition_on_previous_text=False,
            )
            text = result.get("text", "").strip()
        except Exception as e:
            print("Whisper error:", e)

        if text and text != last_transcription:
            self.root.after(
                0, lambda: self.transcript_label.config(text=f"“{text}”")
            )

        # ----- EMOTION -----
        try:
            pred_emotion, disp_feats = predict_emotion_from_audio(
                mono, SAMPLE_RATE
            )
        except Exception as e:
            print("Emotion prediction error:", e)
            return

        if pred_emotion and pred_emotion != "N/A":
            self.root.after(
                0, lambda: self.emotion_value.config(text=pred_emotion.upper())
            )

        def update_feats():
            for name, val in disp_feats.items():
                if name in self.feature_labels:
                    self.feature_labels[name].config(text=f"{val:.2f}")

        self.root.after(0, update_feats)


# ====================
# MAIN
# ====================


def load_models():
    global emotion_model, scaler, feature_cols, whisper_model

    print("Loading Whisper model (tiny)...")
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    print("✅ Whisper model loaded.")

    if not (
        os.path.exists(MODEL_PATH)
        and os.path.exists(SCALER_PATH)
        and os.path.exists(FEATURE_COLS_PATH)
    ):
        print(
            "⚠️ Emotion model files not found.\n"
            "Make sure you ran pulsespeech.py to train and save the model."
        )
        return

    print("Loading emotion model and scaler...")
    emotion_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_COLS_PATH, "r") as f:
        feature_cols = json.load(f)
    print("✅ Emotion model loaded.")


if __name__ == "__main__":
    load_models()

    root = tk.Tk()
    app = LiveMicGUI(root)
    root.mainloop()
