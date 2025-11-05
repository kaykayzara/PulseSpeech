# %%


import numpy as np
import librosa
import parselmouth
import whisper
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import os
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile


# %%
BASE_DIR = "C:/Users/kayre/Downloads/emovome"
AUDIO_DIR = f"{BASE_DIR}/Audios"
LABELS_PATH = f"{BASE_DIR}/labels.csv"
TRANSCRIPTS_PATH = f"{BASE_DIR}/transcriptions.csv"

labels_df = pd.read_csv(LABELS_PATH)
transcripts_df = pd.read_csv(TRANSCRIPTS_PATH)
print(labels_df.head())
print(transcripts_df.head())

merged_df = pd.merge(labels_df, transcripts_df, on="file_id", how="inner")


print("Merged dataset shape:", df.shape)
print(df.head(3))

# PulseSpeech: Multiclass Speech Emotion + Agitation Notebook
# 1. Ingest audio files
# 2. Transcribe to text (Whisper)
# 3. Extract acoustic/prosodic features (pitch, energy, pauses, MFCCs)
# 4. Extract text features (sentiment + embeddings + words/sec)
# 5. Train a MULTICLASS model: e.g. [neutral, happy, sad, angry, frustrated/agitated]
# 6. Support optional per-speaker baselines to personalize detection
# 7. Evaluate with classification report


# %% imports and paths


BASE_DIR = "C:/Users/kayre/Downloads/emovome"
AUDIO_DIR = f"{BASE_DIR}/Audios"
LABELS_PATH = f"{BASE_DIR}/labels.csv"
TRANSCRIPTS_PATH = f"{BASE_DIR}/transcriptions.csv"
PARTICIPANTS_PATH = f"{BASE_DIR}/participants_ids.csv"

# read dataset tables
labels_df = pd.read_csv(LABELS_PATH)
transcripts_df = pd.read_csv(TRANSCRIPTS_PATH)

# build full audio path
merged_df["audio_path"] = merged_df["file_id"].apply(
    lambda x: os.path.join(AUDIO_DIR, f"{x}.wav"))

# pick a label column
merged_df["emotion"] = merged_df["category_E"]

# keep only what we need
merged_df = merged_df[["audio_path", "emotion",
                       "transcription", "participant_id"]]
print(merged_df.head())


# %% load heavy models once
# you can skip whisper if you trust transcriptions.csv


whisper_model = whisper.load_model("small")
sent_model = pipeline("sentiment-analysis")
sbert = SentenceTransformer("all-MiniLM-L6-v2")


# %% load ravdess

RAVDESS_DIR = r"C:\Users\kayre\Downloads\ravdess"


def load_ravdess(base_dir=RAVDESS_DIR):
    audio_paths = []
    labels = []

    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                full = os.path.join(root, f)
                # ravdess filenames encode emotion in the name: 03 = happy, 05 = angry, etd
                parts = f.split("-")
                emotion_id = int(parts[2])
                audio_paths.append(full)
                labels.append(emotion_id)

    df = pd.DataFrame({"audio_path": audio_paths, "emotion_id": labels})
    return df


rav_df = load_ravdess()
print(rav_df.head())


# %%
# record to wav


def record_to_wav(seconds=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1)
    sd.wait()
    print("Done recording.")
    audio = audio.flatten()

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    wav.write(tmp_path, sr, (audio * 32767).astype("int16"))
    return tmp_path


# %% transcription helper (we might not need it if we trust dataset text)


def transcribe_audio(path: str):
    result = whisper_model.transcribe(path)
    text = result["text"].strip()
    segments = result.get("segments", [])
    duration = librosa.get_duration(filename=path)
    return text, segments, duration


# %% audio features


def extract_audio_features(path, sr=16000):
    try:
        y, sr = librosa.load(path, sr=sr)
    except Exception as e:
        print("could not load", path, e)
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()
    return np.concatenate([mfcc, [zcr, rms]])


X_audio = []
X_text = []
y = []

X_audio = np.vstack(X_audio)
X_text = np.vstack(X_text)

# combine audio and text
X = np.hstack([X_text, X_text])


# %% acoustic features


def extract_acoustic_features(path: str) -> dict:
    y, sr = librosa.load(path, sr=16000)
    total_duration = len(y) / sr

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    rms = float(librosa.feature.rms(y=y).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())

    intervals = librosa.effects.split(y, top_db=30)
    speech_dur = sum((i[1] - i[0]) for i in intervals) / sr
    pause_ratio = 1.0 - \
        (speech_dur / total_duration) if total_duration > 0 else 0.0

    # pitch via parselmouth
    snd = parselmouth.Sound(path)
    pitch = snd.to_pitch()
    pitch_vals = pitch.selected_array["frequency"]
    pitch_vals = pitch_vals[pitch_vals > 0]
    if len(pitch_vals) > 0:
        pitch_mean = float(np.mean(pitch_vals))
        pitch_std = float(np.std(pitch_vals))
    else:
        pitch_mean = 0.0
        pitch_std = 0.0

    feats = {
        "duration": float(total_duration),
        "rms": rms,
        "zcr": zcr,
        "pause_ratio": float(pause_ratio),
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
    }

    for i, v in enumerate(mfcc_mean):
        feats[f"mfcc_mean_{i}"] = float(v)
    for i, v in enumerate(mfcc_std):
        feats[f"mfcc_std_{i}"] = float(v)

    return feats


# %% text features


def extract_text_features(text: str, duration: float) -> dict:
    sent = sent_model(text[:512])[0]
    sent_is_pos = 1 if sent["label"].lower().startswith("pos") else 0

    words = text.split()
    wps = len(words) / max(duration, 1e-6)

    emb = sbert.encode([text])[0]

    feats = {
        "sent_pos_label": sent_is_pos,
        "sent_score": float(sent["score"]),
        "words_per_sec": float(wps),
        "embed_mean": float(np.mean(emb)),
        "embed_std": float(np.std(emb)),
    }
    return feats


# %% per-speaker baselines


personal_baselines = {}


def build_baseline_for_speaker(speaker_id: str, audio_files: list):
    rows = []
    for f in audio_files:
        # here we re-transcribe; you could also look up transcription in merged_df instead
        text, segs, dur = transcribe_audio(f)
        a_feats = extract_acoustic_features(f)
        t_feats = extract_text_features(text, dur)
        feats = {**a_feats, **t_feats}
        rows.append(feats)
    df = pd.DataFrame(rows)
    mu = df.mean(numeric_only=True)
    sigma = df.std(numeric_only=True).replace(0, 1e-6)
    personal_baselines[speaker_id] = {
        "mu": mu.to_dict(), "sigma": sigma.to_dict()}
    return personal_baselines[speaker_id]


def deviation_from_baseline(speaker_id: str, feats: dict, keys=None) -> float:
    if speaker_id not in personal_baselines:
        return 0.0
    if keys is None:
        keys = ["pitch_mean", "rms", "words_per_sec", "pause_ratio"]
    base = personal_baselines[speaker_id]
    mu = base["mu"]
    sigma = base["sigma"]
    score = 0.0
    cnt = 0
    for k in keys:
        if k in feats and k in mu:
            z = (feats[k] - mu[k]) / sigma[k]
            if k == "pause_ratio":
                z = abs(z)
            score += max(0, z)
            cnt += 1
    return score / cnt if cnt > 0 else 0.0


# %% extract all features for ONE row from merged_df


def extract_all_features(path: str, transcript: str = None, speaker_id: str = None) -> dict:
    # use provided transcript if we have it (EMOVOME gives it!)
    if transcript is None:
        text, segs, dur = transcribe_audio(path)
    else:
        text = transcript
        dur = librosa.get_duration(filename=path)

    a_feats = extract_acoustic_features(path)
    t_feats = extract_text_features(text, dur)
    feats = {**a_feats, **t_feats}

    dev = deviation_from_baseline(speaker_id, feats) if speaker_id else 0.0
    feats["personal_deviation"] = dev
    feats["transcript"] = text
    return feats


# %%
# live predict


def live_predict(clf, feature_cols, seconds=4, speaker_id=None):
    # 1. record mic
    wav_path = record_to_wav(seconds=seconds, sr=16000)

    # 2. transcribe with whisper
    resuls = predict_emotion(
        clf,
        feature_cols,
        path=wav_path,
        transcript=None,
        speaker_id=speaker_id
    )
    print("Live prediction:", result["predicted_emotion"])
    print("Probs:", result["proba"])
    print("Transcript:", result["transcript"])
    return result


# %% train multiclass model on EMOVOME


def train_multiclass_model_from_merged(df: pd.DataFrame):
    rows = []
    for _, row in df.iterrows():
        audio_path = row["audio_path"]
        emotion = row["emotion"]
        transcript = row["transcription"]
        speaker_id = row["participant_id"]
        feats = extract_all_features(
            audio_path, transcript=transcript, speaker_id=speaker_id)
        feats["label"] = emotion
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    y = feat_df["label"]
    # transcript is text, we already encoded parts of it
    X = feat_df.drop(columns=["label", "transcript"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=250, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    return clf, X.columns.tolist()


# train it
clf, feature_cols = train_multiclass_model_from_merged(merged_df)

# try live mic
live_predict(clf, feature_cols, seconds=4)
# %% prediction helper


def predict_emotion(clf, feature_columns, path: str, transcript: str = None, speaker_id: str = None):
    feats = extract_all_features(
        path, transcript=transcript, speaker_id=speaker_id)
    text = feats.pop("transcript")
    x_vec = [feats.get(col, 0.0) for col in feature_columns]
    proba = clf.predict_proba([x_vec])[0]
    pred_label = clf.classes_[np.argmax(proba)]
    return {
        "predicted_emotion": pred_label,
        "proba": {lab: float(p) for lab, p in zip(clf.classes_, proba)},
        "transcript": text,
        "raw_features": feats,
    }
