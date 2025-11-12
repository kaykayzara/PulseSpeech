import math
import sys
import os
import json
import pandas as pd
import numpy as np
import librosa
import parselmouth
import whisper
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
BASE_RAVDESS_DIR = r"C:/Users/kayre/CBD Agitation/PulseSpeech/data/Audio_Speech_Actors_01-24"
PROGRESS_PATH = "progress.json"
MODEL_PATH = "emotion_model.joblib"
FEATURE_COLS_PATH = "feature_cols.json"
SCALER_PATH = "scaler.joblib"
BATCH_SIZE = 50  # how many audio files to handle per run

# heavy models
WHISPER_MODEL = whisper.load_model("tiny")
SENT_MODEL = pipeline("sentiment-analysis")
SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# PROGRESS HELPERS
# =========================
def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed_paths": {}, "total_processed": 0}


def save_progress(progress):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


# =========================
# DATA LOADING
# =========================
def load_ravdess(base_dir):
    emotion_map = {
        1: "neutral", 2: "calm", 3: "happy", 4: "sad",
        5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
    }
    audio_paths, labels = [], []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".wav"):
                path = os.path.join(root, file)
                # filename looks like "03-01-01-01-01-01-01.wav"
                emotion_id = int(file.split("-")[2])
                audio_paths.append(path)
                labels.append(emotion_map.get(emotion_id, "unknown"))
    df = pd.DataFrame({
        "audio_path": audio_paths,
        "emotion": labels,
        "transcription": "",  # blank for now
        "participant_id": ["rav_" + os.path.basename(p) for p in audio_paths]
    })
    return df


def load_all_datasets():
    # for now we just load RAVDESS, same as before
    return load_ravdess(BASE_RAVDESS_DIR)


# =========================
# TRANSCRIPT SAVING
# =========================
def save_transcript_to_actor_folder(audio_path, transcript, emotion):
    """
    audio_path: .../Audio_Speech_Actors_01-24/Actor_01/xxx.wav
    we will save to: .../Audio_Speech_Actors_01-24/Actor_01/transcriptions (code)/xxx.txt
    """
    actor_dir = os.path.dirname(audio_path)
    trans_dir = os.path.join(actor_dir, "transcriptions (code)")
    os.makedirs(trans_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_path))[0] + ".txt"
    out_path = os.path.join(trans_dir, base_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"[Emotion: {emotion}]\n\n{transcript.strip()}")
    return out_path


# =========================
# FEATURE EXTRACTION
# =========================
def extract_acoustic_features(path):
    y, sr = librosa.load(path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    rms = float(librosa.feature.rms(y=y).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())

    pitch = parselmouth.Sound(path).to_pitch().selected_array["frequency"]
    pitch_vals = pitch[pitch > 0]
    pitch_mean = float(pitch_vals.mean()) if len(pitch_vals) else 0.0
    pitch_std = float(pitch_vals.std()) if len(pitch_vals) else 0.0

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feats = {
        "duration": duration,
        "rms": rms,
        "zcr": zcr,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
    }
    for i, v in enumerate(mfcc.mean(axis=1)):
        feats[f"mfcc_mean_{i}"] = float(v)
    for i, v in enumerate(mfcc.std(axis=1)):
        feats[f"mfcc_std_{i}"] = float(v)
    return feats, duration


def extract_text_features(text, duration):
    # sentiment
    sent = SENT_MODEL(text[:512])[0]
    sent_is_pos = 1 if "pos" in sent["label"].lower() else 0
    words = text.split()
    wps = len(words) / max(duration, 1e-6)
    emb = SBERT_MODEL.encode([text])[0]
    return {
        "sent_pos_label": sent_is_pos,
        "sent_score": float(sent["score"]),
        "words_per_sec": float(wps),
        "embed_mean": float(np.mean(emb)),
        "embed_std": float(np.std(emb)),
    }


def extract_features_for_row(row):
    audio_path = row["audio_path"]
    transcript = row.get("transcription", "")

    # transcribe if empty
    if not transcript:
        result = WHISPER_MODEL.transcribe(audio_path)
        transcript = result["text"]

    # save transcript to folder
    save_transcript_to_actor_folder(audio_path, transcript, row["emotion"])

    # extract audio + text features
    audio_feats, duration = extract_acoustic_features(audio_path)
    text_feats = extract_text_features(transcript, duration)

    feats = {**audio_feats, **text_feats}
    feats["label"] = row["emotion"]
    feats["transcript"] = transcript
    feats["audio_path"] = audio_path
    return feats


# =========================
# MODEL HELPERS
# =========================
def load_model_and_metadata():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_COLS_PATH) and os.path.exists(SCALER_PATH):
        clf = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
        return clf, scaler, feature_cols
    return None, None, None


def save_model_and_metadata(clf, scaler, feature_cols):
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_COLS_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)


# =========================
# TRAINING LOOP (INCREMENTAL)
# =========================
def incremental_train(df):
    progress = load_progress()
    processed_paths = set(progress["processed_paths"].keys())

    # filter to unprocessed rows
    unprocessed_df = df[~df["audio_path"].isin(processed_paths)]
    total_files = len(df)
    remaining = len(unprocessed_df)

    if remaining == 0:
        print("✅ All files have already been processed and trained on.")
        return

    # pick a batch
    batch_df = unprocessed_df.head(BATCH_SIZE)
    print(
        f"Processing {len(batch_df)} new files out of {remaining} remaining...")

    # extract features for this batch
    feat_rows = []
    for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Feature extraction"):
        try:
            feats = extract_features_for_row(row)
            feat_rows.append(feats)
        except Exception as e:
            print(f"⚠️ Skipping {row['audio_path']} due to error: {e}")

    if not feat_rows:
        print("No features extracted in this batch.")
        return

    feat_df = pd.DataFrame(feat_rows)

    # separate features/labels
    y_batch = feat_df["label"]
    X_batch = feat_df.drop(columns=["label", "transcript", "audio_path"])

    # load or init model
    clf, scaler, feature_cols = load_model_and_metadata()
    classes = sorted(df["emotion"].unique())

    if clf is None:
        # first run: we define feature columns from this batch
        feature_cols = list(X_batch.columns)
        scaler = StandardScaler()
        scaler.fit(X_batch[feature_cols])
        X_scaled = scaler.transform(X_batch[feature_cols])

        clf = SGDClassifier(loss="log_loss", random_state=42)
        # initial partial_fit must have classes
        clf.partial_fit(X_scaled, y_batch, classes=classes)
    else:
        # existing model: align columns
        # if new columns appear (shouldn't normally), fill with 0
        for col in feature_cols:
            if col not in X_batch.columns:
                X_batch[col] = 0.0
        X_batch = X_batch[feature_cols]  # same order

        # scale using existing scaler
        X_scaled = scaler.transform(X_batch)

        # incremental update
        clf.partial_fit(X_scaled, y_batch)

    # save updated model + metadata
    save_model_and_metadata(clf, scaler, feature_cols)

    # update progress
    for p in feat_df["audio_path"]:
        if p in progress["processed_paths"]:
            progress["processed_paths"][p] += 1
        else:
            progress["processed_paths"][p] = 1
        progress["total_processed"] += 1

    save_progress(progress)

    # print status
    trained_so_far = len(progress["processed_paths"])
    print(f"✅ Batch complete. Trained on {len(batch_df)} new files.")
    print(f"📊 Total unique files trained on: {trained_so_far}/{total_files}")


# =========================
# OPTIONAL: EVAL ON ALL PROCESSED
# =========================
def evaluate_on_all(df):
    # Only possible if we have model + scaler + feature cols
    clf, scaler, feature_cols = load_model_and_metadata()
    if clf is None:
        print("No model to evaluate.")
        return

    feat_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Re-extract for eval"):
        try:
            feats = extract_features_for_row(row)
            feat_rows.append(feats)
        except Exception:
            continue

    feat_df = pd.DataFrame(feat_rows)
    y_true = feat_df["label"]
    X = feat_df.drop(columns=["label", "transcript", "audio_path"])

    # align
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols]
    X_scaled = scaler.transform(X)

    y_pred = clf.predict(X_scaled)
    print(classification_report(y_true, y_pred))


# =========================


def prepare_test_batch(df, batch_size=5):
    progress = load_progress()
    trained_paths = set(progress["processed_paths"].keys())
    untrained = df[~df["audio_path"].isin(trained_paths)]
    if untrained.empty:
        print("✅ No remaining files left for testing.")
        return None

    os.makedirs("data", exist_ok=True)
    total_batches = math.ceil(len(untrained) / batch_size)
    batch_idx = len([f for f in os.listdir("data") if f.startswith("test")])
    start_idx = batch_idx * batch_size
    batch_df = untrained.iloc[start_idx:start_idx + batch_size]

    out_path = f"data/test{batch_idx+1}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in batch_df.iterrows():
            f.write(
                f"File: {row['audio_path']}\nEmotion: {row['emotion']}\nTranscription: (pending)\n\n")
    print(f"📝 Saved test batch info to {out_path}")
    return batch_df


def test_batch(df):
    clf, scaler, feature_cols = load_model_and_metadata()
    if clf is None:
        print("⚠️ No trained model found.")
        return

    progress = load_progress
    trained_paths = set(progress["processed_paths"].keys())
    untrained = df[~df["audio_path"].isin(trained_paths)]
    test_df = untrained.head(5)
    if test_df.empty:
        print("No untrained samples left.")
        return

    feat_rows = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing batch"):
        feats = extract_features_for_row(row)
        feat_rows.append(feats)
    feat_df = pd.DataFrame(feat_rows)

    y_true = feat_df["label"]
    X = feat_df.drop(columns=["label", "transcript", "audio_path"])
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols]
    X_scaled = scaler.transform(X)
    y_pred = clf.predict(X_scaled)

    out_path = "data/test_results.txt"
    with open(out_path, "a", encoding="utf-8") as f:
        for path, true, pred in zip(feat_df["audio_path"], y_true, y_pred):
            f.write(f"{os.path.basename(path)}: true={true}, predicted={pred}\n")
    print(f"✅ Test results saved to {out_path}")
    print(classification_report(y_true, y_pred))


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    df = load_all_datasets()

    if len(sys.argv) < 2:
        print("Usage: python pulsespeech.py [train|test|prepare-test]")
    else:
        cmd = sys.argv[1]
        if cmd == "train":
            incremental_train(df)
        elif cmd == "prepare-test":
            prepare_test_batch(df)
        elif cmd == "test":
            test_batch(df)
        else:
            print(f"Unknown command: {cmd}")
