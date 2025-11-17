import os
import json
import librosa
import parselmouth
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib

# =========================
# CONFIG
# =========================

BASE_RAVDESS_DIR = r"C:/Users/kayre/CBD Agitation/PulseSpeech/data/Audio_Speech_Actors_01-24"

MODEL_PATH = "rf_emotion_model.joblib"
SCALER_PATH = "rf_scaler.joblib"
FEATURE_COLS_PATH = "rf_feature_cols.json"

# =========================
# LOAD RAVDESS
# =========================


def load_ravdess(base_dir):
    emotion_map = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised",
    }

    audio_paths = []
    labels = []

    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".wav"):
                path = os.path.join(root, f)
                emotion_id = int(f.split("-")[2])
                audio_paths.append(path)
                labels.append(emotion_map[emotion_id])

    df = pd.DataFrame({"audio_path": audio_paths, "emotion": labels})
    return df


# =========================
# FEATURE EXTRACTION
# =========================


def extract_acoustic_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    # RMS + ZCR
    rms = float(librosa.feature.rms(y=y).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())

    # Pitch via Parselmouth
    pitch = parselmouth.Sound(
        audio_path).to_pitch().selected_array["frequency"]
    pitch_vals = pitch[pitch > 0]
    pitch_mean = float(pitch_vals.mean()) if len(pitch_vals) else 0.0
    pitch_std = float(pitch_vals.std()) if len(pitch_vals) else 0.0

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfcc.mean(axis=1)
    mfcc_stds = mfcc.std(axis=1)

    # build feature dict
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


# =========================
# BUILD FEATURE DATASET
# =========================


def build_feature_dataframe(df):
    rows = []
    print("🎧 Extracting acoustic features from all audio files...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        feats = extract_acoustic_features(row["audio_path"])
        feats["emotion"] = row["emotion"]
        rows.append(feats)

    return pd.DataFrame(rows)


# =========================
# TRAIN MODEL
# =========================


def train_model():
    df = load_ravdess(BASE_RAVDESS_DIR)

    # Extract features
    feat_df = build_feature_dataframe(df)

    print("COLUMNS:", feat_df.columns.tolist())
    print("FIRST ROW:", feat_df.head())

    # Separate X/y
    y = feat_df["emotion"]
    X = feat_df.drop(columns=["emotion"])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RF model
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print("📊 Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    # Save model + scaler + column order
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(list(X.columns), f)

    print("✅ Model saved!")


# =========================
# MAIN
# =========================


if __name__ == "__main__":
    train_model()
