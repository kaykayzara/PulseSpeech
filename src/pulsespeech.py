import os
import json
import warnings

import librosa
import parselmouth
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================

# RAVDESS base directory – adjust only if you moved the dataset
BASE_RAVDESS_DIR = r"C:/Users/kayre/CBD Agitation/PulseSpeech/data/Audio_Speech_Actors_01-24"

# Keep these filenames so the GUI can still load the model/scaler
MODEL_PATH = "rf_emotion_model.joblib"
SCALER_PATH = "rf_scaler.joblib"
FEATURE_COLS_PATH = "rf_feature_cols.json"

RANDOM_STATE = 42


# =========================
# DATA LOADING
# =========================

def load_ravdess(base_dir):
    """
    Walk the RAVDESS directory and return a DataFrame with:
    - audio_path
    - emotion (original 8-category label)
    """
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

    return pd.DataFrame({"audio_path": audio_paths, "emotion": labels})


# =========================
# EMOTION GROUPING
# =========================

def map_emotion_group(e):
    """
    Map the 8 original RAVDESS emotions into 4 grouped classes
    to make them more separable for classical ML:

        positive: happy, calm
        negative: sad, disgust
        angry   : angry
        fearful : fearful, surprised

    neutral is mapped to None and dropped from training to avoid
    hurting accuracy (it's essentially "no emotion" / baseline).
    """
    if e in ["happy", "calm"]:
        return "positive"
    if e in ["sad", "disgust"]:
        return "negative"
    if e == "angry":
        return "angry"
    if e in ["fearful", "surprised"]:
        return "fearful"
    # neutral or anything else -> drop from training
    return None


# =========================
# FEATURE EXTRACTION
# =========================

def extract_acoustic_features(audio_path):
    """
    Extract a rich set of acoustic features for emotion recognition.
    This is used both for offline training and (via the GUI) for live audio.
    """
    y, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    # Basic energy + zero-crossing
    rms = float(librosa.feature.rms(y=y).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())

    # Pitch via Parselmouth
    pitch = parselmouth.Sound(
        audio_path).to_pitch().selected_array["frequency"]
    pitch_vals = pitch[pitch > 0]
    pitch_mean = float(pitch_vals.mean()) if len(pitch_vals) else 0.0
    pitch_std = float(pitch_vals.std()) if len(pitch_vals) else 0.0

    # MFCCs + first and second order deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    feats = {
        "duration": duration,
        "rms": rms,
        "zcr": zcr,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
    }

    # MFCC stats
    for i, v in enumerate(mfcc.mean(axis=1)):
        feats[f"mfcc_mean_{i}"] = float(v)
    for i, v in enumerate(mfcc.std(axis=1)):
        feats[f"mfcc_std_{i}"] = float(v)

    # MFCC deltas
    for i, v in enumerate(mfcc_delta.mean(axis=1)):
        feats[f"mfcc_delta_mean_{i}"] = float(v)
    for i, v in enumerate(mfcc_delta2.mean(axis=1)):
        feats[f"mfcc_delta2_mean_{i}"] = float(v)

    # Chroma (harmonic energy over 12 pitch classes)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i, v in enumerate(chroma.mean(axis=1)):
        feats[f"chroma_{i}"] = float(v)

    # Spectral contrast (harmonic vs noise energy)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i, v in enumerate(contrast.mean(axis=1)):
        feats[f"contrast_{i}"] = float(v)

    # Tonnetz (tonal centroid features)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    for i, v in enumerate(tonnetz.mean(axis=1)):
        feats[f"tonnetz_{i}"] = float(v)

    return feats


# =========================
# BUILD FEATURE DATASET
# =========================

def build_feature_dataframe(df):
    """
    Given df with columns [audio_path, emotion], compute all features
    and return a new DataFrame with numeric feature columns + 'emotion'.
    """
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
    # 1. Load original data
    df_raw = load_ravdess(BASE_RAVDESS_DIR)

    # 2. Map to grouped emotion classes and drop neutral / unmapped
    df_raw["emotion"] = df_raw["emotion"].apply(map_emotion_group)
    df_raw = df_raw.dropna(subset=["emotion"]).reset_index(drop=True)

    print("Label distribution after grouping (training classes):")
    print(df_raw["emotion"].value_counts())

    # 3. Extract features
    feat_df = build_feature_dataframe(df_raw)

    print("Number of samples after feature extraction:", len(feat_df))
    print("Number of feature columns:", len(feat_df.columns) - 1)

    # 4. Split X / y
    y = feat_df["emotion"]
    X = feat_df.drop(columns=["emotion"])

    # Shuffle to avoid any ordering bias
    X, y = shuffle(X, y, random_state=RANDOM_STATE)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 5. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Train SVC (RBF) with class balancing
    clf = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,      # keep probs if you want later
        random_state=RANDOM_STATE,
    )

    print("\n🚀 Training SVC (RBF) model...")
    clf.fit(X_train_scaled, y_train)

    # 7. Evaluate
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print("\n📊 Evaluation on held-out test set:")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    # 8. Save model, scaler, and feature column order for GUI
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(list(X.columns), f)

    print("\n✅ Model, scaler, and feature columns saved:")
    print("   Model ->", MODEL_PATH)
    print("   Scaler ->", SCALER_PATH)
    print("   Feature cols ->", FEATURE_COLS_PATH)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    train_model()
