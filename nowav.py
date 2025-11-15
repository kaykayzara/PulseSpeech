import os
python - <<EOF
BASE_RAVDESS_DIR = r"C:/Users/kayre/CBD Agitation/PulseSpeech/data/Audio_Speech_Actors_01-24"
count = 0
for root, _, files in os.walk(BASE_RAVDESS_DIR):
    for f in files:
        if f.endswith(".wav"):
            count += 1
print("Number of audio files:", count)
EOF
