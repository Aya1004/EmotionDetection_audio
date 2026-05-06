import sys
from pathlib import Path
from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import librosa
import librosa.display

from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# =====================================================
# PARAMÈTRES
# =====================================================

TARGET_SR       = 16_000
TARGET_DURATION = 4
MAX_SAMPLES     = TARGET_SR * TARGET_DURATION

N_FFT      = 1024
HOP_LENGTH = 512
IMG_SIZE   = 224

CLASS_NAMES = ["Angry", "Happy", "Neutral", "Sad"]

BASE_DIR = Path(__file__).resolve().parent
BEST_MODEL_PATH = BASE_DIR / "best_efficientnet_raw.pth"

# =====================================================
# DEVICE
# =====================================================

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =====================================================
# CHARGEMENT DU MODÈLE
# =====================================================

def _load_model(device: torch.device) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


# =====================================================
# PIPELINE AUDIO → IMAGE STFT
# =====================================================

def _load_audio(audio_path: str) -> np.ndarray:
    try:
        y, _ = librosa.load(audio_path, sr=TARGET_SR)

        if y is None or len(y) == 0:
            raise ValueError("Audio file is empty or unreadable")

        if len(y) < 100:
            raise ValueError("Audio file too short")

        if len(y) < MAX_SAMPLES:
            y = np.pad(y, (0, MAX_SAMPLES - len(y)))
        else:
            y = y[:MAX_SAMPLES]

        return y

    except Exception as e:
        raise ValueError(f"Audio loading failed: {str(e)}")


def _audio_to_pil(audio_path: str) -> Image.Image:
    y    = _load_audio(audio_path)
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    db   = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fig, ax = plt.subplots(figsize=(3, 3))
    librosa.display.specshow(db, sr=TARGET_SR, hop_length=HOP_LENGTH,
                             x_axis=None, y_axis=None, ax=ax)
    ax.axis("off")
    plt.tight_layout(pad=0)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    buf.seek(0)
    return Image.open(buf).convert("RGB")


# =====================================================
# TRANSFORM
# =====================================================

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# =====================================================
# FONCTION PRINCIPALE
# =====================================================

def predict_emotion(audio_path: str) -> dict:
    try:
        device = _get_device()
        model  = _load_model(device)

        image  = _audio_to_pil(audio_path)
        tensor = _transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            idx    = torch.argmax(probs).item()

        emotion    = CLASS_NAMES[idx]
        confidence = round(probs[idx].item() * 100, 2)

        # ✅ AJOUT : toutes les probabilités par classe
        all_probabilities = {
            CLASS_NAMES[i]: round(probs[i].item() * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return {
            "emotion":           emotion,
            "confidence":        confidence,
            "all_probabilities": all_probabilities,   # ✅ ajouté
        }

    except ValueError as e:
        return {
            "error":             str(e),
            "emotion":           None,
            "confidence":        None,
            "all_probabilities": {},                  # ✅ clé toujours présente
        }


# =====================================================
# POINT D'ENTRÉE CLI
# =====================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python predict.py <chemin_audio>")
        sys.exit(1)

    audio_path = sys.argv[1]

    if not Path(audio_path).exists():
        print(f"Fichier introuvable : {audio_path}")
        sys.exit(1)

    result = predict_emotion(audio_path)
    print(f"Émotion détectée : {result}")