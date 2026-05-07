import uuid
import shutil
import tempfile
from pathlib import Path
from io import BytesIO

import numpy as np
import soundfile as sf

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_emotion

# =====================================================
# CONFIG
# =====================================================

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
UPLOAD_DIR = Path("tmp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Emotion Detection API",
    description="Détecte l'émotion dans un fichier audio : Angry | Happy | Neutral | Sad",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Emotion Detection API is running"}


# =====================================================
# PREDICTION ENDPOINT (fichier)
# =====================================================

@app.post("/predict", tags=["Prediction"])
async def predict(audio_file: UploadFile = File(...)):

    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    suffix = Path(audio_file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Format non supporté : '{suffix}'. "
                   f"Formats acceptés : {sorted(ALLOWED_EXTENSIONS)}",
        )

    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    try:
        with tmp_path.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        file_size = tmp_path.stat().st_size
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        if file_size < 4096:
            raise HTTPException(status_code=400, detail="Audio file is too small or empty")

        result = predict_emotion(str(tmp_path))

        if "error" in result and result["error"]:
            raise HTTPException(status_code=422, detail=f"Prediction failed: {result['error']}")

        return JSONResponse(
            status_code=200,
            content={
                "success":    True,
                "filename":   audio_file.filename,
                "prediction": result["emotion"],
                "confidence": result["confidence"],
                "details":    result["all_probabilities"],
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# =====================================================
# WEBSOCKET — PRÉDICTION EN TEMPS RÉEL
# =====================================================
# Protocole :
#   Client → serveur : bytes bruts PCM float32 (16 kHz, mono)
#                      OU message texte "ping" pour vérifier la connexion
#   Serveur → client : JSON  { emotion, confidence, all_probabilities }
#                      OU    { error: "..." }
# =====================================================

REALTIME_SR          = 16_000          # Hz attendu par predict_emotion
SEGMENT_SECONDS      = 3              # longueur de chaque fenêtre analysée
OVERLAP_SECONDS      = 1              # chevauchement (sliding window)
MIN_SAMPLES          = REALTIME_SR * SEGMENT_SECONDS
OVERLAP_SAMPLES      = REALTIME_SR * OVERLAP_SECONDS


@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connecté")

    # Buffer d'accumulation (float32)
    audio_buffer: list[float] = []

    try:
        while True:
            # On peut recevoir soit des bytes (audio) soit du texte (ping)
            message = await websocket.receive()

            # ── Texte : ping de keepalive ──────────────────────────────
            if message.get("type") == "websocket.receive" and message.get("text"):
                text = message["text"]
                if text == "ping":
                    await websocket.send_json({"type": "pong"})
                continue

            # ── Bytes : chunk audio PCM float32 ───────────────────────
            raw_bytes = message.get("bytes")
            if not raw_bytes:
                continue

            # Convertir bytes → float32
            chunk = np.frombuffer(raw_bytes, dtype=np.float32)
            audio_buffer.extend(chunk.tolist())

            # Attendre d'avoir assez de données
            if len(audio_buffer) < MIN_SAMPLES:
                # Informer le client de la progression
                progress = int(len(audio_buffer) / MIN_SAMPLES * 100)
                await websocket.send_json({"type": "buffering", "progress": progress})
                continue

            # Extraire la fenêtre courante
            window = np.array(audio_buffer[:MIN_SAMPLES], dtype=np.float32)

            # Conserver le chevauchement pour la prochaine fenêtre
            audio_buffer = audio_buffer[MIN_SAMPLES - OVERLAP_SAMPLES:]

            # Sauvegarder dans un fichier WAV temporaire
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                sf.write(tmp_path, window, REALTIME_SR, subtype="FLOAT")
                result = predict_emotion(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            # Renvoyer le résultat
            if result.get("error"):
                await websocket.send_json({"type": "error", "error": result["error"]})
            else:
                await websocket.send_json({
                    "type":              "prediction",
                    "emotion":           result["emotion"],
                    "confidence":        result["confidence"],
                    "all_probabilities": result["all_probabilities"],
                })

    except WebSocketDisconnect:
        print("[WS] Client déconnecté")
    except Exception as e:
        print(f"[WS] Erreur inattendue : {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass