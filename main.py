import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
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
# PREDICTION ENDPOINT
# =====================================================

@app.post("/predict", tags=["Prediction"])
async def predict(audio_file: UploadFile = File(...)):

    # 1. Vérification fichier reçu
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # 2. Vérification extension
    suffix = Path(audio_file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Format non supporté : '{suffix}'. "
                   f"Formats acceptés : {sorted(ALLOWED_EXTENSIONS)}",
        )

    # 3. Création fichier temporaire
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    try:
        # Sauvegarde fichier
        with tmp_path.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

       # 4. Vérification fichier vide ou trop petit
        file_size = tmp_path.stat().st_size
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        
        # ✅ AJOUT : header OGG/MP3 seul sans audio = généralement < 4KB
        if file_size < 4096:
            raise HTTPException(status_code=400, detail="Audio file is too small or empty")

        # 5. Appel modèle
        result = predict_emotion(str(tmp_path))

        # ✅ Vérification : si predict retourne une erreur métier
        if "error" in result and result["error"]:
            raise HTTPException(
                status_code=422,
                detail=f"Prediction failed: {result['error']}"
            )

        return JSONResponse(
            status_code=200,
            content={
                "success":    True,
                "filename":   audio_file.filename,
                "prediction": result["emotion"],
                "confidence": result["confidence"],
                "details":    result["all_probabilities"],  # ✅ clé maintenant présente
            },
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(exc)}",
        )

    finally:
        # 6. Nettoyage
        if tmp_path.exists():
            tmp_path.unlink()