import logging

from fastapi import APIRouter, UploadFile, File, HTTPException

from api.schemas.prediction import PredictionResponse

from api.services.model_service import predict_image
from PIL import Image
import io

router = APIRouter()
logger = logging.getLogger(__name__)


ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


@router.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(
    file: UploadFile = File(...)
):
    logger.info(
        "Prediction request: filename=%s content_type=%s",
        file.filename,
        file.content_type,
    )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(
            "Rejected unsupported file type: %s",
            file.content_type,
        )

        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Use JPEG, PNG or WEBP.",
        )

    content = await file.read()

    if not content:
        logger.warning(
            "Rejected empty image: %s",
            file.filename,
        )

        raise HTTPException(
            status_code=400,
            detail="Image file is empty.",
        )

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire le fichier comme une image.")

    try:
        label, confidence = predict_image(image)
    except FileNotFoundError:
        logger.error("Fichier modèle introuvable")
        raise HTTPException(status_code=503, detail="Modèle non disponible.")

    return PredictionResponse(label=label, confidence=confidence, model_version="v1")
