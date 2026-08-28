import logging

from fastapi import APIRouter, UploadFile, File, HTTPException

from api.schemas.prediction import PredictionResponse


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

    return PredictionResponse(
        label="real",
        confidence=0.5,
        model_version="v0-stub",
    )
