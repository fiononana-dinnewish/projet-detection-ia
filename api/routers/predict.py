
from fastapi import APIRouter, UploadFile, File

from api.schemas.prediction import PredictionResponse


router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(
    file: UploadFile = File(...)
):
    return PredictionResponse(
        label="real",
        confidence=0.5,
        model_version="v0-stub"
    )
