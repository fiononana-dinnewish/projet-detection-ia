
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    model_version: str
