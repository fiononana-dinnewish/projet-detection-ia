
from fastapi import FastAPI

from api.routers.predict import router as predict_router


app = FastAPI(
    title="Détection Images Retouchées/Générées par IA"
)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "detection-api"
    }


app.include_router(predict_router)
