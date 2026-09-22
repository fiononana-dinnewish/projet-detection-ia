import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routers.predict import router as predict_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # api/main.py -> racine du projet
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(PROJECT_ROOT / "monitoring" / "reports")))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Détection Images Retouchées/Générées par IA"
)


@app.get("/")
def health_check():
    logger.info("Health check requested")

    return {
        "status": "ok",
        "service": "detection-api"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "Unhandled error on %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error"
        },
    )

@app.get("/monitoring/list")
def list_monitoring_reports():
    reports = sorted(REPORTS_DIR.glob("*.html"), reverse=True)  # le plus récent en premier
    return {
        "reports": [f"http://localhost:8000/monitoring/{r.name}" for r in reports],
        "latest": reports[0].name if reports else None,
    }

app.include_router(predict_router)
app.mount("/monitoring", StaticFiles(directory=REPORTS_DIR, html=True), name="monitoring")