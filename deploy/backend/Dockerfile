FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    REPORTS_DIR=/tmp/reports \
    MODEL_PATH=/app/models/best_model.pth \
    OMP_NUM_THREADS=1
WORKDIR /app

COPY --chown=user api/requirements-api.txt ./requirements-api.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-api.txt

COPY --chown=user api ./api
COPY --chown=user src ./src
COPY --chown=user models ./models

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]