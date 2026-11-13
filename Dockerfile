FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_XET=1 \
    FASTEMBED_CACHE_PATH=/models
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Bake the embedding model into the image: no ~120 MB download when a container starts.
ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBEDDING_MODEL}')"
COPY . .
EXPOSE 8000 8501 7860
# Default: API only (docker-compose runs the UI as a second service).
# Single-container hosting (Hugging Face Spaces, Render...): CMD ["sh", "deploy/start.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
