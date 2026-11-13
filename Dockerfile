# One image for every target:
# - default: API + UI in one container, UI on $PORT (Hugging Face Spaces, Render, Railway...)
# - docker-compose: runs the API and the UI as two services from this same image
FROM python:3.11-slim

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
ENV PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_XET=1 \
    FASTEMBED_CACHE_PATH=/home/user/models \
    PORT=7860 \
    LLM_BASE_URL=https://api.groq.com/openai/v1 \
    LLM_MODEL=openai/gpt-oss-120b

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
# Bake the embedding model into the image: no ~120 MB download when a container starts.
ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBEDDING_MODEL}')"

COPY --chown=user . .
EXPOSE 7860
CMD ["sh", "deploy/start.sh"]
