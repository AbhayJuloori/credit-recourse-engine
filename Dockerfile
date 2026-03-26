# Credit Recourse Engine — Docker image
# Optimised for Hugging Face Spaces deployment (free tier).

FROM python:3.9-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Artifacts — pre-trained models committed via Git LFS
# (backend/artifacts/*.pkl tracked in .gitattributes)
COPY backend/artifacts/ ./backend/artifacts/

# Port 7860 is the HuggingFace Spaces default
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
