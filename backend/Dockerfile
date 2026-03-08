# ── Backend Dockerfile ────────────────────────────────────────
# Python 3.11 slim per immagine leggera
FROM python:3.11-slim

# Metadati
LABEL maintainer="rag-chatbot"
LABEL description="FastAPI backend con RAG pipeline"

# Evita prompt interattivi durante apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dipendenze di sistema (necessarie per PyMuPDF e sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia e installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-scarica il modello di embedding alla build
# (così non lo scarica ogni volta che parte il container)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copia il codice
COPY . .

# Esponi la porta FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Avvio con uvicorn (--reload solo in sviluppo)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
