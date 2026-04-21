FROM python:3.11-slim

WORKDIR /app

# Dipendenze minime. Qui non installiamo tool superflui per tenere il container
# semplice da leggere e più prevedibile in sede di demo.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
