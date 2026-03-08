FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent_api.py .

EXPOSE 8100

CMD ["uvicorn", "agent_api:app", "--host", "0.0.0.0", "--port", "8100", "--reload"]
