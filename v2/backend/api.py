"""
Backend RAG v2.

Questa versione introduce tre miglioramenti principali rispetto alla v1:
1. collection separate per documenti e configurazione HA
2. retrieval multi-source con oversampling e deduplica
3. prompt più trasparente e meno incline a risposte non grounded
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import chromadb
import ollama
from chromadb.utils import embedding_functions
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Configurazione da environment ──────────────────────────────────────────
CHROMA_HOST = os.getenv("CHROMA_HOST", "http://chromadb:8000")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
DOCS_COLLECTION = os.getenv("DOCS_COLLECTION", "documents")
CONFIG_COLLECTION = os.getenv("CONFIG_COLLECTION", "ha_config")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.1:8b")
TOP_K = int(os.getenv("TOP_K", "10"))
RETRIEVAL_OVERSAMPLE = int(os.getenv("RETRIEVAL_OVERSAMPLE", "3"))
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.30"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


def parse_allowed_origins() -> list[str]:
    """Converte la lista CSV di origin in una lista Python pulita."""
    raw = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


ALLOWED_ORIGINS = parse_allowed_origins()

app = FastAPI(title="RAG HA Agent Backend v2", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Admin-Token"],
)


@dataclass
class SearchHit:
    """Struttura interna per rappresentare un chunk recuperato."""

    text: str
    source: str
    page: int
    score: float
    collection: str
    chunk_id: str


class ChatRequest(BaseModel):
    """Payload della chat RAG v2."""

    question: str = Field(..., min_length=1)
    top_k: int = Field(default=TOP_K, ge=1, le=20)


class SearchHitResponse(BaseModel):
    """Versione serializzabile del chunk da restituire al frontend."""

    text: str
    source: str
    page: int
    score: float
    collection: str


class ChatResponse(BaseModel):
    """Risposta strutturata della chat."""

    answer: str
    model: str
    chunks: list[SearchHitResponse]


class ReindexRequest(BaseModel):
    """Payload per lanciare l'indicizzazione della config HA."""

    config_root: str = "/ha_config"
    collection_name: str = CONFIG_COLLECTION


# ── Helper per Chroma e Ollama ─────────────────────────────────────────────
def get_chroma_client() -> chromadb.HttpClient:
    """Restituisce il client HTTP verso ChromaDB."""
    parsed = urlparse(CHROMA_HOST)
    return chromadb.HttpClient(host=parsed.hostname or "chromadb", port=parsed.port or 8000)



def get_embedding_function():
    """Embedding function condivisa tra collection diverse."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)



def get_collection(name: str):
    """Restituisce o crea una collection con spazio vettoriale cosine."""
    return get_chroma_client().get_or_create_collection(
        name=name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )



def get_ollama() -> ollama.Client:
    """Client Ollama locale."""
    return ollama.Client(host=OLLAMA_HOST)


# ── Retrieval multi-source ─────────────────────────────────────────────────
def normalize_score(distance: float) -> float:
    """Converte una distance cosine in uno score intuitivo 0..1."""
    return round(1 - (distance / 2), 4)



def query_collection(collection_name: str, question: str, top_k: int) -> list[SearchHit]:
    """Recupera risultati da una singola collection con oversampling."""
    collection = get_collection(collection_name)
    requested = max(top_k * RETRIEVAL_OVERSAMPLE, top_k)
    results = collection.query(
        query_texts=[question],
        n_results=requested,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0] if results.get("ids") else [f"{collection_name}-{i}" for i, _ in enumerate(documents)]

    hits: list[SearchHit] = []
    for doc, meta, dist, chunk_id in zip(documents, metadatas, distances, ids):
        score = normalize_score(dist)
        if score < MIN_RETRIEVAL_SCORE:
            continue
        hits.append(
            SearchHit(
                text=doc,
                source=meta.get("source", "unknown"),
                page=int(meta.get("page", 0)),
                score=score,
                collection=collection_name,
                chunk_id=str(chunk_id),
            )
        )
    return hits



def deduplicate_hits(hits: Iterable[SearchHit]) -> list[SearchHit]:
    """Rimuove duplicati basandosi su collection, source e prefisso testuale."""
    seen: set[tuple[str, str, str]] = set()
    unique: list[SearchHit] = []
    for hit in sorted(hits, key=lambda item: item.score, reverse=True):
        signature = (hit.collection, hit.source, hit.text[:180])
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(hit)
    return unique



def retrieve_chunks(question: str, top_k: int) -> list[SearchHit]:
    """Interroga sia la collection documentale sia quella della config HA."""
    docs_hits = query_collection(DOCS_COLLECTION, question, top_k)
    config_hits = query_collection(CONFIG_COLLECTION, question, top_k)
    merged = deduplicate_hits([*docs_hits, *config_hits])
    return merged[:top_k]


# ── Prompt building ────────────────────────────────────────────────────────
def build_prompt(question: str, hits: list[SearchHit]) -> str:
    """Costruisce un prompt grounded e trasparente."""
    if not hits:
        context = "Nessun chunk rilevante trovato nelle collection indicizzate."
    else:
        parts = []
        for idx, hit in enumerate(hits, start=1):
            parts.append(
                f"[Fonte {idx} | collection={hit.collection} | score={hit.score:.0%} | source={Path(hit.source).name} | page={hit.page}]\n{hit.text}"
            )
        context = "\n\n---\n\n".join(parts)

    return f"""Sei un assistente tecnico specializzato in Home Assistant e RAG locale.

Regole:
1. Usa solo il contesto fornito.
2. Se il contesto è insufficiente, dichiaralo esplicitamente.
3. Distingui sempre tra evidenza osservata e inferenza tecnica.
4. Se citi un file o una configurazione, usa il source presente nel contesto.
5. Rispondi in italiano.

=== CONTESTO RECUPERATO ===
{context}

=== DOMANDA ===
{question}

=== RISPOSTA ===
"""


# ── Endpoint FastAPI ───────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    """Endpoint rapido per verificare lo stato del backend v2."""
    info = {
        "api": "ok",
        "model": LLAMA_MODEL,
        "docs_collection": DOCS_COLLECTION,
        "config_collection": CONFIG_COLLECTION,
        "allowed_origins": ALLOWED_ORIGINS,
        "admin_protected_endpoints": bool(ADMIN_TOKEN),
    }

    # Verifica di connettività minima con Chroma.
    try:
        docs_count = get_collection(DOCS_COLLECTION).count()
        config_count = get_collection(CONFIG_COLLECTION).count()
        info["chromadb"] = f"ok (documents={docs_count}, ha_config={config_count})"
    except Exception as exc:  # pragma: no cover - gestione operativa
        info["chromadb"] = f"error: {exc}"

    # Verifica minima con Ollama.
    try:
        get_ollama().list()
        info["ollama"] = f"ok ({LLAMA_MODEL})"
    except Exception as exc:  # pragma: no cover - gestione operativa
        info["ollama"] = f"error: {exc}"

    return info


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Esegue retrieval multi-source e genera una risposta grounded."""
    hits = retrieve_chunks(req.question, req.top_k)
    prompt = build_prompt(req.question, hits)

    try:
        response = get_ollama().chat(
            model=LLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "top_p": 0.9, "num_predict": 2048},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Errore Ollama: {exc}") from exc

    return ChatResponse(
        answer=response["message"]["content"],
        model=LLAMA_MODEL,
        chunks=[
            SearchHitResponse(
                text=hit.text,
                source=hit.source,
                page=hit.page,
                score=hit.score,
                collection=hit.collection,
            )
            for hit in hits
        ],
    )


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Upload semplice di documenti per la collection documentale.

    Nota: in questa proposta v2 l'upload non esegue parsing PDF avanzato.
    È pensato per file testuali e markdown, lasciando la pipeline PDF estesa a
    una fase successiva oppure al backend v1 già esistente.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".txt", ".md", ".markdown", ".yaml", ".yml", ".json"}:
        raise HTTPException(status_code=400, detail="Formato non supportato per l'upload v2.")

    content = await file.read()
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="File troppo grande.")

    text = content.decode("utf-8", errors="ignore")
    chunks = [text[i : i + 1200] for i in range(0, len(text), 1000)]

    collection = get_collection(DOCS_COLLECTION)
    ids = [f"upload::{file.filename}::{idx}" for idx, _ in enumerate(chunks)]
    metadatas = [
        {
            "source": file.filename or "upload",
            "page": idx,
            "source_kind": "upload",
        }
        for idx, _ in enumerate(chunks)
    ]
    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)

    return {"status": "ok", "filename": file.filename, "chunks": len(chunks), "collection": DOCS_COLLECTION}


@app.post("/admin/reindex-ha-config")
def reindex_ha_config(
    request: ReindexRequest,
    x_admin_token: str | None = Header(default=None),
) -> dict:
    """Lancia l'indicizzazione della cartella config di HA.

    Per semplicità operativa chiama la funzione locale del modulo di ingest.
    In un'evoluzione successiva si può trasformare in job asincrono.
    """
    if ADMIN_TOKEN and x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Admin token non valido.")

    from ingest_ha_config import ingest_home_assistant_config

    result = ingest_home_assistant_config(
        config_root=request.config_root,
        collection_name=request.collection_name,
        chroma_host=CHROMA_HOST,
        embed_model=EMBED_MODEL,
    )
    return result
