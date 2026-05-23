"""
Backend RAG v2.

Questa versione introduce tre miglioramenti principali rispetto alla v1:
1. collection separate per documenti e configurazione HA
2. retrieval multi-source con oversampling e deduplica
3. prompt più trasparente e meno incline a risposte non grounded

Fix aggiuntivi:
- /config e /stats compatibili con il frontend
- nessuna chiamata al modello se non ci sono chunk rilevanti
- endpoint di upload compatibile anche con /upload-pdf
- endpoint di cancellazione DB/documenti per la UI
"""

from __future__ import annotations

import os
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import chromadb
import requests
import ollama
from rank_bm25 import BM25Okapi
try:
    from ingest_ha_config import ingest_home_assistant_config as _ingest_ha_config
except ImportError:
    _ingest_ha_config = None
class OllamaEmbeddingFunction:
    """Embedding via Ollama — usa nomic-embed-text o qualsiasi modello embed Ollama."""
    def __init__(self, model: str, host: str):
        self.model = model
        self.host = host.rstrip("/")

    def name(self) -> str:
        return f"ollama:{self.model}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        resp = requests.post(
            f"{self.host}/api/embed",
            json={"model": self.model, "input": input},
            timeout=120,
        )
        resp.raise_for_status()
        import numpy as np

        return np.asarray(resp.json()["embeddings"], dtype="float32")

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from functools import lru_cache

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

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "600"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "4096"))

log = logging.getLogger("uvicorn.error")


def parse_allowed_origins() -> list[str]:
    """Converte la lista CSV di origin in una lista Python pulita."""
    raw = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


ALLOWED_ORIGINS = parse_allowed_origins()

# ── Smart text splitting (sentence-boundary aware) ─────────────────────────
def _split_text_smart(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary == -1:
                boundary = text.rfind("\n", start, end)
            if boundary != -1 and boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


# ── BM25 hybrid reranking ──────────────────────────────────────────────────
def _bm25_rerank_hits(
    query: str, hits: list[SearchHit], alpha: float = 0.6
) -> list[SearchHit]:
    if len(hits) <= 1:
        return hits
    tokenized = [h.text.lower().split() for h in hits]
    bm25 = BM25Okapi(tokenized)
    raw = bm25.get_scores(query.lower().split())
    max_raw = max(raw) if max(raw) > 0 else 1.0
    scored = [
        (alpha * h.score + (1 - alpha) * (raw[i] / max_raw), h)
        for i, h in enumerate(hits)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored]


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
    grounded: bool = True


class ReindexRequest(BaseModel):
    """Payload per lanciare l'indicizzazione della config HA."""
    config_root: str = "/ha_config"
    collection_name: str = CONFIG_COLLECTION


class DeleteAllResponse(BaseModel):
    """Risposta dello svuotamento DB."""
    chunks_deleted: int
    collections_cleared: list[str]


# ── Helper per Chroma e Ollama ─────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.HttpClient:
    """Restituisce un solo client HTTP verso ChromaDB, riusato tra le chiamate."""
    parsed = urlparse(CHROMA_HOST)
    return chromadb.HttpClient(
        host=parsed.hostname or "chromadb",
        port=parsed.port or 8000,
    )


def get_embedding_function():
    """Embedding function condivisa tra collection diverse."""
    return OllamaEmbeddingFunction(model=EMBED_MODEL, host=OLLAMA_HOST)


def get_collection(name: str):
    """Restituisce o crea una collection con spazio vettoriale cosine."""
    return get_chroma_client().get_or_create_collection(
        name=name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

def get_raw_chroma_client() -> chromadb.HttpClient:
    """Client Chroma senza embedding function, utile per operazioni admin."""
    parsed = urlparse(CHROMA_HOST)
    return chromadb.HttpClient(
        host=parsed.hostname or "chromadb",
        port=parsed.port or 8000,
    )


def reset_collection(collection_name: str) -> dict:
    """
    Elimina e ricrea una collection Chroma.

    Utile per fare reset pulito di documenti o config prima di una nuova ingestione.
    """
    client = get_raw_chroma_client()

    existing = {c.name for c in client.list_collections()}
    existed_before = collection_name in existing

    if existed_before:
        client.delete_collection(collection_name)

    # Ricrea la collection con la stessa configurazione attesa dal backend
    client.get_or_create_collection(
        name=collection_name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    return {
        "status": "ok",
        "collection": collection_name,
        "existed_before": existed_before,
        "reset": True,
    }

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
    ids = (
        results.get("ids", [[]])[0]
        if results.get("ids")
        else [f"{collection_name}-{i}" for i, _ in enumerate(documents)]
    )

    hits: list[SearchHit] = []
    for doc, meta, dist, chunk_id in zip(documents, metadatas, distances, ids):
        meta = meta or {}
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
    """Interroga sia la collection documentale sia quella della config HA, poi riordina con BM25."""
    docs_hits = query_collection(DOCS_COLLECTION, question, top_k)
    config_hits = query_collection(CONFIG_COLLECTION, question, top_k)
    merged = deduplicate_hits([*docs_hits, *config_hits])
    reranked = _bm25_rerank_hits(question, merged)
    return reranked[:top_k]


# ── Prompt building ────────────────────────────────────────────────────────
def build_prompt(question: str, hits: list[SearchHit]) -> str:
    """Costruisce un prompt grounded, trasparente, con few-shot examples."""
    parts = []
    for idx, hit in enumerate(hits, start=1):
        parts.append(
            f"[Fonte {idx} | {Path(hit.source).name} | score={hit.score:.0%} | tipo={hit.collection} | pag.{hit.page}]\n{hit.text}"
        )
    context = "\n\n---\n\n".join(parts) if parts else "Nessun documento rilevante trovato."

    return f"""Sei un esperto tecnico di Home Assistant. Rispondi SEMPRE in italiano.

Le fonti sono ordinate per rilevanza decrescente (Fonte 1 = più pertinente, già pesata con BM25 + similarità semantica).

REGOLE ASSOLUTE:
1. Usa SOLO le informazioni presenti nelle fonti fornite.
2. Per ogni affermazione fattuale cita la fonte: [Fonte N - nomefile].
3. Se l'informazione NON è nelle fonti, scrivi: "Non ho dati su questo nel contesto fornito."
4. Non inventare configurazioni, entità, valori o comportamenti non presenti nelle fonti.
5. Distingui certezza (presente nelle fonti) da ipotesi (usa "probabilmente", "potrebbe").
6. Mostra configurazioni YAML in blocchi ```yaml ... ```.
7. Rispondi in modo strutturato: prima risposta diretta, poi dettagli.

ESEMPIO CORRETTO:
  Domanda: "Come si configura il sensore di temperatura?"
  Risposta: "Dalla [Fonte 1 - configuration.yaml] la configurazione esistente è:
  ```yaml
  sensor:
    - platform: template
  ```
  Per aggiungere il sensore, devi..."

ESEMPIO SBAGLIATO:
  "Devi usare la piattaforma mqtt." (se mqtt non è nelle fonti)

=== FONTI (ordinate per rilevanza, Fonte 1 = più rilevante) ===
{context}

=== DOMANDA ===
{question}

=== RISPOSTA ==="""


# ── Utility statistiche ────────────────────────────────────────────────────
def estimate_db_size_mb(total_chunks: int) -> float:
    """Stima semplice della dimensione DB per la UI."""
    estimated_bytes = total_chunks * 1500
    return round(estimated_bytes / (1024 * 1024), 2)


def list_docs_documents() -> list[dict]:
    """Costruisce una lista semplificata dei documenti caricati nella collection docs."""
    collection = get_collection(DOCS_COLLECTION)
    raw = collection.get(include=["metadatas"])

    metadatas = raw.get("metadatas", []) or []
    grouped: dict[str, dict] = {}

    for meta in metadatas:
        meta = meta or {}
        source = meta.get("source", "unknown")
        grouped.setdefault(
            source,
            {
                "name": source,
                "chunks": 0,
                "pages": 0,
            },
        )
        grouped[source]["chunks"] += 1
        grouped[source]["pages"] = max(grouped[source]["pages"], int(meta.get("page", 0)) + 1)

    return sorted(grouped.values(), key=lambda item: item["name"].lower())


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

    try:
        docs_count = get_collection(DOCS_COLLECTION).count()
        config_count = get_collection(CONFIG_COLLECTION).count()
        total_chunks = docs_count + config_count
        info["chromadb"] = f"ok ({total_chunks} chunks)"
        info["chromadb_details"] = {
            "documents_chunks": docs_count,
            "config_chunks": config_count,
        }
    except Exception as exc:
        info["chromadb"] = f"error: {exc}"

    try:
        tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        tags.raise_for_status()
        payload = tags.json() if tags.text else {}
        model_names = []
        for item in payload.get("models", []):
            name = item.get("name")
            if name:
                model_names.append(name)

        if LLAMA_MODEL in model_names:
            info["ollama"] = f"ok (modello '{LLAMA_MODEL}' pronto)"
        else:
            info["ollama"] = f"warn (modello '{LLAMA_MODEL}' non presente)"
            info["ollama_models"] = model_names
    except Exception as exc:
        info["ollama"] = f"error: {exc}"

    return info


@app.get("/config")
def config() -> dict:
    """Config minima compatibile con il frontend."""
    return {
        "llm_model": LLAMA_MODEL,
        "embed_model": EMBED_MODEL,
        "top_k": TOP_K,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "docs_collection": DOCS_COLLECTION,
        "config_collection": CONFIG_COLLECTION,
    }


@app.get("/stats")
def stats() -> dict:
    """Statistiche semplici per alimentare la sidebar del frontend."""
    docs_chunks = get_collection(DOCS_COLLECTION).count()
    config_chunks = get_collection(CONFIG_COLLECTION).count()
    total_chunks = docs_chunks + config_chunks
    documents = list_docs_documents()

    return {
        "documents_chunks": docs_chunks,
        "config_chunks": config_chunks,
        "total_chunks": total_chunks,
        "total_documents": len(documents),
        "estimated_size_mb": estimate_db_size_mb(total_chunks),
        "documents": documents,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Esegue retrieval multi-source e genera una risposta grounded.

    Se non esistono chunk rilevanti, NON chiama Ollama e restituisce
    una risposta onesta e controllata.
    """
    started = time.perf_counter()
    hits = retrieve_chunks(req.question, req.top_k)
    retrieval_ms = int((time.perf_counter() - started) * 1000)

    if not hits:
        return ChatResponse(
            answer=(
                "Non ho trovato contenuti rilevanti nelle collection indicizzate.\n\n"
                "Per ottenere risposte contestuali devi prima:\n"
                "- caricare documenti nella collection documentale, oppure\n"
                "- indicizzare la configurazione di Home Assistant.\n\n"
                "In questo momento non posso rispondere in modo grounded alla tua domanda."
            ),
            model=LLAMA_MODEL,
            chunks=[],
            grounded=False,
        )

    prompt = build_prompt(req.question, hits)

    try:
        llm_started = time.perf_counter()
        response = get_ollama().chat(
            model=LLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": LLM_NUM_PREDICT,
                "num_ctx": LLM_NUM_CTX,
            },
        )
        llm_ms = int((time.perf_counter() - llm_started) * 1000)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Errore Ollama: {exc}") from exc

    total_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "chat completed model=%s hits=%s retrieval_ms=%s llm_ms=%s total_ms=%s",
        LLAMA_MODEL,
        len(hits),
        retrieval_ms,
        llm_ms,
        total_ms,
    )

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
        grounded=True,
    )


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Upload semplice di documenti per la collection documentale."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".txt", ".md", ".markdown", ".yaml", ".yml", ".json", ".pdf"}:
        raise HTTPException(status_code=400, detail="Formato non supportato per l'upload v2.")

    content = await file.read()
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="File troppo grande.")

    text = content.decode("utf-8", errors="ignore")
    if not text.strip():
        text = f"File caricato: {file.filename or 'upload'}"

    chunks = _split_text_smart(text, CHUNK_SIZE, CHUNK_OVERLAP)

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

    return {
        "status": "ok",
        "filename": file.filename,
        "chunks": len(chunks),
        "collection": DOCS_COLLECTION,
    }


@app.post("/upload-pdf")
async def upload_pdf_compat(file: UploadFile = File(...)) -> dict:
    """Alias compatibile con il frontend esistente."""
    return await upload_document(file)


@app.delete("/documents/all", response_model=DeleteAllResponse)
def delete_all_documents(x_admin_token: str | None = Header(default=None)) -> DeleteAllResponse:
    """Svuota completamente le collection principali."""
    if ADMIN_TOKEN and x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Admin token non valido.")

    docs = get_collection(DOCS_COLLECTION)
    cfg = get_collection(CONFIG_COLLECTION)

    docs_deleted = docs.count()
    cfg_deleted = cfg.count()

    docs_raw = docs.get(include=[])
    cfg_raw = cfg.get(include=[])

    docs_ids = docs_raw.get("ids", []) or []
    cfg_ids = cfg_raw.get("ids", []) or []

    if docs_ids:
        docs.delete(ids=docs_ids)
    if cfg_ids:
        cfg.delete(ids=cfg_ids)

    return DeleteAllResponse(
        chunks_deleted=docs_deleted + cfg_deleted,
        collections_cleared=[DOCS_COLLECTION, CONFIG_COLLECTION],
    )


@app.delete("/document/{doc_name}")
def delete_document(doc_name: str, x_admin_token: str | None = Header(default=None)) -> dict:
    """Elimina tutti i chunk di un singolo documento dalla collection docs."""
    if ADMIN_TOKEN and x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Admin token non valido.")

    collection = get_collection(DOCS_COLLECTION)
    raw = collection.get(include=["metadatas"])
    ids = raw.get("ids", []) or []
    metas = raw.get("metadatas", []) or []

    to_delete = []
    for chunk_id, meta in zip(ids, metas):
        meta = meta or {}
        if meta.get("source") == doc_name:
            to_delete.append(chunk_id)

    if not to_delete:
        raise HTTPException(status_code=404, detail="Documento non trovato.")

    collection.delete(ids=to_delete)
    return {"status": "ok", "document": doc_name, "chunks_deleted": len(to_delete)}

@app.delete("/admin/reset-documents-index")
def reset_documents_index(
    x_admin_token: str | None = Header(default=None),
) -> dict:
    """
    Reset pulito della collection documentale.
    """
    if ADMIN_TOKEN and x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Admin token non valido.")

    try:
        return reset_collection(DOCS_COLLECTION)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Errore reset collection '{DOCS_COLLECTION}': {exc}",
        ) from exc

@app.delete("/admin/reset-ha-config-index")
def reset_ha_config_index(
    x_admin_token: str | None = Header(default=None),
) -> dict:
    """
    Reset pulito della sola collection di configurazione HA.
    """
    if ADMIN_TOKEN and x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Admin token non valido.")

    try:
        return reset_collection(CONFIG_COLLECTION)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Errore reset collection '{CONFIG_COLLECTION}': {exc}",
        ) from exc

@app.post("/admin/reindex-ha-config")
def reindex_ha_config(
    request: ReindexRequest,
    x_admin_token: str | None = Header(default=None),
) -> dict:
    """Lancia l'indicizzazione della cartella config di HA."""
    if ADMIN_TOKEN and x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Admin token non valido.")

    if _ingest_ha_config is None:
        raise HTTPException(status_code=501, detail="Modulo ingest_ha_config non disponibile.")

    result = _ingest_ha_config(
        config_root=request.config_root,
        collection_name=request.collection_name,
        chroma_host=CHROMA_HOST,
        embed_model=EMBED_MODEL,
    )
    return result
