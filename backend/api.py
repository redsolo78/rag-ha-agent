"""
RAG Chatbot - FastAPI Backend
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator
from urllib.parse import urlparse

import chromadb
import ollama
from chromadb.utils import embedding_functions
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ── Configurazione da env vars (Docker) ─────────────────────────────────────
CHROMA_HOST = os.getenv("CHROMA_HOST", "http://chromadb:8000")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
COLLECTION = os.getenv("COLLECTION", "documents")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3")
TRANSLATE_QUERY = os.getenv("TRANSLATE_QUERY", "false").lower() == "true"
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MIN_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.30"))
OVERSAMPLE_FACTOR = max(int(os.getenv("RETRIEVAL_OVERSAMPLE", "3")), 1)


def get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


TOP_K_DEFAULT = max(get_env_int("TOP_K", 10), 1)
CHUNK_SIZE_ESTIMATE = max(get_env_int("CHUNK_SIZE", 1200), 1)


def parse_allowed_origins() -> list[str]:
    raw = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


ALLOWED_ORIGINS = parse_allowed_origins()
SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

# ────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Chatbot API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Admin-Token"],
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def parse_host(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    return parsed.hostname or "localhost", parsed.port or 80


def require_admin_token(x_admin_token: str | None) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Operazione amministrativa disabilitata: ADMIN_TOKEN non configurato.",
        )
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Token amministrativo non valido.")


def get_collection():
    """ChromaDB via HTTP (Docker)."""
    host, port = parse_host(CHROMA_HOST)
    client = chromadb.HttpClient(host=host, port=port)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def get_ollama():
    """Ollama client con host da env var."""
    return ollama.Client(host=OLLAMA_HOST)


def clamp_top_k(value: int) -> int:
    return max(1, min(value, 20))


def normalize_filename(filename: str | None) -> str:
    if not filename:
        return "uploaded_file"
    return Path(filename).name


def metadata_source_name(meta: dict[str, Any] | None) -> str:
    if not meta:
        return "unknown"
    return str(meta.get("source", "unknown")).split("/")[-1]


# ── Modelli dati ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=TOP_K_DEFAULT, ge=1, le=20)
    stream: bool = False


class ChunkResult(BaseModel):
    text: str
    source: str
    page: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    chunks: list[ChunkResult]
    model: str


# ── Core logic ──────────────────────────────────────────────────────────────
def translate_query(question: str) -> str:
    """Traduce la query in inglese per documenti in inglese."""
    client_ollama = get_ollama()
    resp = client_ollama.chat(
        model=LLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Translate this to English, reply ONLY with the translation, "
                    f"no explanation: {question}"
                ),
            }
        ],
        options={"temperature": 0, "num_predict": 100},
    )
    return resp["message"]["content"].strip()


def retrieve_chunks(question: str, top_k: int) -> list[ChunkResult]:
    collection = get_collection()
    requested = clamp_top_k(top_k)
    raw_limit = min(max(requested * OVERSAMPLE_FACTOR, requested), 50)
    results = collection.query(
        query_texts=[question],
        n_results=raw_limit,
        include=["documents", "metadatas", "distances"],
    )

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    dedup_seen: set[tuple[str, int]] = set()
    chunks: list[ChunkResult] = []

    for doc, meta, dist in zip(docs, metas, distances):
        similarity_score = round(1 - (float(dist) / 2), 4)
        source_name = metadata_source_name(meta)
        page = int((meta or {}).get("page", 0) or 0)
        dedup_key = (source_name, page)
        if similarity_score < MIN_SCORE or dedup_key in dedup_seen:
            continue
        dedup_seen.add(dedup_key)
        chunks.append(
            ChunkResult(
                text=doc,
                source=source_name,
                page=page,
                score=similarity_score,
            )
        )
        if len(chunks) >= requested:
            break

    return chunks


def build_prompt(question: str, chunks: list[ChunkResult]) -> str:
    if not chunks:
        context = "Nessun documento rilevante trovato nel database."
    else:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Fonte {i} - File {chunk.source} - Pagina {chunk.page} - Rilevanza: {chunk.score:.0%}]\n{chunk.text}"
            )
        context = "\n\n---\n\n".join(context_parts)

    return f"""Sei un assistente esperto che risponde basandosi PRIORITARIAMENTE sui documenti forniti.

REGOLE:
1. Usa le informazioni nel contesto come fonte principale.
2. Se il contesto è insufficiente, dichiaralo esplicitamente invece di inventare dettagli.
3. Combina più fonti solo se sono coerenti tra loro.
4. Per comandi Linux includi descrizione, sintassi e almeno un esempio pratico se presenti nel contesto.
5. Rispondi nella stessa lingua della domanda.
6. Cita sempre file e pagina quando fai affermazioni fattuali prese dal contesto.

═══════════════════════════════════════
CONTESTO INDICIZZATO:
═══════════════════════════════════════
{context}

═══════════════════════════════════════
DOMANDA: {question}
═══════════════════════════════════════

RISPOSTA DETTAGLIATA:"""


# ── API ─────────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "La domanda non può essere vuota")

    query = translate_query(req.question) if TRANSLATE_QUERY else req.question
    chunks = retrieve_chunks(query, req.top_k)
    prompt = build_prompt(req.question, chunks)

    try:
        response = get_ollama().chat(
            model=LLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "top_p": 0.9, "num_predict": 2048},
        )
        answer = response["message"]["content"]
    except Exception as exc:
        raise HTTPException(500, f"Errore Llama: {exc}") from exc

    return ChatResponse(answer=answer, chunks=chunks, model=LLAMA_MODEL)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    query = translate_query(req.question) if TRANSLATE_QUERY else req.question
    chunks = retrieve_chunks(query, req.top_k)
    prompt = build_prompt(req.question, chunks)

    async def generate() -> AsyncGenerator[str, None]:
        import json

        yield (
            f"data: {json.dumps({'type': 'chunks', 'chunks': [c.model_dump() for c in chunks]})}\n\n"
        )
        try:
            stream = get_ollama().chat(
                model=LLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={"temperature": 0.1},
            )
            for part in stream:
                yield (
                    f"data: {json.dumps({'type': 'token', 'text': part['message']['content']})}\n\n"
                )
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    import json as _json

    filename = normalize_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            400,
            "Formato non supportato. Usa .pdf, .txt, .md o .markdown",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(400, "Il file caricato è vuoto")
    if len(payload) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File troppo grande. Limite: {MAX_UPLOAD_MB} MB")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(payload)
        tmp_path = tmp.name

    async def stream_progress():
        proc = await asyncio.create_subprocess_exec(
            "python",
            "ingest.py",
            "--pdf",
            tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path(__file__).parent),
        )

        yield f"data: {_json.dumps({'type':'progress','pct':5,'msg':'📄 Lettura documento...'})}\n\n"

        async for line in proc.stdout:
            text = line.decode("utf-8", errors="ignore").strip()
            if "Lettura e chunking" in text:
                yield f"data: {_json.dumps({'type':'progress','pct':10,'msg':'📄 Lettura e chunking...'})}\n\n"
            elif "Chunking completato" in text or "Chunk totali" in text:
                yield f"data: {_json.dumps({'type':'progress','pct':30,'msg':'✂️ Chunking completato'})}\n\n"
            elif "Creazione embedding" in text:
                yield f"data: {_json.dumps({'type':'progress','pct':40,'msg':'🔢 Creazione embedding...'})}\n\n"
            elif "chunks esistenti" in text:
                yield f"data: {_json.dumps({'type':'progress','pct':50,'msg':'💾 Connessione ChromaDB...'})}\n\n"
            elif "Salvataggio in ChromaDB" in text:
                yield f"data: {_json.dumps({'type':'progress','pct':60,'msg':'💾 Salvataggio embedding...'})}\n\n"
            elif "Completato" in text and "batch" in text:
                yield f"data: {_json.dumps({'type':'progress','pct':90,'msg':'⚡ Indicizzazione...'})}\n\n"
            elif "saltati" in text and "⚠️" in text:
                yield f"data: {_json.dumps({'type':'progress','pct':85,'msg':text})}\n\n"

        await proc.wait()
        Path(tmp_path).unlink(missing_ok=True)

        if proc.returncode != 0:
            err = (await proc.stderr.read()).decode("utf-8", errors="ignore")
            yield f"data: {_json.dumps({'type':'error','msg':f'Errore ingestion: {err[:300]}'})}\n\n"
        else:
            yield f"data: {_json.dumps({'type':'progress','pct':100,'msg':'✅ Completato!'})}\n\n"
            done_msg = f"File '{filename}' indicizzato con successo"
            yield f"data: {_json.dumps({'type':'done','message':done_msg})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_progress(), media_type="text/event-stream")


@app.get("/health")
async def health():
    status: dict[str, Any] = {
        "api": "ok",
        "chromadb": "unknown",
        "ollama": "unknown",
        "allowed_origins": ALLOWED_ORIGINS,
        "admin_protected_endpoints": bool(ADMIN_TOKEN),
    }

    try:
        col = get_collection()
        status["chromadb"] = f"ok ({col.count()} chunks)"
    except Exception as exc:
        status["chromadb"] = f"error: {exc}"

    try:
        result = get_ollama().list()
        model_names = [m.model for m in result.models]
        base_name = LLAMA_MODEL.split(":")[0]
        if any(base_name in model for model in model_names):
            status["ollama"] = f"ok (modello '{LLAMA_MODEL}' pronto)"
        else:
            status["ollama"] = f"connesso ma '{LLAMA_MODEL}' non trovato. Modelli: {model_names}"
    except Exception as exc:
        status["ollama"] = f"non raggiungibile: {exc}"

    return status


@app.get("/chunks")
async def list_chunks(limit: int = 10, offset: int = 0):
    try:
        col = get_collection()
        limit = max(1, min(limit, 100))
        offset = max(0, offset)
        results = col.get(limit=limit, offset=offset, include=["documents", "metadatas"])
        return {
            "total_chunks": col.count(),
            "chunks": [
                {
                    "id": id_,
                    "text": (doc[:200] + "...") if len(doc) > 200 else doc,
                    "metadata": meta,
                }
                for id_, doc, meta in zip(
                    results.get("ids", []),
                    results.get("documents", []),
                    results.get("metadatas", []),
                )
            ],
        }
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/config")
async def get_config():
    return {
        "llm_model": LLAMA_MODEL,
        "embed_model": EMBED_MODEL,
        "top_k": TOP_K_DEFAULT,
        "chunk_size": CHUNK_SIZE_ESTIMATE,
        "ollama_host": OLLAMA_HOST,
        "chroma_host": CHROMA_HOST,
        "translate_query": TRANSLATE_QUERY,
        "allowed_origins": ALLOWED_ORIGINS,
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.get("/stats")
async def get_stats():
    try:
        col = get_collection()
        total_chunks = col.count()
        batch_size = 500
        sources: dict[str, dict[str, Any]] = {}
        offset = 0

        while offset < total_chunks:
            batch = col.get(include=["metadatas"], limit=batch_size, offset=offset)
            for meta in (batch.get("metadatas") or []):
                src_name = metadata_source_name(meta)
                if src_name not in sources:
                    sources[src_name] = {"chunks": 0, "pages": set()}
                sources[src_name]["chunks"] += 1
                sources[src_name]["pages"].add((meta or {}).get("page", 0))
            offset += batch_size

        estimated_mb = round(total_chunks * CHUNK_SIZE_ESTIMATE / (1024 * 1024), 2)
        docs_detail = [
            {"name": src, "chunks": value["chunks"], "pages": len(value["pages"])}
            for src, value in sorted(sources.items())
        ]

        return {
            "total_chunks": total_chunks,
            "total_documents": len(sources),
            "estimated_size_mb": estimated_mb,
            "documents": docs_detail,
        }
    except Exception as exc:
        return {
            "total_chunks": 0,
            "total_documents": 0,
            "estimated_size_mb": 0,
            "documents": [],
            "error": str(exc),
        }


@app.delete("/document/{doc_name}")
async def delete_document(doc_name: str, x_admin_token: str | None = Header(default=None)):
    require_admin_token(x_admin_token)
    try:
        col = get_collection()
        results = col.get(include=["metadatas"])
        ids_to_delete = [
            id_
            for id_, meta in zip(results.get("ids", []), results.get("metadatas", []))
            if metadata_source_name(meta) == doc_name
        ]

        if not ids_to_delete:
            raise HTTPException(404, f"Documento '{doc_name}' non trovato nel DB")

        col.delete(ids=ids_to_delete)
        return {
            "status": "ok",
            "message": f"Documento '{doc_name}' eliminato",
            "chunks_deleted": len(ids_to_delete),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.delete("/documents/all")
async def delete_all_documents(x_admin_token: str | None = Header(default=None)):
    require_admin_token(x_admin_token)
    try:
        col = get_collection()
        total = col.count()
        host, port = parse_host(CHROMA_HOST)
        client = chromadb.HttpClient(host=host, port=port)
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        client.delete_collection(COLLECTION)
        client.create_collection(
            name=COLLECTION,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        return {"status": "ok", "chunks_deleted": total}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
