"""
RAG Chatbot - FastAPI Backend
"""

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from typing import AsyncGenerator
from urllib.parse import urlparse

import chromadb
import ollama
from chromadb.utils import embedding_functions
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Configurazione da env vars (Docker) ──────────────────────
CHROMA_HOST  = os.getenv("CHROMA_HOST", "http://chromadb:8000")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://ollama:11434")
COLLECTION   = "documents"
EMBED_MODEL  = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
LLAMA_MODEL  = os.getenv("LLAMA_MODEL", "llama3")
TOP_K           = int(os.getenv("TOP_K", "10"))
TRANSLATE_QUERY = os.getenv("TRANSLATE_QUERY", "false").lower() == "true"
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_collection():
    """ChromaDB via HTTP (Docker)."""
    parsed = urlparse(CHROMA_HOST)
    client = chromadb.HttpClient(
        host=parsed.hostname or "chromadb",
        port=parsed.port or 8000
    )
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )


def get_ollama():
    """Ollama client con host da env var."""
    return ollama.Client(host=OLLAMA_HOST)


class ChatRequest(BaseModel):
    question: str
    top_k: int = TOP_K
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


def translate_query(question: str) -> str:
    """Traduce la query in inglese per documenti in inglese."""
    client_ollama = get_ollama()
    resp = client_ollama.chat(
        model=LLAMA_MODEL,
        messages=[{"role": "user", "content": f"Translate this to English, reply ONLY with the translation, no explanation: {question}"}],
        options={"temperature": 0, "num_predict": 100}
    )
    return resp["message"]["content"].strip()


def retrieve_chunks(question: str, top_k: int) -> list[ChunkResult]:
    collection = get_collection()
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        similarity_score = round(1 - (dist / 2), 4)
        chunks.append(ChunkResult(
            text=doc,
            source=meta.get("source", "unknown"),
            page=meta.get("page", 0),
            score=similarity_score
        ))
    return [c for c in chunks if c.score > 0.3]


def build_prompt(question: str, chunks: list[ChunkResult]) -> str:
    if not chunks:
        context = "Nessun documento rilevante trovato nel database."
    else:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Fonte {i} - Pagina {chunk.page} - Rilevanza: {chunk.score:.0%}]\n{chunk.text}"
            )
        context = "\n\n---\n\n".join(context_parts)

    return f"""Sei un assistente esperto che risponde a domande basandosi ESCLUSIVAMENTE sui documenti forniti.

REGOLE:
1. Usa TUTTE le informazioni nel contesto, anche se distribuite su più fonti
2. SINTETIZZA e COMBINA le informazioni per dare una risposta COMPLETA
3. Per comandi Linux includi SEMPRE: descrizione, sintassi, opzioni con esempi pratici
4. Elabora e spiega in dettaglio, non limitarti a citare
5. Rispondi nella stessa lingua della domanda

═══════════════════════════════════════
CONTESTO DAL PDF:
═══════════════════════════════════════
{context}

═══════════════════════════════════════
DOMANDA: {question}
═══════════════════════════════════════

RISPOSTA DETTAGLIATA:"""


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "La domanda non può essere vuota")

    query = translate_query(req.question) if TRANSLATE_QUERY else req.question
    chunks = retrieve_chunks(query, req.top_k)
    prompt = build_prompt(req.question, chunks)

    try:
        client_ollama = get_ollama()
        response = client_ollama.chat(
            model=LLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "top_p": 0.9, "num_predict": 2048}
        )
        answer = response["message"]["content"]
    except Exception as e:
        raise HTTPException(500, f"Errore Llama: {str(e)}")

    return ChatResponse(answer=answer, chunks=chunks, model=LLAMA_MODEL)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    query = translate_query(req.question) if TRANSLATE_QUERY else req.question
    chunks = retrieve_chunks(query, req.top_k)
    prompt = build_prompt(req.question, chunks)

    async def generate() -> AsyncGenerator[str, None]:
        import json
        yield f"data: {json.dumps({'type': 'chunks', 'chunks': [c.model_dump() for c in chunks]})}\n\n"
        try:
            stream = get_ollama().chat(
                model=LLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={"temperature": 0.1}
            )
            for part in stream:
                yield f"data: {json.dumps({'type': 'token', 'text': part['message']['content']})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    import json as _json
    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, "Formato non supportato. Usa .pdf, .txt o .md")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    filename = file.filename

    async def stream_progress():
        proc = await asyncio.create_subprocess_exec(
            "python", "ingest.py", "--pdf", tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path(__file__).parent)
        )

        # Fase 1: estrazione e chunking (0-20%)
        yield f"data: {_json.dumps({'type':'progress','pct':5,'msg':'📄 Lettura documento...'})}\n\n"

        stderr_lines = []
        async for line in proc.stdout:
            text = line.decode("utf-8", errors="ignore").strip()
            # Nuova barra: legge messaggi chiave da stdout
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
            yield f"data: {_json.dumps({'type':'error','msg':f'Errore ingestion: {err[:200]}'})}\n\n"
        else:
            yield f"data: {_json.dumps({'type':'progress','pct':100,'msg':'✅ Completato!'})}\n\n"
            done_msg = f"File '{filename}' indicizzato con successo"
            yield f"data: {_json.dumps({'type':'done','message':done_msg})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_progress(), media_type="text/event-stream")


@app.get("/health")
async def health():
    status = {"api": "ok", "chromadb": "unknown", "ollama": "unknown"}

    try:
        col = get_collection()
        status["chromadb"] = f"ok ({col.count()} chunks)"
    except Exception as e:
        status["chromadb"] = f"error: {e}"

    try:
        result = get_ollama().list()
        model_names = [m.model for m in result.models]
        base_name = LLAMA_MODEL.split(":")[0]
        if any(base_name in m for m in model_names):
            status["ollama"] = f"ok (modello '{LLAMA_MODEL}' pronto)"
        else:
            status["ollama"] = f"connesso ma '{LLAMA_MODEL}' non trovato. Modelli: {model_names}"
    except Exception as e:
        status["ollama"] = f"non raggiungibile: {e}"

    return status


@app.get("/chunks")
async def list_chunks(limit: int = 10, offset: int = 0):
    try:
        col = get_collection()
        results = col.get(limit=limit, offset=offset, include=["documents", "metadatas"])
        return {
            "total_chunks": col.count(),
            "chunks": [
                {"id": id_, "text": doc[:200] + "...", "metadata": meta}
                for id_, doc, meta in zip(results["ids"], results["documents"], results["metadatas"])
            ]
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/config")
async def get_config():
    """Ritorna la configurazione attuale del sistema — usata dalla UI."""
    return {
        "llm_model":   LLAMA_MODEL,
        "embed_model": EMBED_MODEL,
        "top_k":       TOP_K,
        "chunk_size":  int(os.getenv("CHUNK_SIZE", "1200")),
        "ollama_host": OLLAMA_HOST,
        "chroma_host": CHROMA_HOST,
        "translate_query": TRANSLATE_QUERY,
    }


@app.get("/stats")
async def get_stats():
    """Statistiche dettagliate del database — usate dalla UI."""
    try:
        col = get_collection()
        total_chunks = col.count()

        # Leggi metadati a batch per evitare il limite SQLite (999 variabili)
        BATCH = 500
        sources = {}
        offset = 0

        while offset < total_chunks:
            batch = col.get(include=["metadatas"], limit=BATCH, offset=offset)
            for m in (batch["metadatas"] or []):
                src_name = (m.get("source") or "unknown").split("/")[-1]
                if src_name not in sources:
                    sources[src_name] = {"chunks": 0, "pages": set()}
                sources[src_name]["chunks"] += 1
                sources[src_name]["pages"].add(m.get("page", 0))
            offset += BATCH

        avg_chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
        estimated_mb = round(total_chunks * avg_chunk_size / (1024 * 1024), 2)

        docs_detail = [
            {"name": src, "chunks": v["chunks"], "pages": len(v["pages"])}
            for src, v in sorted(sources.items())
        ]

        return {
            "total_chunks":      total_chunks,
            "total_documents":   len(sources),
            "estimated_size_mb": estimated_mb,
            "documents":         docs_detail
        }

    except Exception as e:
        return {
            "total_chunks": 0,
            "total_documents": 0,
            "estimated_size_mb": 0,
            "documents": [],
            "error": str(e)
        }


@app.delete("/document/{doc_name}")
async def delete_document(doc_name: str):
    """Elimina tutti i chunk di un documento dal DB."""
    try:
        col = get_collection()

        # Trova tutti i chunk con questo documento
        results = col.get(include=["metadatas"])
        ids_to_delete = [
            id_ for id_, meta in zip(results["ids"], results["metadatas"])
            if meta.get("source", "").split("/")[-1] == doc_name
        ]

        if not ids_to_delete:
            raise HTTPException(404, f"Documento '{doc_name}' non trovato nel DB")

        col.delete(ids=ids_to_delete)

        return {
            "status": "ok",
            "message": f"Documento '{doc_name}' eliminato",
            "chunks_deleted": len(ids_to_delete)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/documents/all")
async def delete_all_documents():
    """Elimina tutti i chunk da ChromaDB."""
    try:
        col = get_collection()
        total = col.count()
        # Ricrea la collezione vuota
        _p = urlparse(CHROMA_HOST)
        client = chromadb.HttpClient(
            host=_p.hostname or "chromadb",
            port=_p.port or 8000
        )
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        client.delete_collection(COLLECTION)
        client.create_collection(
            name=COLLECTION,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"}
        )
        return {"status": "ok", "chunks_deleted": total}
    except Exception as e:
        raise HTTPException(500, str(e))
