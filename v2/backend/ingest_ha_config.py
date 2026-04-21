"""
Indicizzatore della configurazione Home Assistant per la v2.

Questo script è volutamente leggibile e commentato: il suo scopo non è solo
fare ingest, ma anche mostrare chiaramente il criterio con cui la v2 tratta la
configurazione locale come sorgente primaria di conoscenza.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import chromadb
from chromadb.utils import embedding_functions

# Directory da escludere per ridurre rumore, peso inutile e duplicazioni.
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".storage",
    "tts",
    "deps",
    "backups",
    "www",
    "custom_icons",
}

# Estensioni considerate utili per una knowledge base tecnica della config HA.
ALLOWED_EXTENSIONS = {".yaml", ".yml", ".json", ".txt", ".md"}



def get_collection(chroma_host: str, collection_name: str, embed_model: str):
    """Restituisce la collection Chroma dedicata alla config HA."""
    parsed = urlparse(chroma_host)
    client = chromadb.HttpClient(host=parsed.hostname or "chromadb", port=parsed.port or 8000)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )



def should_skip(path: Path) -> bool:
    """Decide se un file va escluso in base alle directory attraversate."""
    return any(part in SKIP_DIRS for part in path.parts)



def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """Chunking semplice ma prevedibile.

    Per file YAML e testuali l'obiettivo non è spezzare perfettamente il senso,
    ma mantenere chunk ragionevoli, con un piccolo overlap per non perdere
    contesto fra un blocco e il successivo.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(cleaned), step):
        chunk = cleaned[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks



def ingest_home_assistant_config(
    config_root: str = "/ha_config",
    collection_name: str = "ha_config",
    chroma_host: str = "http://chromadb:8000",
    embed_model: str = "all-MiniLM-L6-v2",
) -> dict:
    """Indicizza la config HA in Chroma e restituisce un riepilogo.

    La funzione è richiamabile sia come script CLI sia dal backend via import.
    """
    root = Path(config_root)
    if not root.exists():
        raise FileNotFoundError(f"Cartella config non trovata: {root}")

    collection = get_collection(chroma_host, collection_name, embed_model)

    indexed_files = 0
    indexed_chunks = 0

    # Per semplicità la collection viene ricostruita con upsert su ID stabili.
    # Una strategia più sofisticata può aggiungere delete selettivo per file rimossi.
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if should_skip(path):
            continue
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text)
        if not chunks:
            continue

        rel = str(path.relative_to(root))
        indexed_files += 1

        for idx, chunk in enumerate(chunks):
            ids.append(f"ha_config::{rel}::{idx}")
            documents.append(chunk)
            metadatas.append(
                {
                    "source": rel,
                    "page": idx,
                    "source_kind": "ha_config",
                    "path": rel,
                    "extension": path.suffix.lower(),
                }
            )
            indexed_chunks += 1

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return {
        "status": "ok",
        "config_root": str(root),
        "collection": collection_name,
        "indexed_files": indexed_files,
        "indexed_chunks": indexed_chunks,
    }


if __name__ == "__main__":
    result = ingest_home_assistant_config(
        config_root=os.getenv("HA_CONFIG_PATH", "/ha_config"),
        collection_name=os.getenv("CONFIG_COLLECTION", "ha_config"),
        chroma_host=os.getenv("CHROMA_HOST", "http://chromadb:8000"),
        embed_model=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"),
    )
    print(result)
