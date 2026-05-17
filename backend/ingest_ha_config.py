"""
Indicizzatore della configurazione Home Assistant per la v2.

Obiettivi di questa versione:
- indicizzare la configurazione HA realmente utile al reasoning
- ridurre il rumore (translations, asset statici, file build, ecc.)
- supportare batch upsert per ChromaDB
- mantenere il codice leggibile e facilmente estendibile

Nota progettuale:
questa versione privilegia file YAML, YML, JSON utili e pochi file testuali.
Per impostazione predefinita esclude directory e pattern notoriamente rumorosi.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse

import chromadb
class OllamaEmbeddingFunction:
    def __init__(self, model: str, host: str):
        self.model = model
        self.host = host.rstrip("/")

    def name(self) -> str:
        return f"ollama:{self.model}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        import requests as _req
        resp = _req.post(
            f"{self.host}/api/embed",
            json={"model": self.model, "input": input},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


# ── Config generale ─────────────────────────────────────────────────────────

# Directory sicuramente rumorose o non utili al reasoning HA.
SKIP_DIRS = {
    ".git",
    "__pycache__",
    "deps",
    "backups",
    "backup",
    "tts",
    "www",
    "custom_icons",
    "node_modules",
    "dist",
    "build",
    "translations",
    "translation",
    "locale",
    "locales",
    ".cloud",
    ".azure",
}

# Estensioni considerate potenzialmente utili.
# I .json vengono ulteriormente filtrati da una allowlist dedicata.
ALLOWED_EXTENSIONS = {".yaml", ".yml", ".json", ".txt", ".md"}

# File JSON considerati utili se si decide di includere parte della .storage
# o altri JSON strutturali rilevanti.
USEFUL_JSON_PATTERNS = [
    ".storage/core.entity_registry",
    ".storage/core.device_registry",
    ".storage/core.area_registry",
    ".storage/core.floor_registry",
    ".storage/core.config_entries",
    ".storage/lovelace",
    ".storage/lovelace.",
    "automations.json",
    "scenes.json",
    "scripts.json",
]

# Pattern rumorosi anche se l'estensione sarebbe teoricamente ammessa.
NOISY_PATH_PATTERNS = [
    "/translations/",
    "/translation/",
    "/locale/",
    "/locales/",
    "intl-displaynames",
    "hacs_frontend/static",
    "/frontend_latest/",
    "/frontend_es5/",
    ".min.js",
    ".bundle.",
    ".map.json",
    ".gz",
]

# Chunking
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 200

# Batch sicuro per ChromaDB
DEFAULT_MAX_BATCH_SIZE = 4000

# Se True, esclude completamente la directory .storage.
# Per ora lo imposto a True per ridurre rumore e tenere retrieval più pulito.
# In una v2.1 potresti volerlo impostare a False e usare allowlist selettiva.
EXCLUDE_STORAGE_DIR = True


# ── Helper Chroma ───────────────────────────────────────────────────────────

def get_collection(chroma_host: str, collection_name: str, embed_model: str):
    """Restituisce la collection Chroma dedicata alla config HA."""
    parsed = urlparse(chroma_host)
    client = chromadb.HttpClient(
        host=parsed.hostname or "chromadb",
        port=parsed.port or 8000,
    )

    ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    embed_fn = OllamaEmbeddingFunction(model=embed_model, host=ollama_host)

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


# ── Helper filtro file ──────────────────────────────────────────────────────

def should_skip(path: Path, root: Path) -> bool:
    """
    Decide se un file va escluso.

    Criteri:
    - directory da saltare
    - .storage esclusa completamente se EXCLUDE_STORAGE_DIR=True
    - pattern rumorosi noti
    """
    rel_path = path.relative_to(root)
    rel_str = "/" + str(rel_path).replace("\\", "/").lower()

    # Esclusione .storage completa se richiesta
    if EXCLUDE_STORAGE_DIR and any(part == ".storage" for part in rel_path.parts):
        return True

    # Esclusione directory note
    if any(part.lower() in {d.lower() for d in SKIP_DIRS} for part in rel_path.parts):
        return True

    # Pattern rumorosi
    if any(pattern in rel_str for pattern in NOISY_PATH_PATTERNS):
        return True

    return False


def is_allowed_file(path: Path, root: Path) -> bool:
    """
    Controlla se il file è ammesso per estensione e utilità.
    """
    suffix = path.suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        return False

    # YAML/YML ammessi sempre
    if suffix in {".yaml", ".yml"}:
        return True

    # TXT/MD ammessi, ma spesso secondari
    if suffix in {".txt", ".md"}:
        return True

    # JSON: ammessi solo se sembrano utili
    if suffix == ".json":
        rel_str = str(path.relative_to(root)).replace("\\", "/").lower()
        return any(token.lower() in rel_str for token in USEFUL_JSON_PATTERNS)

    return False


# ── Helper contenuto ────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """
    Chunking semplice, prevedibile e stabile.

    Non tenta parsing semantico sofisticato: lo scopo è produrre chunk coerenti
    e ripetibili, con overlap sufficiente per non perdere contesto.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)

    for start in range(0, len(cleaned), step):
        chunk = cleaned[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def build_stable_chunk_id(rel_path: str, chunk_index: int, chunk_text_value: str) -> str:
    """
    Costruisce un ID stabile e relativamente robusto.

    Mantiene il path e aggiunge un hash del contenuto per evitare collisioni
    strane in caso di file profondamente cambiati.
    """
    digest = hashlib.sha1(chunk_text_value.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"ha_config::{rel_path}::{chunk_index}::{digest}"


def read_file_text(path: Path) -> str:
    """
    Legge il file in modo tollerante.
    """
    return path.read_text(encoding="utf-8", errors="ignore")


def summarize_file_kind(path: Path) -> str:
    """
    Tag logico utile per metadata/retrieval/debug.
    """
    rel = str(path).replace("\\", "/").lower()

    if "automations" in rel:
        return "automation"
    if "scripts" in rel:
        return "script"
    if "scenes" in rel:
        return "scene"
    if "template" in rel:
        return "template"
    if "configuration.yaml" in rel:
        return "root_config"
    if rel.endswith(".yaml") or rel.endswith(".yml"):
        return "yaml"
    if rel.endswith(".json"):
        return "json"
    if rel.endswith(".md"):
        return "markdown"
    if rel.endswith(".txt"):
        return "text"

    return "generic"


# ── Ingest principale ───────────────────────────────────────────────────────

def ingest_home_assistant_config(
    config_root: str = "/ha_config",
    collection_name: str = "ha_config",
    chroma_host: str = "http://chromadb:8000",
    embed_model: str = "all-MiniLM-L6-v2",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
) -> dict:
    """
    Indicizza la config HA in Chroma e restituisce un riepilogo.

    Comportamento:
    - scansiona i file ammessi
    - esclude asset e rumore
    - spezza il contenuto in chunk
    - carica in ChromaDB a batch
    """
    root = Path(config_root)
    if not root.exists():
        raise FileNotFoundError(f"Cartella config non trovata: {root}")

    collection = get_collection(chroma_host, collection_name, embed_model)

    indexed_files = 0
    indexed_chunks = 0
    skipped_files = 0

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    print(f"[ingest_ha_config] Avvio scansione: root={root}")
    print(f"[ingest_ha_config] collection={collection_name} embed_model={embed_model}")
    print(f"[ingest_ha_config] chunk_size={chunk_size} overlap={overlap} max_batch_size={max_batch_size}")

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if should_skip(path, root):
            skipped_files += 1
            continue

        if not is_allowed_file(path, root):
            skipped_files += 1
            continue

        try:
            text = read_file_text(path)
        except Exception as exc:
            skipped_files += 1
            print(f"[ingest_ha_config] Lettura fallita: {path} -> {exc}")
            continue

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            skipped_files += 1
            continue

        rel = str(path.relative_to(root)).replace("\\", "/")
        file_kind = summarize_file_kind(path)
        indexed_files += 1

        for idx, chunk in enumerate(chunks):
            chunk_id = build_stable_chunk_id(rel, idx, chunk)

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "source": rel,
                    "page": idx,
                    "source_kind": "ha_config",
                    "path": rel,
                    "extension": path.suffix.lower(),
                    "file_kind": file_kind,
                    "chunk_index": idx,
                }
            )
            indexed_chunks += 1

    print(f"[ingest_ha_config] File indicizzati: {indexed_files}")
    print(f"[ingest_ha_config] File saltati: {skipped_files}")
    print(f"[ingest_ha_config] Chunk totali da caricare: {indexed_chunks}")

    if ids:
        total = len(ids)

        for start in range(0, total, max_batch_size):
            end = min(start + max_batch_size, total)

            batch_ids = ids[start:end]
            batch_documents = documents[start:end]
            batch_metadatas = metadatas[start:end]

            print(
                f"[ingest_ha_config] Upsert batch {start}-{end} / {total} "
                f"(size={len(batch_ids)})"
            )

            collection.upsert(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas,
            )

    return {
        "status": "ok",
        "config_root": str(root),
        "collection": collection_name,
        "indexed_files": indexed_files,
        "indexed_chunks": indexed_chunks,
        "skipped_files": skipped_files,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "max_batch_size": max_batch_size,
        "exclude_storage_dir": EXCLUDE_STORAGE_DIR,
    }


if __name__ == "__main__":
    result = ingest_home_assistant_config(
        config_root=os.getenv("HA_CONFIG_PATH", "/ha_config"),
        collection_name=os.getenv("CONFIG_COLLECTION", "ha_config"),
        chroma_host=os.getenv("CHROMA_HOST", "http://chromadb:8000"),
        embed_model=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"),
        chunk_size=int(os.getenv("CONFIG_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))),
        overlap=int(os.getenv("CONFIG_CHUNK_OVERLAP", str(DEFAULT_OVERLAP))),
        max_batch_size=int(os.getenv("CONFIG_MAX_BATCH_SIZE", str(DEFAULT_MAX_BATCH_SIZE))),
    )
    print(result)