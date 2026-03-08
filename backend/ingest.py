"""
STEP 1 - INGEST: PDF → Chunks → Embeddings → ChromaDB

Uso:
  python ingest.py --pdf ./data/tools.pdf
  python ingest.py --pdf ./data/tools.pdf --chunk-size 1200 --chunk-overlap 300
"""

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


def print_progress(current, total, label="", bar_width=42):
    pct = current / total if total > 0 else 0
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)
    if current >= total:
        print(f"\r  [{bar}] 100%  {'Completato':<35}", flush=True)
    else:
        print(f"\r  [{bar}] {int(pct*100):3d}%  {label[:35]:<35}", end="", flush=True)

import fitz  # PyMuPDF
import json
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# ── Configurazione da env vars (Docker) ──────────────────────
CHROMA_HOST   = os.getenv("CHROMA_HOST", "http://chromadb:8000")   # fix: era localhost:8001
COLLECTION    = "documents"
EMBED_MODEL   = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1200"))                # fix: era 500, troppo piccolo
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))              # fix: era 100
# ─────────────────────────────────────────────────────────────


# ── Pulizia e normalizzazione testo ──────────────────────────

def extract_yaml_frontmatter(text: str) -> tuple[dict, str]:
    """
    Estrae il front-matter YAML da file .md e lo rimuove dal testo.
    Ritorna (metadata_dict, testo_pulito).
    """
    metadata = {}
    # Cerca il front-matter: --- ... ---  in testa al file
    fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
    if fm_match:
        fm_raw = fm_match.group(1)
        # Parsing manuale delle chiavi YAML (key: value)
        for line in fm_raw.splitlines():
            m = re.match(r'^(\w[\w_]*):\s*(.+)$', line.strip())
            if m:
                key, val = m.group(1), m.group(2).strip().strip('"\'\' ')
                metadata[key] = val
        text = text[fm_match.end():]  # rimuovi front-matter
    return metadata, text


def clean_markdown(text: str) -> str:
    """
    Pulisce un testo markdown rimuovendo rumore ma conservando struttura utile.
    """
    # 1. Rimuovi tag Liquid/Jekyll: {% ... %} e {{ ... }}
    text = re.sub(r'{%[^%]*%}', '', text)
    text = re.sub(r'{{[^}]*}}', '', text)

    # 2. Rimuovi tag HTML (es. <div>, <span>, commenti)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)

    # 3. Rimuovi link markdown ma conserva il testo: [testo](url) → testo
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # 4. Rimuovi immagini: ![alt](url)
    text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)

    # 5. Rimuovi separatori orizzontali
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # 6. Normalizza spazi multipli (ma non newline)
    text = re.sub(r'[ \t]+', ' ', text)

    # 7. Riduci righe vuote multiple
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def clean_txt(text: str) -> str:
    """Pulizia base per file .txt."""
    # Rimuovi caratteri di controllo eccetto newline e tab
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clean_pdf_text(text: str) -> str:
    """Pulizia per testo estratto da PDF (rimuove artefatti OCR comuni)."""
    # Rimuovi header/footer tipici (numeri di pagina isolati)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # Rimuovi lineette di sillabazione a fine riga
    text = re.sub(r'-\n', '', text)
    # Normalizza
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Estrae il testo pagina per pagina dal PDF."""
    doc = fitz.open(pdf_path)
    pages = []

    print(f"   Pagine totali: {len(doc)}")

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = clean_pdf_text(text)

        if text:
            pages.append({
                "page_num": page_num,
                "text": text,
                "source": pdf_path
            })

    doc.close()
    return pages


def extract_text_from_txt(file_path: str) -> list[dict]:
    """Estrae e pulisce testo da file .txt e .md."""
    ext = Path(file_path).suffix.lower()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    metadata = {}
    if ext == ".md":
        # Estrai metadati dal front-matter YAML
        metadata, raw = extract_yaml_frontmatter(raw)
        raw = clean_markdown(raw)
    else:
        raw = clean_txt(raw)

    # Prepend metadati utili come testo se presenti
    meta_prefix = ""
    for key in ("title", "description", "sidebar_label", "ha_category"):
        if key in metadata:
            meta_prefix += f"{key}: {metadata[key]}\n"
    if meta_prefix:
        raw = meta_prefix + "\n" + raw

    # Dividi in blocchi virtuali da ~50 righe
    lines = raw.splitlines()
    PAGE_LINES = 50
    pages = []

    for i in range(0, len(lines), PAGE_LINES):
        block = "\n".join(lines[i:i + PAGE_LINES]).strip()
        if block:
            pages.append({
                "page_num": (i // PAGE_LINES) + 1,
                "text": block,
                "source": file_path,
                "metadata_extra": metadata if i == 0 else {}
            })

    pass  # verbose print rimosso
    return pages


def extract_text(file_path: str) -> list[dict]:
    """Router: sceglie il parser giusto in base all'estensione."""
    ext = Path(file_path).suffix.lower()
    ext = EXTENSION_ALIASES.get(ext, ext)  # .markdown → .md
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".txt", ".md"):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Formato non supportato: {ext}. Usa .pdf, .txt o .md")


def split_into_chunks(pages: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """Divide il testo in chunk con overlap."""
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                boundary = text.rfind('. ', start, end)
                if boundary == -1:
                    boundary = text.rfind('\n', start, end)
                if boundary != -1 and boundary > start + chunk_size // 2:
                    end = boundary + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                # fix: ID include page_num e chunk_index → niente più DuplicateIDError
                chunk_id = hashlib.md5(
                    f"{page['source']}_{page['page_num']}_{chunk_index}_{chunk_text[:100]}".encode()
                ).hexdigest()

                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": page["source"],
                        "page": page["page_num"],
                        "chunk_index": chunk_index
                    }
                })
                chunk_index += 1

            start = end - overlap

    return chunks


def ingest_to_chromadb(chunks: list[dict]) -> None:
    """Crea gli embedding e li salva in ChromaDB via HTTP."""
    parsed = urlparse(CHROMA_HOST)
    client = chromadb.HttpClient(
        host=parsed.hostname or "chromadb",
        port=parsed.port or 8000
    )

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # Usa collezione esistente o creane una nuova — NON cancellare mai
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )
    existing = collection.count()
    print(f"📚 Collezione '{COLLECTION}': {existing} chunks esistenti — aggiungo nuovi documenti")

    BATCH_SIZE = 100
    total = len(chunks)

    print(f"\n🔢 Creazione embedding per {total} chunks...")
    print(f"   Modello: {EMBED_MODEL}")

    batches = list(range(0, total, BATCH_SIZE))
    for idx, i in enumerate(batches):
        batch = chunks[i:i + BATCH_SIZE]
        collection.add(
            ids       = [c["id"] for c in batch],
            documents = [c["text"] for c in batch],
            metadatas = [c["metadata"] for c in batch]
        )
        print_progress(idx + 1, len(batches), f"batch {idx+1}/{len(batches)}")

    print(f"\n✅ Ingestion completata!")
    print(f"   {total} chunks salvati in ChromaDB")
    print(f"   Collezione: '{COLLECTION}'")


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

# Alias: tratta .markdown come .md
EXTENSION_ALIASES = {".markdown": ".md"}


def collect_files(folder: str) -> list[str]:
    """Raccoglie ricorsivamente tutti i file supportati in una cartella."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(Path(folder).rglob(f"*{ext}"))
    return sorted([str(f) for f in files])


def main():
    parser = argparse.ArgumentParser(
        description="Ingestion documenti → ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ESEMPI:
  python ingest.py --pdf /app/data/documento.pdf
  python ingest.py --folder /app/data/ha_docs/ha_docs_clean
  python ingest.py --folder /app/data/ha_integrations --ext .md .yaml
  python ingest.py --folder /app/data/ha_docs --chunk-size 1200 --chunk-overlap 300

FORMATI SUPPORTATI:  .pdf  .txt  .md  .markdown  .yaml  .rst

NOTA: I documenti vengono AGGIUNTI al DB esistente (non cancellati).
      Per svuotare usa il pulsante nell interfaccia web o DELETE /documents/all
"""
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf",    metavar="FILE",
                       help="Singolo file da indicizzare (.pdf .txt .md .markdown)")
    group.add_argument("--folder", metavar="DIR",
                       help="Cartella da indicizzare ricorsivamente")
    parser.add_argument("--chunk-size",    type=int, default=CHUNK_SIZE, metavar="N",
                        help=f"Dimensione chunk in caratteri (default: {CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, metavar="N",
                        help=f"Overlap tra chunk (default: {CHUNK_OVERLAP})")
    parser.add_argument("--ext", nargs="+", default=None, metavar="EXT",
                        help="Filtra estensioni (es: --ext .md .yaml .rst)")
    args = parser.parse_args()

    print("\n🚀 Avvio pipeline RAG Ingestion\n" + "─" * 50)

    # ── Modalità CARTELLA ────────────────────────────────────────
    if args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"❌ Cartella non trovata: {args.folder}")
            sys.exit(1)

        # Filtra estensioni se specificate
        allowed = set(args.ext) if args.ext else SUPPORTED_EXTENSIONS
        files = [
            str(f) for f in sorted(folder.rglob("*"))
            if f.is_file() and f.suffix.lower() in allowed
        ]

        if not files:
            print(f"❌ Nessun file supportato trovato in: {args.folder}")
            print(f"   Estensioni cercate: {allowed}")
            sys.exit(1)

        print(f"📁 Cartella: {args.folder}")
        print(f"   File trovati: {len(files)}")
        print(f"   Estensioni: {allowed}\n")

        all_chunks = []
        skipped = []
        print(f"\n📄 Lettura e chunking file...")
        for i, file_path in enumerate(files, 1):
            fname = Path(file_path).name
            try:
                pages  = extract_text(file_path)
                chunks = split_into_chunks(pages, args.chunk_size, args.chunk_overlap)
                all_chunks.extend(chunks)
            except Exception as e:
                skipped.append((fname, str(e)))
            print_progress(i, len(files), fname)

        if skipped:
            print(f"\n  ⚠️  {len(skipped)} file saltati:")
            for fname_err, reason in skipped:
                print(f"     - {fname_err}: {reason}")

        print(f"\n✂️  Chunking completato:")
        print(f"   File processati: {len(files)}")
        print(f"   Chunk totali:    {len(all_chunks)}")
        print(f"   Chunk size:      {args.chunk_size} caratteri")

        ingest_to_chromadb(all_chunks)

    # ── Modalità FILE SINGOLO ────────────────────────────────────
    else:
        if not Path(args.pdf).exists():
            print(f"❌ File non trovato: {args.pdf}")
            sys.exit(1)

        pages  = extract_text(args.pdf)
        chunks = split_into_chunks(pages, args.chunk_size, args.chunk_overlap)

        print(f"\n✂️  Chunking completato:")
        print(f"   Pagine estratte: {len(pages)}")
        print(f"   Chunk creati:    {len(chunks)}")
        print(f"   Chunk size:      {args.chunk_size} caratteri")
        print(f"   Overlap:         {args.chunk_overlap} caratteri")

        ingest_to_chromadb(chunks)


if __name__ == "__main__":
    main()
