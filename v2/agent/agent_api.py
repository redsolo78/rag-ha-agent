"""
HA Agent v2.

Questa implementazione punta a rendere l'agent più sicuro e più leggibile:
- letture separate tra stato reale, log e config
- retrieval opzionale sulla collection ha_config
- azioni Home Assistant solo tramite flusso propose → confirm → execute
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from urllib.parse import urlparse

import chromadb
import requests
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Configurazione ─────────────────────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "http://chromadb:8000")
HA_HOST = os.getenv("HA_HOST", "http://homeassistant:8123")
HA_TOKEN = os.getenv("HA_TOKEN", "")
HA_CONFIG_PATH = os.getenv("HA_CONFIG_PATH", "/ha_config")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.1:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CONFIG_COLLECTION = os.getenv("CONFIG_COLLECTION", "ha_config")
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"
ALLOW_HA_ACTIONS = os.getenv("ALLOW_HA_ACTIONS", "false").lower() == "true"
ACTION_CONFIRM_TOKEN = os.getenv("ACTION_CONFIRM_TOKEN", "")
MAX_CONFIG_CHARS = int(os.getenv("MAX_CONFIG_CHARS", "7000"))
MAX_EXTRA_CHARS = int(os.getenv("MAX_EXTRA_CHARS", "4000"))

# Storage in-memory delle proposte di azione.
# In una fase successiva si può sostituire con Redis o persistenza dedicata.
PENDING_ACTIONS: dict[str, dict] = {}



def parse_allowed_origins() -> list[str]:
    raw = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(title="HA Agent v2", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Confirm-Token"],
)


class AgentChatRequest(BaseModel):
    """Richiesta libera verso l'agent v2."""

    message: str = Field(..., min_length=1)


class ActionProposalRequest(BaseModel):
    """Richiesta per proporre una chiamata a servizio HA."""

    service: str = Field(..., description="Formato domain.service, es. light.turn_on")
    entity_id: str | None = Field(default=None, description="Entity target opzionale")
    data: dict = Field(default_factory=dict, description="Payload extra da inviare al service call")
    reason: str = Field(default="", description="Motivazione leggibile per l'utente")


class ActionConfirmRequest(BaseModel):
    """Conferma di una proposta precedentemente generata."""

    action_id: str


# ── Helper Chroma / HA ────────────────────────────────────────────────────
def get_config_collection():
    """Collection dedicata ai chunk della config HA."""
    parsed = urlparse(CHROMA_HOST)
    client = chromadb.HttpClient(host=parsed.hostname or "chromadb", port=parsed.port or 8000)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(name=CONFIG_COLLECTION, embedding_function=embed_fn)



def ha_headers() -> dict[str, str]:
    """Header standard per autenticarsi verso l'API di Home Assistant."""
    return {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}



def ha_get(path: str):
    """GET verso l'API HA con gestione eccezioni semplice."""
    response = requests.get(f"{HA_HOST}/api{path}", headers=ha_headers(), timeout=10)
    response.raise_for_status()
    return response.json() if response.text else {"status": "ok"}



def ha_post(path: str, payload: dict):
    """POST verso l'API HA per esecuzione servizi o altre operazioni."""
    response = requests.post(f"{HA_HOST}/api{path}", headers=ha_headers(), json=payload, timeout=10)
    response.raise_for_status()
    return response.json() if response.text else {"status": "ok"}



def search_config_context(query: str, limit: int = 4) -> str:
    """Recupera chunk utili dalla collection `ha_config`.

    Questa funzione serve a rendere l'agent più consapevole della configurazione
    reale, senza dover leggere sempre file interi dal filesystem.
    """
    collection = get_config_collection()
    results = collection.query(
        query_texts=[query],
        n_results=limit,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        return "Nessun chunk rilevante trovato in ha_config."

    parts: list[str] = []
    for idx, (doc, meta) in enumerate(zip(docs, metas), start=1):
        source = meta.get("source", "unknown")
        parts.append(f"[Config {idx} | {source}]\n{doc[:1200]}")
    return "\n\n---\n\n".join(parts)[:MAX_CONFIG_CHARS]



def read_small_config_sample() -> str:
    """Legge alcuni file principali per una vista minima anche senza retrieval."""
    root = Path(HA_CONFIG_PATH)
    if not root.exists():
        return "Cartella config HA non montata o non presente."

    priority_files = ["configuration.yaml", "automations.yaml", "scripts.yaml", "scenes.yaml"]
    parts: list[str] = []
    used = 0

    for name in priority_files:
        path = root / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        snippet = text[:1600]
        block = f"--- {name} ---\n{snippet}"
        if used + len(block) > MAX_CONFIG_CHARS:
            break
        parts.append(block)
        used += len(block)

    return "\n\n".join(parts) if parts else "Nessun file prioritario trovato nella config HA."


# ── Endpoint operativi ─────────────────────────────────────────────────────
@app.get("/agent/health")
def health() -> dict:
    """Health endpoint utile per verificare l'integrazione reale con HA."""
    ha_online = False
    try:
        response = requests.get(f"{HA_HOST}/api/", headers=ha_headers(), timeout=5)
        ha_online = response.status_code == 200
    except Exception:
        ha_online = False

    return {
        "status": "ok",
        "model": LLAMA_MODEL,
        "ha_host": HA_HOST,
        "ha_online": ha_online,
        "allow_ha_actions": ALLOW_HA_ACTIONS,
        "config_collection": CONFIG_COLLECTION,
        "web_search_enabled": ENABLE_WEB_SEARCH,
    }


@app.post("/agent/chat")
def agent_chat(body: AgentChatRequest) -> dict:
    """Restituisce un contesto tecnico arricchito, pronto per un LLM o una UI.

    In questa proposta v2 l'endpoint restituisce una struttura trasparente e
    leggibile. In un'estensione successiva può anche inviare il contesto ad
    Ollama per produrre la risposta finale direttamente server-side.
    """
    states = ha_get("/states")
    entities_preview = []
    for item in states[:20]:
        entities_preview.append(
            {
                "entity_id": item.get("entity_id"),
                "state": item.get("state"),
                "friendly_name": item.get("attributes", {}).get("friendly_name", ""),
            }
        )

    return {
        "status": "ok",
        "message": body.message,
        "ha_summary": {
            "total_entities": len(states),
            "entities_preview": entities_preview,
        },
        "config_context": search_config_context(body.message),
        "config_sample": read_small_config_sample(),
        "note": "Questo endpoint v2 restituisce un contesto strutturato e grounded da usare nella UI o in una chiamata LLM successiva.",
    }


@app.post("/agent/action/propose")
def propose_action(body: ActionProposalRequest) -> dict:
    """Genera una proposta di azione ma non la esegue.

    Questo è il cuore del modello propose → confirm → execute.
    """
    if "." not in body.service:
        raise HTTPException(status_code=400, detail="Il service deve essere nel formato domain.service")

    action_id = secrets.token_urlsafe(12)
    proposal = {
        "action_id": action_id,
        "service": body.service,
        "entity_id": body.entity_id,
        "data": body.data,
        "reason": body.reason,
        "ha_host": HA_HOST,
    }
    PENDING_ACTIONS[action_id] = proposal

    return {
        "status": "confirmation_required",
        "proposal": proposal,
        "message": "Azione proposta ma non eseguita. Conferma richiesta.",
    }


@app.post("/agent/action/confirm")
def confirm_action(body: ActionConfirmRequest, x_confirm_token: str | None = Header(default=None)) -> dict:
    """Esegue una proposta già creata solo dopo verifica del token di conferma."""
    if not ACTION_CONFIRM_TOKEN or x_confirm_token != ACTION_CONFIRM_TOKEN:
        raise HTTPException(status_code=401, detail="Confirm token non valido.")

    if not ALLOW_HA_ACTIONS:
        raise HTTPException(status_code=403, detail="Le azioni HA sono disabilitate da configurazione.")

    proposal = PENDING_ACTIONS.get(body.action_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Azione proposta non trovata o già eseguita.")

    domain, service = proposal["service"].split(".", 1)
    payload = dict(proposal.get("data", {}))
    if proposal.get("entity_id"):
        payload["entity_id"] = proposal["entity_id"]

    result = ha_post(f"/services/{domain}/{service}", payload)
    PENDING_ACTIONS.pop(body.action_id, None)

    return {
        "status": "executed",
        "service": proposal["service"],
        "entity_id": proposal.get("entity_id"),
        "result": result,
    }
