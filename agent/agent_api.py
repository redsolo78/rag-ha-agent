"""
HA Agent API — assistente specializzato in Home Assistant.
Endpoint SSE: POST /agent/chat   { "message": "..." }
"""

import asyncio
import json
import os
import re
import traceback
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any
from urllib.parse import urlparse

import chromadb
import requests
from chromadb.utils import embedding_functions
from duckduckgo_search import DDGS
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ── Config ─────────────────────────────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "http://chromadb:8000")
HA_HOST = os.getenv("HA_HOST", "http://homeassistant:8123")
HA_TOKEN = os.getenv("HA_TOKEN", "")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.1:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
COLLECTION = os.getenv("COLLECTION", "documents")
HA_CONFIG_PATH = os.getenv("HA_CONFIG_PATH", "/ha_config")
TOP_K = max(int(os.getenv("TOP_K", "8")), 1)
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"
ALLOW_HA_ACTIONS = os.getenv("ALLOW_HA_ACTIONS", "false").lower() == "true"
MAX_CONFIG_CHARS = max(int(os.getenv("MAX_CONFIG_CHARS", "7000")), 1000)
MAX_EXTRA_CHARS = max(int(os.getenv("MAX_EXTRA_CHARS", "4000")), 1000)
ACTION_CONFIRM_TOKEN = os.getenv("ACTION_CONFIRM_TOKEN", "")


def parse_allowed_origins() -> list[str]:
    raw = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


ALLOWED_ORIGINS = parse_allowed_origins()
AVAILABLE_TOOLS = [
    "ha_get_config",
    "ha_get_states",
    "ha_get_logs",
    "read_config_file",
    "list_config_files",
    "search_ha_docs",
    "web_search",
    "ha_call_service_preview",
    "ha_call_service_confirm",
]
PENDING_ACTIONS: dict[str, dict[str, Any]] = {}

app = FastAPI(title="HA Agent", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Confirm-Token"],
)


# ── Helpers ────────────────────────────────────────────────────────────────
def parse_host(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    return parsed.hostname or "localhost", parsed.port or 80


def get_chroma_collection():
    host, port = parse_host(CHROMA_HOST)
    client = chromadb.HttpClient(host=host, port=port)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(name=COLLECTION, embedding_function=embed_fn)


def ha_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if HA_TOKEN:
        headers["Authorization"] = f"Bearer {HA_TOKEN}"
    return headers


def sanitize_rel_path(value: str) -> Path:
    cleaned = value.strip().strip('"\'')
    rel = Path(cleaned)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Path non consentito")
    return rel


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def shorten_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n...[troncato, lunghezza originale {len(text)} caratteri]"


def clip_extra(parts: list[str], limit: int) -> str:
    out = []
    total = 0
    for part in parts:
        if not part:
            continue
        remaining = limit - total
        if remaining <= 0:
            break
        clipped = part[:remaining]
        out.append(clipped)
        total += len(clipped)
    return "\n\n".join(out)


def extract_explicit_entity_ids(message: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\b[a-z_]+\.[a-z0-9_]+\b", message.lower())))


def build_pending_action(service_call: str, entity_id: str | None = None, extra_data: dict[str, Any] | None = None) -> dict[str, Any]:
    action_id = f"action_{len(PENDING_ACTIONS) + 1}"
    payload = {
        "action_id": action_id,
        "service_call": service_call,
        "entity_id": entity_id,
        "data": extra_data or {},
    }
    PENDING_ACTIONS[action_id] = payload
    return payload


def require_confirm_token(token: str | None) -> None:
    if not ACTION_CONFIRM_TOKEN:
        raise HTTPException(503, "Conferma azioni disabilitata: ACTION_CONFIRM_TOKEN non configurato.")
    if token != ACTION_CONFIRM_TOKEN:
        raise HTTPException(403, "Token di conferma non valido.")


def ha_get(path: str) -> dict | list | str:
    try:
        response = requests.get(f"{HA_HOST}/api{path}", headers=ha_headers(), timeout=10)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        return response.text
    except Exception as exc:
        return {"error": str(exc)}


def ha_post(path: str, data: dict[str, Any]) -> dict[str, Any] | list[Any]:
    try:
        response = requests.post(
            f"{HA_HOST}/api{path}", headers=ha_headers(), json=data, timeout=10
        )
        response.raise_for_status()
        if response.text:
            try:
                return response.json()
            except Exception:
                return {"status": "ok", "raw": response.text[:500]}
        return {"status": "ok"}
    except Exception as exc:
        return {"error": str(exc)}


# ════════════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════════════
def tool_search_docs(query: str) -> str:
    """Cerca nella documentazione di Home Assistant indicizzata in ChromaDB."""
    try:
        col = get_chroma_collection()
        results = col.query(query_texts=[query], n_results=min(TOP_K * 2, 16), include=["documents", "metadatas"])
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        if not docs:
            return "Nessun documento rilevante trovato."
        out = []
        seen: set[tuple[str, int]] = set()
        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            meta = meta or {}
            src = str(meta.get("source", "?")).split("/")[-1]
            page = int(meta.get("page", 0) or 0)
            dedup = (src, page)
            if dedup in seen:
                continue
            seen.add(dedup)
            out.append(f"[Doc {i} - {src} - pag. {page}]\n{shorten_text(doc, 700)}")
            if len(out) >= TOP_K:
                break
        return "\n\n---\n\n".join(out)
    except Exception as exc:
        return f"Errore ricerca docs: {exc}"


def tool_ha_get_states(filter_str: str = "") -> str:
    """Legge gli stati delle entità di Home Assistant."""
    try:
        filter_str = filter_str.strip().lower()
        if filter_str and "." in filter_str:
            result = ha_get(f"/states/{filter_str}")
            if isinstance(result, dict) and "error" in result:
                return f"Entità non trovata: {filter_str}"
            return json.dumps(
                {
                    "entity_id": result.get("entity_id"),
                    "state": result.get("state"),
                    "attributes": result.get("attributes", {}),
                },
                indent=2,
                ensure_ascii=False,
            )

        all_states = ha_get("/states")
        if isinstance(all_states, dict) and "error" in all_states:
            return f"Errore HA: {all_states['error']}"

        states = all_states
        if filter_str:
            states = [s for s in all_states if s.get("entity_id", "").startswith(filter_str + ".")]
        if not states:
            return f"Nessuna entità trovata per: '{filter_str}'"

        domains: dict[str, list[dict[str, str]]] = {}
        for state in states:
            entity_id = state.get("entity_id", "unknown.unknown")
            domain = entity_id.split(".")[0]
            domains.setdefault(domain, []).append(
                {
                    "id": entity_id,
                    "state": str(state.get("state", "unknown")),
                    "name": str(state.get("attributes", {}).get("friendly_name", "")),
                }
            )

        lines = [f"Totale entità selezionate: {len(states)}"]
        for domain, entities in sorted(domains.items()):
            lines.append(f"\n## {domain.upper()} ({len(entities)} entità)")
            for entity in entities[:20]:
                lines.append(f"  - {entity['id']}: {entity['state']} ({entity['name']})")
            if len(entities) > 20:
                lines.append(f"  ... e altri {len(entities) - 20}")

        return "\n".join(lines)
    except Exception:
        err = traceback.format_exc()
        print(f"[TOOL ERROR ha_get_states] {err}", flush=True)
        return f"Errore: {err}"


def tool_ha_get_logs(lines_str: str = "50") -> str:
    """Legge gli ultimi log di errore di Home Assistant."""
    try:
        num_lines = int(lines_str.strip()) if lines_str.strip().isdigit() else 50
        result = ha_get("/error_log")
        if isinstance(result, dict) and "error" in result:
            return f"Errore: {result['error']}"
        log_text = str(result)
        lines = log_text.splitlines()
        selected = "\n".join(lines[-num_lines:]) if len(lines) > num_lines else log_text
        return shorten_text(selected, 2500)
    except Exception as exc:
        return f"Errore lettura log: {exc}"


def tool_ha_call_service_preview(input_str: str) -> str:
    """Prepara un'azione HA ma NON la esegue."""
    try:
        parts = input_str.strip().split()
        if not parts:
            return "Formato: 'domain.service entity_id'"
        service_call = parts[0]
        entity_id = parts[1] if len(parts) > 1 else None
        if "." not in service_call:
            return f"Formato non valido: '{service_call}'. Usa 'domain.service'"
        action = build_pending_action(service_call, entity_id)
        return json.dumps(
            {
                "type": "confirmation_required",
                "message": f"Conferma richiesta per {service_call} su {entity_id or 'tutti'}",
                **action,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        return f"Errore preview azione: {exc}"


def tool_ha_call_service_confirm(action_id: str) -> str:
    """Esegue un'azione pending su Home Assistant solo se abilitata."""
    if not ALLOW_HA_ACTIONS:
        return "Azioni HA disabilitate: imposta ALLOW_HA_ACTIONS=true per consentirle."
    action = PENDING_ACTIONS.get(action_id.strip())
    if not action:
        return f"Azione pending non trovata: {action_id}"
    service_call = action["service_call"]
    entity_id = action.get("entity_id")
    if "." not in service_call:
        return f"Formato servizio non valido: {service_call}"
    domain, service = service_call.split(".", 1)
    data = dict(action.get("data") or {})
    if entity_id:
        data["entity_id"] = entity_id
    result = ha_post(f"/services/{domain}/{service}", data)
    if isinstance(result, dict) and "error" in result:
        return f"Errore servizio: {result['error']}"
    PENDING_ACTIONS.pop(action_id, None)
    return f"✅ Servizio {service_call} eseguito su {entity_id or 'tutti'}"


def tool_read_config_file(filename: str) -> str:
    """Legge un file di configurazione di Home Assistant."""
    try:
        rel = sanitize_rel_path(filename)
        base = Path(HA_CONFIG_PATH)
        target = base / rel
        if not target.exists():
            matches = list(base.rglob(rel.name))
            if not matches:
                available = [str(f.relative_to(base)) for f in base.rglob("*.yaml")][:30]
                if not available:
                    return (
                        f"ATTENZIONE: La cartella di configurazione HA sembra vuota o non montata correttamente. "
                        f"Percorso: {base}. Usa ha_get_config per leggere la configurazione via API."
                    )
                return f"File '{filename}' non trovato.\nFile YAML disponibili:\n" + "\n".join(available)
            target = matches[0]
        content = safe_read_text(target)
        return f"📄 {target.relative_to(base)}\n\n{shorten_text(content, 5000)}"
    except ValueError as exc:
        return f"Errore lettura file: {exc}"
    except Exception as exc:
        return f"Errore lettura file: {exc}"


def tool_list_config_files(path: str = "") -> str:
    """Lista i file di configurazione disponibili."""
    try:
        rel = sanitize_rel_path(path) if path.strip() else Path("")
        base = (Path(HA_CONFIG_PATH) / rel).resolve()
        root = Path(HA_CONFIG_PATH).resolve()
        if root not in [base, *base.parents]:
            return "Cartella non consentita"
        if not base.exists():
            return f"Cartella non trovata: {path}"
        files = []
        for file_path in sorted(base.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in (".yaml", ".yml", ".json", ".txt"):
                rel_path = str(file_path.relative_to(root))
                files.append(f"  {rel_path} ({file_path.stat().st_size} bytes)")
        if not files:
            return "Nessun file di configurazione trovato."
        return f"File di configurazione ({len(files)} totali):\n" + "\n".join(files[:100])
    except ValueError as exc:
        return f"Errore: {exc}"
    except Exception as exc:
        return f"Errore: {exc}"


def tool_web_search(query: str) -> str:
    """Cerca informazioni aggiornate sul web."""
    if not ENABLE_WEB_SEARCH:
        return "⚠️ Web search disabilitato. Imposta ENABLE_WEB_SEARCH=true nel file .env per abilitarlo."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query + " Home Assistant", max_results=5))
        if not results:
            return "Nessun risultato trovato."
        out = []
        for result in results:
            out.append(
                f"**{result.get('title', '?')}**\n{shorten_text(result.get('body', ''), 400)}\n🔗 {result.get('href', '')}"
            )
        return "\n\n---\n\n".join(out)
    except Exception as exc:
        return f"Errore web search: {exc}"


def tool_ha_get_config(dummy: str = "") -> str:
    """Legge la configurazione generale di Home Assistant."""
    try:
        result = ha_get("/config")
        if isinstance(result, dict) and "error" in result:
            return f"Errore: {result['error']}"
        keys = [
            "version",
            "location_name",
            "time_zone",
            "unit_system",
            "components",
            "config_dir",
            "latitude",
            "longitude",
        ]
        summary = {key: result.get(key) for key in keys if key in result}
        components = result.get("components", [])
        summary["components_count"] = len(components)
        summary["components_sample"] = sorted(components)[:30]
        return json.dumps(summary, indent=2, ensure_ascii=False)
    except Exception as exc:
        return f"Errore: {exc}"


# ── Prompt ─────────────────────────────────────────────────────────────────
ANALYSIS_PROMPT = """Sei un esperto senior di Home Assistant che analizza un sistema reale. Rispondi SEMPRE in italiano.

CONTESTO: Hai accesso a configurazione, stati, log e documentazione recuperati per questa richiesta.

REGOLE ASSOLUTE:
1. Usa solo i dati presenti nel contesto.
2. Se un dato manca o è incompleto, dichiaralo esplicitamente.
3. Non inventare file, entità, errori o automazioni.
4. Cita sempre file, entità o sezioni log quando fai affermazioni fattuali.
5. Se proponi una correzione YAML, separa chiaramente PRIMA (osservazione) e DOPO (proposta).
6. Se vedi Node-RED attivo, evita di suggerire automazioni duplicate senza motivazione.
7. Se la richiesta implica un'azione sui dispositivi, NON dichiarare che l'hai eseguita: puoi solo proporla o chiedere conferma.

FORMATO RISPOSTA:
- Inizia con un sommario diretto.
- Poi dividi in: Evidenze, Analisi, Correzioni/Proposte, Priorità.
- Mantieni le affermazioni tecniche ancorate ai dati reali.

=== CONFIGURAZIONE SISTEMA ===
{ha_config}

=== FILE DI CONFIGURAZIONE YAML ===
{config_yaml}

=== DATI AGGIUNTIVI (dispositivi, log, documentazione) ===
{extra_data}

=== DOMANDA ===
{question}

Rispondi analizzando i dati reali sopra. Se non bastano, dichiaralo chiaramente."""


# ── Intent detection ───────────────────────────────────────────────────────
def detect_intent(message: str) -> list[str]:
    msg = message.lower()
    tools_to_use = ["ha_config"]

    if any(word in msg for word in ["configurazione", "yaml", "config", "file", "problema", "errore", "analizza", "controlla"]):
        tools_to_use.append("config_yaml")
    if any(word in msg for word in ["log", "errore", "crash", "problema", "debug", "fault"]):
        tools_to_use.append("logs")
    if any(word in msg for word in ["dispositivi", "entità", "stato", "acceso", "spento", "luce", "switch", "sensore", "attivi"]):
        tools_to_use.append("states")
    if any(word in msg for word in ["automazione", "automation", "script", "scene"]):
        tools_to_use.append("automations")
    if any(word in msg for word in ["documentazione", "come si", "come fare", "tutorial", "guida"]):
        tools_to_use.append("docs")
    if any(word in msg for word in ["cerca online", "cerca su internet", "web", "novità", "aggiornamento", "release"]):
        tools_to_use.append("web")
    if any(word in msg for word in ["accendi", "spegni", "attiva", "disattiva", "esegui", "trigger"]):
        tools_to_use.append("action")

    return list(dict.fromkeys(tools_to_use))


def gather_context(intent: list[str], message: str, queue: Queue) -> dict[str, str]:
    """Raccoglie solo i dati rilevanti alla domanda."""
    ctx = {"ha_config": "", "config_yaml": "", "extra_data": "", "question": message}
    extra_parts: list[str] = []
    msg = message.lower()
    base = Path(HA_CONFIG_PATH)

    skip_dirs = {
        "deps",
        "tts",
        "www",
        "backups",
        "custom_icons",
        ".storage",
        "themes",
        "blueprints",
        "__pycache__",
        ".git",
    }

    priority_files = {
        "automaz": ["automations.yaml"],
        "script": ["scripts.yaml"],
        "scene": ["scenes.yaml"],
        "sensor": ["sensors.yaml"],
        "problem": ["configuration.yaml"],
        "errore": ["configuration.yaml"],
        "shelly": ["configuration.yaml"],
        "luce": ["automations.yaml", "configuration.yaml"],
        "allar": ["configuration.yaml", "automations.yaml"],
        "energy": ["configuration.yaml"],
        "node-red": ["configuration.yaml"],
        "nodered": ["configuration.yaml"],
    }

    target_files = ["configuration.yaml"]
    for keyword, files in priority_files.items():
        if keyword in msg:
            target_files.extend(files)

    for entity_id in extract_explicit_entity_ids(message):
        domain = entity_id.split(".", 1)[0]
        if domain in {"automation", "script", "scene", "light", "switch", "sensor", "binary_sensor"}:
            if domain == "automation":
                target_files.append("automations.yaml")
            elif domain == "script":
                target_files.append("scripts.yaml")
            elif domain == "scene":
                target_files.append("scenes.yaml")
            else:
                target_files.append("configuration.yaml")

    target_files = list(dict.fromkeys(target_files))

    queue.put(json.dumps({"type": "tool_start", "tool": "ha_get_config", "msg": "⏳ Leggo configurazione HA..."}))
    cfg = tool_ha_get_config()
    try:
        cfg_dict = json.loads(cfg)
        cfg_dict.pop("components_sample", None)
        cfg = json.dumps(cfg_dict, indent=2, ensure_ascii=False)
    except Exception:
        pass
    ctx["ha_config"] = shorten_text(cfg, 1800)
    queue.put(json.dumps({"type": "tool_end", "preview": "Config HA letta"}))

    queue.put(json.dumps({"type": "tool_start", "tool": "read_config_file", "msg": "⏳ Leggo file YAML rilevanti..."}))
    yaml_parts: list[str] = []
    total_chars = 0

    def add_yaml_file(file_path: Path, label: str | None = None, per_file_limit: int = 2200) -> None:
        nonlocal total_chars
        try:
            text = safe_read_text(file_path).strip()
        except Exception:
            return
        if not text:
            return
        chunk = f"--- {label or str(file_path.relative_to(base))} ---\n{shorten_text(text, per_file_limit)}"
        if total_chars + len(chunk) > MAX_CONFIG_CHARS:
            return
        yaml_parts.append(chunk)
        total_chars += len(chunk)

    for file_name in target_files:
        candidate = base / file_name
        if not candidate.exists():
            matches = list(base.rglob(file_name))
            candidate = matches[0] if matches else None
        if candidate and candidate.exists() and candidate.is_file():
            add_yaml_file(candidate, str(candidate.relative_to(base)))

    if total_chars < MAX_CONFIG_CHARS:
        for file_path in sorted(base.rglob("*.yaml")):
            if any(skip in file_path.parts for skip in skip_dirs):
                continue
            if str(file_path.relative_to(base)) in [part.split(" ---", 1)[0].replace("--- ", "") for part in yaml_parts]:
                continue
            add_yaml_file(file_path, str(file_path.relative_to(base)), per_file_limit=900)
            if total_chars >= MAX_CONFIG_CHARS:
                break

    ctx["config_yaml"] = "\n\n".join(yaml_parts)
    queue.put(json.dumps({"type": "tool_end", "preview": f"{len(yaml_parts)} file letti, {total_chars} chars"}))

    need_states = "states" in intent or bool(extract_explicit_entity_ids(message))
    if need_states:
        queue.put(json.dumps({"type": "tool_start", "tool": "ha_get_states", "msg": "⏳ Leggo stati dispositivi..."}))
        explicit_entities = extract_explicit_entity_ids(message)
        if explicit_entities:
            state_blocks = []
            for entity_id in explicit_entities[:5]:
                state_blocks.append(f"=== STATO {entity_id} ===\n{tool_ha_get_states(entity_id)}")
            extra_parts.append("\n\n".join(state_blocks))
        else:
            domain = None
            if "luce" in msg or "light" in msg:
                domain = "light"
            elif "switch" in msg:
                domain = "switch"
            elif "sensore" in msg or "sensor" in msg:
                domain = "sensor"
            elif "automat" in msg:
                domain = "automation"
            extra_parts.append(f"=== STATI DISPOSITIVI ===\n{shorten_text(tool_ha_get_states(domain or ''), 1800)}")
        queue.put(json.dumps({"type": "tool_end", "preview": "Stati letti"}))

    if "logs" in intent:
        queue.put(json.dumps({"type": "tool_start", "tool": "ha_get_logs", "msg": "⏳ Leggo i log di errore..."}))
        extra_parts.append(f"=== LOG ERRORI ===\n{tool_ha_get_logs('80')}")
        queue.put(json.dumps({"type": "tool_end", "preview": "Log letti"}))

    if "docs" in intent:
        queue.put(json.dumps({"type": "tool_start", "tool": "search_ha_docs", "msg": "⏳ Cerco nella documentazione..."}))
        extra_parts.append(f"=== DOCUMENTAZIONE ===\n{shorten_text(tool_search_docs(message), 1800)}")
        queue.put(json.dumps({"type": "tool_end", "preview": "Documentazione trovata"}))

    if "web" in intent and ENABLE_WEB_SEARCH:
        queue.put(json.dumps({"type": "tool_start", "tool": "web_search", "msg": "🌐 Cerco sul web..."}))
        extra_parts.append(f"=== WEB ===\n{shorten_text(tool_web_search(message), 1200)}")
        queue.put(json.dumps({"type": "tool_end", "preview": "Risultati trovati"}))
    elif "web" in intent and not ENABLE_WEB_SEARCH:
        queue.put(json.dumps({"type": "action", "tool": "web_search", "input": "", "msg": "🔒 Web search disabilitato"}))

    if "action" in intent:
        action_hint = (
            "=== AZIONI HA ===\n"
            "Le azioni sui dispositivi non vengono eseguite automaticamente. "
            "Puoi usare ha_call_service_preview per preparare un'azione e poi confermarla via endpoint dedicato."
        )
        extra_parts.append(action_hint)

    ctx["extra_data"] = clip_extra(extra_parts, MAX_EXTRA_CHARS)
    return ctx


# ── FastAPI endpoints ──────────────────────────────────────────────────────
@app.post("/agent/chat")
async def agent_chat(body: dict[str, Any]):
    message = str(body.get("message", "")).strip()
    if not message:
        return {"error": "Messaggio vuoto"}

    queue: Queue = Queue()

    def run_agent() -> None:
        try:
            queue.put(json.dumps({"type": "thinking", "msg": "🔍 Analizzo la tua richiesta..."}))
            intent = detect_intent(message)
            ctx = gather_context(intent, message, queue)
            queue.put(json.dumps({"type": "thinking", "msg": "🤔 Elaboro la risposta..."}))
            prompt = ANALYSIS_PROMPT.format(**ctx)

            response = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": LLAMA_MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "options": {"num_ctx": 8192, "temperature": 0.1},
                },
                stream=True,
                timeout=600,
            )
            response.raise_for_status()

            queue.put(json.dumps({"type": "stream_start"}))
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    queue.put(json.dumps({"type": "stream_token", "token": token}))
                if chunk.get("done"):
                    break
            queue.put(json.dumps({"type": "stream_end"}))
        except Exception as exc:
            print(traceback.format_exc(), flush=True)
            queue.put(json.dumps({"type": "error", "msg": f"❌ {str(exc)[:300]}"}))
        finally:
            queue.put("__DONE__")

    Thread(target=run_agent, daemon=True).start()

    async def stream():
        while True:
            try:
                item = await asyncio.get_event_loop().run_in_executor(None, lambda: queue.get(timeout=600))
                if item == "__DONE__":
                    yield "data: [DONE]\n\n"
                    break
                yield f"data: {item}\n\n"
            except Empty:
                yield "data: " + json.dumps({"type": "heartbeat"}) + "\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.post("/agent/action/preview")
async def agent_action_preview(body: dict[str, Any]):
    service_call = str(body.get("service_call", "")).strip()
    entity_id = str(body.get("entity_id", "")).strip() or None
    if not service_call:
        raise HTTPException(400, "service_call obbligatorio")
    payload = build_pending_action(service_call, entity_id)
    return {
        "type": "confirmation_required",
        "message": f"Conferma richiesta per {service_call} su {entity_id or 'tutti'}",
        **payload,
    }


@app.post("/agent/action/confirm")
async def agent_action_confirm(body: dict[str, Any], x_confirm_token: str | None = Header(default=None)):
    require_confirm_token(x_confirm_token)
    action_id = str(body.get("action_id", "")).strip()
    if not action_id:
        raise HTTPException(400, "action_id obbligatorio")
    result = tool_ha_call_service_confirm(action_id)
    if result.startswith("Errore"):
        raise HTTPException(500, result)
    if result.startswith("Azione pending non trovata"):
        raise HTTPException(404, result)
    if result.startswith("Azioni HA disabilitate"):
        raise HTTPException(409, result)
    return {"status": "ok", "message": result}


@app.get("/agent/health")
async def health():
    ha_ok = False
    try:
        response = requests.get(f"{HA_HOST}/api/", headers=ha_headers(), timeout=5)
        ha_ok = response.status_code == 200
    except Exception:
        pass
    return {
        "status": "ok",
        "model": LLAMA_MODEL,
        "ha_host": HA_HOST,
        "ha_online": ha_ok,
        "allowed_origins": ALLOWED_ORIGINS,
        "allow_ha_actions": ALLOW_HA_ACTIONS,
        "pending_actions": len(PENDING_ACTIONS),
        "tools": AVAILABLE_TOOLS,
    }


@app.get("/agent/config")
async def agent_config():
    return {
        "model": LLAMA_MODEL,
        "embed_model": EMBED_MODEL,
        "ha_host": HA_HOST,
        "tools": AVAILABLE_TOOLS,
        "top_k": TOP_K,
        "ha_config_path": HA_CONFIG_PATH,
        "enable_web_search": ENABLE_WEB_SEARCH,
        "allow_ha_actions": ALLOW_HA_ACTIONS,
        "allowed_origins": ALLOWED_ORIGINS,
    }
