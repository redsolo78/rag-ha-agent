"""
HA Agent API — LangChain ReAct Agent specializzato in Home Assistant
Endpoint SSE: POST /agent/chat   { "message": "..." }
"""

import asyncio
import json
import os
import re
import traceback
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import Any

import chromadb
import requests
import yaml
from chromadb.utils import embedding_functions
from duckduckgo_search import DDGS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks.base import BaseCallbackHandler
from langchain.tools import Tool
from langchain_ollama import OllamaLLM

# ── Config ─────────────────────────────────────────────────────────────────
OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://ollama:11434")
CHROMA_HOST  = os.getenv("CHROMA_HOST",  "http://chromadb:8000")
HA_HOST      = os.getenv("HA_HOST",      "http://homeassistant:8123")
HA_TOKEN     = os.getenv("HA_TOKEN",     "")
LLAMA_MODEL  = os.getenv("LLAMA_MODEL",  "llama3.1:8b")
EMBED_MODEL  = os.getenv("EMBED_MODEL",  "all-MiniLM-L6-v2")
COLLECTION   = os.getenv("COLLECTION",   "documents")
HA_CONFIG_PATH = os.getenv("HA_CONFIG_PATH", "/ha_config")
TOP_K             = int(os.getenv("TOP_K", "8"))
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"

app = FastAPI(title="HA Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── ChromaDB client ─────────────────────────────────────────────────────────
def get_chroma_collection():
    from urllib.parse import urlparse
    p = urlparse(CHROMA_HOST)
    client = chromadb.HttpClient(host=p.hostname or "chromadb", port=p.port or 8000)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(name=COLLECTION, embedding_function=embed_fn)

# ── HA helpers ───────────────────────────────────────────────────────────────
def ha_headers():
    return {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}

def ha_get(path: str) -> dict | list | str:
    try:
        r = requests.get(f"{HA_HOST}/api{path}", headers=ha_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def ha_post(path: str, data: dict) -> dict:
    try:
        r = requests.post(f"{HA_HOST}/api{path}", headers=ha_headers(), json=data, timeout=10)
        r.raise_for_status()
        return r.json() if r.text else {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}

# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════

def tool_search_docs(query: str) -> str:
    """Cerca nella documentazione di Home Assistant indicizzata in ChromaDB."""
    try:
        col = get_chroma_collection()
        results = col.query(query_texts=[query], n_results=TOP_K)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        if not docs:
            return "Nessun documento rilevante trovato."
        out = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            src = meta.get("source", "?").split("/")[-1]
            out.append(f"[Doc {i+1} - {src}]\n{doc[:600]}")
        return "\n\n---\n\n".join(out)
    except Exception as e:
        return f"Errore ricerca docs: {e}"


def tool_ha_get_states(filter_str: str = "") -> str:
    """
    Legge gli stati delle entità di Home Assistant.
    Puoi filtrare per dominio (es: 'light', 'switch', 'sensor', 'automation')
    o per entity_id specifico (es: 'light.soggiorno').
    Lascia vuoto per avere un riepilogo generale.
    """
    try:
        filter_str = filter_str.strip().lower()

        # Entità specifica
        if filter_str and "." in filter_str:
            result = ha_get(f"/states/{filter_str}")
            if "error" in result:
                return f"Entità non trovata: {filter_str}"
            return json.dumps({
                "entity_id": result.get("entity_id"),
                "state": result.get("state"),
                "attributes": result.get("attributes", {})
            }, indent=2, ensure_ascii=False)

        # Tutti gli stati o filtrati per dominio
        all_states = ha_get("/states")
        if isinstance(all_states, dict) and "error" in all_states:
            return f"Errore HA: {all_states['error']}"

        if filter_str:
            states = [s for s in all_states if s["entity_id"].startswith(filter_str + ".")]
        else:
            states = all_states

        if not states:
            return f"Nessuna entità trovata per: '{filter_str}'"

        # Raggruppa per dominio
        domains = {}
        for s in states:
            domain = s["entity_id"].split(".")[0]
            domains.setdefault(domain, []).append({
                "id": s["entity_id"],
                "state": s["state"],
                "name": s.get("attributes", {}).get("friendly_name", "")
            })

        lines = []
        for domain, entities in sorted(domains.items()):
            lines.append(f"\n## {domain.upper()} ({len(entities)} entità)")
            for e in entities[:20]:  # max 20 per dominio
                lines.append(f"  - {e['id']}: {e['state']}  ({e['name']})")
            if len(entities) > 20:
                lines.append(f"  ... e altri {len(entities)-20}")

        return f"Totale entità: {len(states)}\n" + "\n".join(lines)

    except Exception as e:
        err = traceback.format_exc()
        print(f"[TOOL ERROR ha_get_states] {err}", flush=True)
        return f"Errore: {err}"


def tool_ha_get_logs(lines_str: str = "50") -> str:
    """
    Legge gli ultimi log di errore di Home Assistant.
    Specifica il numero di righe da leggere (default 50).
    """
    try:
        n = int(lines_str.strip()) if lines_str.strip().isdigit() else 50
        result = ha_get("/error_log")
        if isinstance(result, dict) and "error" in result:
            return f"Errore: {result['error']}"
        log_text = str(result)
        lines = log_text.splitlines()
        return "\n".join(lines[-n:]) if len(lines) > n else log_text
    except Exception as e:
        return f"Errore lettura log: {e}"


def tool_ha_call_service(input_str: str) -> str:
    """
    Chiama un servizio di Home Assistant.
    Formato input: 'domain.service entity_id'
    Esempi:
      'light.turn_on light.soggiorno'
      'switch.turn_off switch.pompa'
      'automation.trigger automation.goodnight'
    ATTENZIONE: questa azione modifica lo stato dei dispositivi reali!
    """
    try:
        parts = input_str.strip().split()
        if len(parts) < 1:
            return "Formato: 'domain.service entity_id'"
        service_call = parts[0]
        entity_id    = parts[1] if len(parts) > 1 else None

        if "." not in service_call:
            return f"Formato non valido: '{service_call}'. Usa 'domain.service'"

        domain, service = service_call.split(".", 1)
        data = {"entity_id": entity_id} if entity_id else {}
        result = ha_post(f"/services/{domain}/{service}", data)

        if isinstance(result, dict) and "error" in result:
            return f"Errore servizio: {result['error']}"
        return f"✅ Servizio {service_call} eseguito su {entity_id or 'tutti'}"
    except Exception as e:
        return f"Errore: {e}"


def tool_read_config_file(filename: str) -> str:
    """
    Legge un file di configurazione di Home Assistant.
    Specifica il nome del file (es: 'configuration.yaml', 'automations.yaml').
    Puoi anche specificare un path relativo (es: 'packages/lights.yaml').
    """
    try:
        filename = filename.strip().strip('"\'')
        base = Path(HA_CONFIG_PATH)

        # Cerca il file
        target = base / filename
        if not target.exists():
            # Prova a cercarlo ricorsivamente
            matches = list(base.rglob(filename))
            if not matches:
                # Lista file disponibili
                available = [str(f.relative_to(base)) for f in base.rglob("*.yaml")][:30]
                if not available:
                    return f"ATTENZIONE: La cartella di configurazione HA sembra vuota o non montata correttamente. Percorso: {base}. Usa ha_get_config per leggere la configurazione via API."
                return f"File '{filename}' non trovato.\nFile YAML disponibili:\n" + "\n".join(available)
            target = matches[0]

        content = target.read_text(encoding="utf-8", errors="ignore")
        # Limita la lunghezza
        if len(content) > 4000:
            content = content[:4000] + f"\n\n... [troncato, file di {len(content)} caratteri]"
        return f"📄 {target.relative_to(base)}\n\n{content}"

    except Exception as e:
        return f"Errore lettura file: {e}"


def tool_list_config_files(path: str = "") -> str:
    """
    Lista i file di configurazione di Home Assistant disponibili.
    Puoi specificare una sottocartella (es: 'packages', 'automations').
    """
    try:
        base = Path(HA_CONFIG_PATH) / path.strip()
        if not base.exists():
            return f"Cartella non trovata: {path}"

        files = []
        for f in sorted(base.rglob("*")):
            if f.is_file() and f.suffix in (".yaml", ".json", ".txt"):
                rel = str(f.relative_to(Path(HA_CONFIG_PATH)))
                size = f.stat().st_size
                files.append(f"  {rel} ({size} bytes)")

        if not files:
            return "Nessun file di configurazione trovato."
        return f"File di configurazione ({len(files)} totali):\n" + "\n".join(files[:50])
    except Exception as e:
        return f"Errore: {e}"


def tool_web_search(query: str) -> str:
    """
    Cerca informazioni aggiornate sul web.
    Utile per trovare soluzioni a problemi specifici, novità di Home Assistant,
    integrazioni, blueprint, best practice.
    """
    if not ENABLE_WEB_SEARCH:
        return "⚠️ Web search disabilitato. Imposta ENABLE_WEB_SEARCH=true nel file .env per abilitarlo."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query + " Home Assistant", max_results=5))
        if not results:
            return "Nessun risultato trovato."
        out = []
        for r in results:
            out.append(f"**{r.get('title','?')}**\n{r.get('body','')[:400]}\n🔗 {r.get('href','')}")
        return "\n\n---\n\n".join(out)
    except Exception as e:
        return f"Errore web search: {e}"


def tool_ha_get_config(dummy: str = "") -> str:
    """
    Legge la configurazione generale di Home Assistant (versione, fuso orario, unità di misura, ecc.)
    """
    try:
        result = ha_get("/config")
        if isinstance(result, dict) and "error" in result:
            return f"Errore: {result['error']}"
        keys = ["version", "location_name", "time_zone", "unit_system",
                "components", "config_dir", "latitude", "longitude"]
        summary = {k: result.get(k) for k in keys if k in result}
        components = result.get("components", [])
        summary["components_count"] = len(components)
        summary["components_sample"] = sorted(components)[:30]
        return json.dumps(summary, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Errore: {e}"



# ── ReAct Prompt ────────────────────────────────────────────────────────────
# Prompt semplice per analisi diretta (no ReAct loop)
ANALYSIS_PROMPT = """Sei un esperto senior di Home Assistant che analizza un sistema reale. Rispondi SEMPRE in italiano.

CONTESTO: Hai accesso COMPLETO al sistema reale dell'utente: file di configurazione, stati dispositivi, log. Non sei un chatbot generico — sei un tecnico con accesso diretto al sistema.

REGOLE ASSOLUTE — MAI VIOLARLE:
1. MAI dire "non posso sapere" o "non ho accesso" — hai TUTTI i dati qui sotto
2. MAI rimandare alla documentazione online o al supporto esterno — sei TU il supporto
3. MAI dare consigli generici — ogni risposta deve citare file, entità o errori SPECIFICI trovati nei dati
4. Se trovi un errore nei log, cerca il file responsabile nei YAML e indica dove si trova
5. Se suggerisci una correzione YAML, mostra il codice PRIMA (errato) e DOPO (corretto)
6. Se vedi Node-RED attivo, NON suggerire automazioni HA per cose già gestite da Node-RED
7. Usa i nomi ESATTI delle entità che vedi nei dati — mai nomi inventati

FORMATO RISPOSTA:
- Inizia con un sommario diretto (es: "Ho trovato 3 errori:")
- Per ogni problema: file/entità coinvolta, causa, soluzione con codice YAML se applicabile
- Concludi con le azioni prioritarie numerate

=== CONFIGURAZIONE SISTEMA ===
{ha_config}

=== FILE DI CONFIGURAZIONE YAML ===
{config_yaml}

=== DATI AGGIUNTIVI (dispositivi, log, documentazione) ===
{extra_data}

=== DOMANDA ===
{question}

Rispondi analizzando i dati reali sopra. Ogni affermazione deve essere supportata da dati concreti trovati nei file o nei log."""

# ── SSE Callback Handler ─────────────────────────────────────────────────────
class SSECallbackHandler(BaseCallbackHandler):
    """Cattura gli eventi dell'agente e li mette in una queue per SSE."""

    def __init__(self, queue: Queue):
        self.queue = queue
        self._seen = {}  # tool+input → count

    def _send(self, event_type: str, **data):
        self.queue.put(json.dumps({"type": event_type, **data}))

    def on_agent_action(self, action, **kwargs):
        key = f"{action.tool}:{str(action.tool_input)[:100]}"
        self._seen[key] = self._seen.get(key, 0) + 1
        if self._seen[key] > 2:
            self._send("action", tool=action.tool,
                       input=str(action.tool_input)[:200],
                       msg=f"⚠️ Loop rilevato su **{action.tool}** — forzo conclusione...")
            # Metti un segnale nella queue per terminare
            self.queue.put('{"type":"final","answer":"Ho analizzato quello che potevo. Il file configuration.yaml è stato letto. Usa \'ha_get_config\' via API per info sul sistema, o chiedi qualcosa di più specifico."}')
        else:
            self._send("action", tool=action.tool,
                       input=str(action.tool_input)[:200],
                       msg=f"🔧 Uso strumento: **{action.tool}**")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "?")
        self._send("tool_start", tool=tool_name, msg=f"⏳ Esecuzione {tool_name}...")

    def on_tool_end(self, output, **kwargs):
        preview = str(output)[:150].replace("\n", " ")
        self._send("tool_end", preview=preview, msg=f"✅ Risultato ricevuto")

    def on_agent_finish(self, finish, **kwargs):
        self._send("final", answer=finish.return_values.get("output", ""))

    def on_llm_error(self, error, **kwargs):
        self._send("error", msg=f"❌ Errore LLM: {str(error)[:200]}")

    def on_tool_error(self, error, **kwargs):
        self._send("error", msg=f"❌ Errore strumento: {str(error)[:200]}")


# ── Intent detection: quale tool usare ───────────────────────────────────────
def detect_intent(message: str) -> list[str]:
    """Determina quali tool chiamare in base alla domanda."""
    msg = message.lower()
    tools_to_use = []

    # Sempre utile avere la config base
    tools_to_use.append("ha_config")

    if any(w in msg for w in ["configurazione", "yaml", "config", "file", "problema", "errore", "analizza", "controlla"]):
        tools_to_use.append("config_yaml")

    if any(w in msg for w in ["log", "errore", "crash", "problema", "debug", "fault"]):
        tools_to_use.append("logs")

    if any(w in msg for w in ["dispositivi", "entità", "stato", "acceso", "spento", "luce", "switch", "sensore", "attivi"]):
        tools_to_use.append("states")

    if any(w in msg for w in ["automazione", "automation", "script", "scene"]):
        tools_to_use.append("automations")

    if any(w in msg for w in ["documentazione", "come si", "come fare", "tutorial", "guida"]):
        tools_to_use.append("docs")

    if any(w in msg for w in ["cerca online", "cerca su internet", "web", "novità", "aggiornamento", "release"]):
        tools_to_use.append("web")

    return tools_to_use


def gather_context(intent: list[str], message: str, queue: Queue) -> dict:
    """
    Raccoglie solo i dati rilevanti alla domanda — niente di superfluo.
    Su CPU ogni token conta: mandiamo il minimo necessario per rispondere bene.
    """
    ctx = {"ha_config": "", "config_yaml": "", "extra_data": "", "question": message}
    extra_parts = []

    SKIP_DIRS = {"deps", "tts", "www", "backups", "custom_icons", ".storage",
                 "themes", "blueprints", "__pycache__", ".git", "custom_components"}

    # ── Parole chiave dalla domanda ─────────────────────────────────────────
    msg = message.lower()

    # File YAML prioritari in base alla domanda
    PRIORITY_FILES = {
        "automaz": ["automations.yaml"],
        "script":  ["scripts.yaml"],
        "scene":   ["scenes.yaml"],
        "sensor":  ["sensors.yaml"],
        "log":     [],
        "errore":  [],
        "problem": ["configuration.yaml"],
        "shelly":  ["configuration.yaml"],
        "luce":    ["automations.yaml"],
        "allar":   ["configuration.yaml", "automations.yaml"],
        "energy":  ["configuration.yaml"],
    }

    target_files = ["configuration.yaml"]  # sempre incluso
    for kw, files in PRIORITY_FILES.items():
        if kw in msg:
            target_files.extend(files)
    target_files = list(dict.fromkeys(target_files))  # deduplica

    # ── 1. Config HA (sempre, è leggera) ────────────────────────────────────
    queue.put(json.dumps({"type": "tool_start", "tool": "ha_get_config",
                          "msg": "⏳ Leggo configurazione HA..."}))
    cfg = tool_ha_get_config()
    # Tieni solo le info essenziali — rimuovi la lista components enorme
    try:
        import json as _j
        cfg_dict = _j.loads(cfg)
        cfg_dict.pop("components_sample", None)
        cfg = _j.dumps(cfg_dict, indent=2, ensure_ascii=False)
    except Exception:
        pass
    ctx["ha_config"] = cfg[:1500]
    queue.put(json.dumps({"type": "tool_end", "preview": "Config HA letta"}))

    # ── 2. File YAML selettivi ───────────────────────────────────────────────
    base = Path(HA_CONFIG_PATH)
    all_yaml = []
    total_chars = 0
    MAX_YAML_CHARS = 4000  # budget YAML

    # Prima i file prioritari
    for fname in target_files:
        fpath = base / fname
        if not fpath.exists():
            # cerca ricorsivamente
            matches = list(base.rglob(fname))
            if matches:
                fpath = matches[0]
            else:
                continue
        try:
            content_f = fpath.read_text(encoding="utf-8", errors="ignore").strip()
            if not content_f:
                continue
            if len(content_f) > 2000:
                content_f = content_f[:2000] + f"\n...[troncato]"
            chunk = f"--- {fname} ---\n{content_f}"
            if total_chars + len(chunk) <= MAX_YAML_CHARS:
                all_yaml.append(chunk)
                total_chars += len(chunk)
        except Exception:
            pass

    # Se rimane budget, aggiungi altri file
    if total_chars < MAX_YAML_CHARS:
        for f in sorted(base.rglob("*.yaml")):
            if any(skip in f.parts for skip in SKIP_DIRS):
                continue
            fname = f.name
            if fname in target_files:
                continue  # già incluso
            try:
                content_f = f.read_text(encoding="utf-8", errors="ignore").strip()
                if not content_f or len(content_f) < 10:
                    continue
                content_f = content_f[:800]
                chunk = f"--- {fname} ---\n{content_f}"
                if total_chars + len(chunk) > MAX_YAML_CHARS:
                    break
                all_yaml.append(chunk)
                total_chars += len(chunk)
            except Exception:
                pass

    queue.put(json.dumps({"type": "tool_end",
                          "preview": f"{len(all_yaml)} file letti, {total_chars} chars"}))
    ctx["config_yaml"] = "\n\n".join(all_yaml)

    # ── 3. Stati dispositivi (solo se richiesti o analisi generale) ─────────
    need_states = any(w in msg for w in [
        "dispositiv", "entit", "stato", "acceso", "spento", "luce", "switch",
        "sensore", "attiv", "analizza", "quanti", "elenco"
    ])
    if need_states:
        queue.put(json.dumps({"type": "tool_start", "tool": "ha_get_states",
                              "msg": "⏳ Leggo stati dispositivi..."}))
        # Leggi per dominio rilevante invece di tutto
        domain = None
        if "luce" in msg or "light" in msg:
            domain = "light"
        elif "switch" in msg:
            domain = "switch"
        elif "sensore" in msg or "sensor" in msg:
            domain = "sensor"
        elif "automat" in msg:
            domain = "automation"
        states = tool_ha_get_states(domain or "")
        # Limita a 1500 chars
        extra_parts.append(f"=== STATI DISPOSITIVI ===\n{states[:1500]}")
        queue.put(json.dumps({"type": "tool_end", "preview": "Stati letti"}))

    # ── 4. Log (solo se richiesti) ──────────────────────────────────────────
    if "logs" in intent:
        queue.put(json.dumps({"type": "tool_start", "tool": "ha_get_logs",
                              "msg": "⏳ Leggo i log di errore..."}))
        logs = tool_ha_get_logs("50")
        extra_parts.append(f"=== LOG ERRORI ===\n{logs[:2000]}")
        queue.put(json.dumps({"type": "tool_end", "preview": "Log letti"}))

    # ── 5. Documentazione (solo se richiesta) ───────────────────────────────
    if "docs" in intent:
        queue.put(json.dumps({"type": "tool_start", "tool": "search_ha_docs",
                              "msg": "⏳ Cerco nella documentazione..."}))
        docs = tool_search_docs(message)
        extra_parts.append(f"=== DOCUMENTAZIONE ===\n{docs[:1500]}")
        queue.put(json.dumps({"type": "tool_end", "preview": "Documentazione trovata"}))

    # ── 6. Web search (solo se esplicitamente richiesto) ────────────────────
    if "web" in intent and ENABLE_WEB_SEARCH:
        queue.put(json.dumps({"type": "tool_start", "tool": "web_search",
                              "msg": "🌐 Cerco sul web..."}))
        web = tool_web_search(message)
        extra_parts.append(f"=== WEB ===\n{web[:1000]}")
        queue.put(json.dumps({"type": "tool_end", "preview": "Risultati trovati"}))
    elif "web" in intent and not ENABLE_WEB_SEARCH:
        queue.put(json.dumps({"type": "action", "tool": "web_search", "input": "",
                              "msg": "🔒 Web search disabilitato"}))

    ctx["extra_data"] = "\n\n".join(extra_parts)
    return ctx


# ── FastAPI endpoints ────────────────────────────────────────────────────────
@app.post("/agent/chat")
async def agent_chat(body: dict):
    message = body.get("message", "").strip()
    if not message:
        return {"error": "Messaggio vuoto"}

    q: Queue = Queue()

    def run_agent():
        try:
            q.put(json.dumps({"type": "thinking", "msg": "🔍 Analizzo la tua richiesta..."}))

            # 1. Determina intent
            intent = detect_intent(message)

            # 2. Raccoglie dati
            ctx = gather_context(intent, message, q)

            # 3. Chiama LLM con streaming
            q.put(json.dumps({"type": "thinking", "msg": "🤔 Elaboro la risposta..."}))
            prompt = ANALYSIS_PROMPT.format(**ctx)

            print(f"[AGENT] Invio prompt a Ollama ({len(prompt)} chars)...", flush=True)
            resp = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": LLAMA_MODEL, "prompt": prompt, "stream": True,
                      "options": {"num_ctx": 8192, "temperature": 0.1}},
                stream=True, timeout=600
            )
            resp.raise_for_status()
            print(f"[AGENT] Ollama risponde HTTP {resp.status_code}, inizio streaming...", flush=True)

            q.put(json.dumps({"type": "stream_start"}))
            token_count = 0
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    token_count += 1
                    q.put(json.dumps({"type": "stream_token", "token": token}))
                if chunk.get("done"):
                    print(f"[AGENT] Streaming done, {token_count} tokens inviati", flush=True)
                    break
            q.put(json.dumps({"type": "stream_end"}))

        except Exception as e:
            err_msg = f"❌ {str(e)[:300]}"
            print(f"[AGENT ERROR] {err_msg}", flush=True)
            print(traceback.format_exc(), flush=True)
            q.put(json.dumps({"type": "error", "msg": err_msg}))
        finally:
            q.put("__DONE__")

    thread = Thread(target=run_agent, daemon=True)
    thread.start()

    async def stream():
        while True:
            try:
                item = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: q.get(timeout=600)
                )
                if item == "__DONE__":
                    yield "data: [DONE]\n\n"
                    break
                yield f"data: {item}\n\n"
            except Empty:
                # Heartbeat per mantenere connessione SSE viva
                yield "data: " + json.dumps({"type": "heartbeat"}) + "\n\n"
                continue

    return StreamingResponse(stream(), media_type="text/event-stream",
                                headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.get("/agent/health")
async def health():
    ha_ok = False
    try:
        r = requests.get(f"{HA_HOST}/api/", headers=ha_headers(), timeout=5)
        ha_ok = r.status_code == 200
    except Exception:
        pass
    return {
        "status":    "ok",
        "model":     LLAMA_MODEL,
        "ha_host":   HA_HOST,
        "ha_online": ha_ok,
        "tools":     [t.name for t in TOOLS]
    }


@app.get("/agent/config")
async def agent_config():
    return {
        "model":        LLAMA_MODEL,
        "embed_model":  EMBED_MODEL,
        "ha_host":      HA_HOST,
        "tools":        [t.name for t in TOOLS],
        "top_k":        TOP_K
    }
