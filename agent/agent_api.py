"""
HA Agent API v3.1.0

Obiettivi:
- target canonico e alias esterni via YAML
- ranking stabile e meno dipendente da tuning manuale
- distinzione netta tra:
  - exact
  - direct_child
  - composite_related
  - partial_related
  - weak_related
- comportamento coerente su query find e azioni operative
- filtro dei weak_related nelle azioni operative
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import chromadb
import ollama as ollama_lib
import requests
import yaml
class OllamaEmbeddingFunction:
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
        return resp.json()["embeddings"]
import json as _json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


HA_HOST = os.getenv("HA_HOST", "").rstrip("/")
HA_TOKEN = os.getenv("HA_TOKEN", "")
_allow_ha_actions: bool = os.getenv("ALLOW_HA_ACTIONS", "false").lower() == "true"
_enable_web_search: bool = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"

CHROMA_HOST = os.getenv("CHROMA_HOST", "http://chromadb:8000")
CONFIG_COLLECTION = os.getenv("CONFIG_COLLECTION", "ha_config")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
ALIASES_FILE = os.getenv("AREA_ALIASES_FILE", "/app/area_aliases.yaml")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.1:8b")
AGENT_TOP_K = int(os.getenv("AGENT_TOP_K", "8"))

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://192.168.1.15:3000",
    ).split(",")
    if origin.strip()
]

ACTION_PATTERNS = {
    "turn_on": [r"\baccendi\b", r"\battiva\b", r"\bon\b", r"\bturn on\b"],
    "turn_off": [r"\bspegni\b", r"\bdisattiva\b", r"\boff\b", r"\bturn off\b"],
    "find": [r"\btrova\b", r"\bquali\b", r"\belenca\b", r"\bmostra\b", r"\brelative\b", r"\brelativi\b"],
}

STOPWORDS = {
    "accendi", "spegni", "attiva", "disattiva", "trova", "quali", "elenca",
    "mostra", "relative", "relativi", "entita", "entità", "luce", "luci",
    "light", "switch", "entity", "entities", "ci", "sono", "per", "di", "a",
    "da", "con", "il", "lo", "la", "i", "gli", "le", "un", "una",
    "del", "della", "delle", "degli", "dei", "al", "allo", "alla", "alle"
}

WEAK_LOCATION_TOKENS = {"su", "giu", "giu'", "terra", "esterno", "interno", "piano"}

RELATION_SCORE = {
    "exact": 1000,
    "direct_child": 700,
    "composite_related": 430,
    "partial_related": 250,
    "weak_related": 60,
    "none": 0,
}

DOMAIN_PRIORITY_FIND = {
    "light": 130,
    "switch": 125,
    "input_boolean": 118,
    "cover": 112,
    "fan": 108,
    "scene": 100,
    "script": 96,
    "binary_sensor": 90,
    "sensor": 82,
    "climate": 76,
    "media_player": 70,
    "button": 35,
    "device_tracker": 25,
    "event": 18,
    "update": 8,
}

GROUP_LABELS = {
    "light": "attuatori",
    "switch": "attuatori",
    "input_boolean": "attuatori",
    "cover": "attuatori",
    "fan": "attuatori",
    "scene": "attuatori",
    "script": "attuatori",
    "binary_sensor": "sensori_binari",
    "sensor": "sensori",
    "button": "comandi_diagnostica",
    "device_tracker": "tracking",
    "event": "eventi",
    "climate": "altro",
    "media_player": "altro",
    "update": "altro",
}

CONTROLLABLE_DOMAINS = {"light", "switch", "input_boolean", "fan", "cover", "scene", "script"}

app = FastAPI(title="HA Agent v3", version="3.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


class HistoryMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Richiesta utente")
    history: list[HistoryMessage] = Field(default_factory=list, description="Cronologia conversazione")


def get_embedding_function():
    return OllamaEmbeddingFunction(model=EMBED_MODEL, host=OLLAMA_HOST)


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.HttpClient:
    parsed = urlparse(CHROMA_HOST)
    return chromadb.HttpClient(
        host=parsed.hostname or "chromadb",
        port=parsed.port or 8000,
    )


def get_collection(name: str):
    return get_chroma_client().get_or_create_collection(
        name=name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def query_config_context(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    if top_k is None:
        top_k = AGENT_TOP_K
    try:
        collection = get_collection(CONFIG_COLLECTION)
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    hits: list[dict[str, Any]] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        hits.append(
            {
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", meta.get("page", 0)),
                "file_kind": meta.get("file_kind", meta.get("extension", "unknown")),
                "score": round(1 - (float(dist) / 2), 4),
                "text": doc,
            }
        )
    return hits


def require_ha_config() -> None:
    if not HA_HOST:
        raise HTTPException(status_code=500, detail="HA_HOST non configurato.")
    if not HA_TOKEN:
        raise HTTPException(status_code=500, detail="HA_TOKEN non configurato.")


def ha_headers() -> dict[str, str]:
    require_ha_config()
    return {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }


def ha_get(path: str) -> Any:
    url = f"{HA_HOST}/api{path}"
    response = requests.get(url, headers=ha_headers(), timeout=20)
    response.raise_for_status()
    return response.json()


def ha_get_text(path: str) -> str:
    url = f"{HA_HOST}/api{path}"
    response = requests.get(url, headers=ha_headers(), timeout=20)
    response.raise_for_status()
    return response.text


def ha_post(path: str, payload: dict[str, Any]) -> Any:
    url = f"{HA_HOST}/api{path}"
    response = requests.post(url, headers=ha_headers(), json=payload, timeout=20)
    response.raise_for_status()
    if response.text.strip():
        try:
            return response.json()
        except Exception:
            return {"raw": response.text}
    return {"status": "ok"}


def normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = value.replace(".", "_").replace("-", "_").replace("/", "_")
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9_àèéìòù]", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


@lru_cache(maxsize=1)
def load_area_aliases() -> dict[str, list[str]]:
    path = Path(ALIASES_FILE)
    if not path.exists():
        return {}

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    out: dict[str, list[str]] = {}
    for canonical, aliases in data.items():
        if not canonical:
            continue

        canonical_norm = normalize_text(str(canonical))
        alias_list = aliases or []
        alias_norms: list[str] = []

        for alias in alias_list:
            a = normalize_text(str(alias))
            if a:
                alias_norms.append(a)

        if canonical_norm and canonical_norm not in alias_norms:
            alias_norms.insert(0, canonical_norm)

        out[canonical_norm] = alias_norms

    return out


def detect_action(message: str) -> str:
    msg = message.lower()
    for action, patterns in ACTION_PATTERNS.items():
        if any(re.search(pattern, msg) for pattern in patterns):
            return action
    return "find"


def is_operational_action(action: str) -> bool:
    return action in {"turn_on", "turn_off"}


def smart_join_candidates(tokens: list[str]) -> list[str]:
    out: list[str] = []

    for i in range(len(tokens) - 1):
        a = tokens[i]
        b = tokens[i + 1]
        if a in STOPWORDS or b in STOPWORDS:
            continue
        out.append(f"{a}_{b}")

    for i in range(len(tokens) - 2):
        a = tokens[i]
        b = tokens[i + 1]
        c = tokens[i + 2]
        if a in STOPWORDS or b in STOPWORDS or c in STOPWORDS:
            continue
        out.append(f"{a}_{b}_{c}")

    return out


def extract_raw_candidates(message: str) -> list[str]:
    message_norm = normalize_text(message)
    raw_tokens = [t for t in message_norm.split("_") if t]

    candidates: list[str] = []

    for match in re.finditer(r"\b([a-zàèéìòù]+)_(\d+)\b", message_norm):
        left = match.group(1)
        num = match.group(2)
        candidates.append(f"{left}_{num}")
        candidates.append(left)
        candidates.append(num)

    candidates.extend(smart_join_candidates(raw_tokens))

    for t in raw_tokens:
        if t not in STOPWORDS:
            candidates.append(t)

    out: list[str] = []
    seen = set()
    for c in candidates:
        c = normalize_text(c)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_canonical_target(message: str) -> tuple[str | None, list[str]]:
    raw_candidates = extract_raw_candidates(message)
    aliases_map = load_area_aliases()

    for candidate in raw_candidates:
        for canonical, aliases in aliases_map.items():
            if candidate in aliases:
                return canonical, raw_candidates

    for candidate in raw_candidates:
        if re.fullmatch(r"[a-zàèéìòù]+_\d+", candidate):
            return candidate, raw_candidates

    for candidate in raw_candidates:
        if "_" in candidate and candidate not in WEAK_LOCATION_TOKENS:
            return candidate, raw_candidates

    for candidate in raw_candidates:
        if candidate not in WEAK_LOCATION_TOKENS and not candidate.isdigit():
            return candidate, raw_candidates

    return None, raw_candidates


def split_composite_parts(text: str) -> list[str]:
    """
    Esempio:
    bagno_su_e_anticamera_su_ingresso_0
    -> bagno_su, anticamera_su
    """
    text = normalize_text(text)
    if "_e_" not in text and "_and_" not in text:
        return []

    parts: list[str] = []
    chunks = re.split(r"_e_|_and_", text)
    for chunk in chunks:
        chunk = normalize_text(chunk)
        if not chunk:
            continue

        tokens = chunk.split("_")
        if len(tokens) >= 2:
            parts.append("_".join(tokens[:2]))
        else:
            parts.append(chunk)

    seen = set()
    out = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def enrich_entity(state_obj: dict[str, Any]) -> dict[str, Any]:
    entity_id = (state_obj.get("entity_id") or "").lower()
    attrs = state_obj.get("attributes", {}) or {}

    domain = entity_id.split(".", 1)[0] if "." in entity_id else ""
    object_id = entity_id.split(".", 1)[1] if "." in entity_id else entity_id
    object_id_norm = normalize_text(object_id)

    friendly_name_raw = str(attrs.get("friendly_name", ""))
    friendly_name_norm = normalize_text(friendly_name_raw)

    tokens = [t for t in object_id_norm.split("_") if t]
    composite_parts = split_composite_parts(object_id_norm)
    is_composite = len(composite_parts) > 1

    class_group = GROUP_LABELS.get(domain, "altro")
    is_controllable = domain in CONTROLLABLE_DOMAINS
    is_diagnostic = any(
        k in object_id_norm
        for k in [
            "riavvia",
            "restart",
            "reboot",
            "firmware",
            "sovra_potenza",
            "surriscaldamento",
            "sovracorrente",
            "sovratensione",
        ]
    )

    return {
        "raw": state_obj,
        "entity_id": entity_id,
        "domain": domain,
        "object_id": object_id,
        "object_id_norm": object_id_norm,
        "friendly_name": attrs.get("friendly_name"),
        "friendly_name_norm": friendly_name_norm,
        "state": state_obj.get("state"),
        "tokens": tokens,
        "is_composite": is_composite,
        "composite_parts": composite_parts,
        "group": class_group,
        "is_controllable": is_controllable,
        "is_diagnostic": is_diagnostic,
    }


def classify_relation(entity: dict[str, Any], canonical_target: str | None, raw_candidates: list[str]) -> tuple[str, list[str]]:
    if not canonical_target:
        return "none", []

    reasons: list[str] = []
    object_id_norm = entity["object_id_norm"]
    friendly_name_norm = entity["friendly_name_norm"]
    entity_id = entity["entity_id"]

    if object_id_norm == canonical_target:
        reasons.append(f"match target esatto object_id={canonical_target}")
        return "exact", reasons

    if friendly_name_norm == canonical_target:
        reasons.append(f"match target esatto friendly_name={canonical_target}")
        return "exact", reasons

    if entity_id.endswith(f".{canonical_target}"):
        reasons.append(f"match target esatto entity_id={canonical_target}")
        return "exact", reasons

    if object_id_norm.startswith(f"{canonical_target}_") and not object_id_norm.startswith(f"{canonical_target}_e_"):
        reasons.append(f"entità figlia diretta del target={canonical_target}")
        return "direct_child", reasons

    if entity["is_composite"] and canonical_target in entity["composite_parts"]:
        reasons.append(f"entità composita collegata al target={canonical_target}")
        return "composite_related", reasons

    if (
        f"{canonical_target}_" in object_id_norm
        or f"_{canonical_target}" in object_id_norm
        or canonical_target in object_id_norm
    ):
        reasons.append(f"target contenuto in object_id={canonical_target}")
        return "partial_related", reasons

    for cand in raw_candidates:
        if not cand or cand in WEAK_LOCATION_TOKENS:
            continue
        if cand in object_id_norm or cand in friendly_name_norm:
            reasons.append(f"match debole candidato={cand}")
            return "weak_related", reasons

    return "none", []


def domain_bonus_for_find(entity: dict[str, Any]) -> int:
    return DOMAIN_PRIORITY_FIND.get(entity["domain"], 0)


def relation_rank_value(relation: str) -> int:
    return {
        "exact": 5,
        "direct_child": 4,
        "composite_related": 3,
        "partial_related": 2,
        "weak_related": 1,
        "none": 0,
    }.get(relation, 0)


def score_entity_v3(entity: dict[str, Any], canonical_target: str | None, raw_candidates: list[str], action: str) -> tuple[int, list[str], int, str]:
    relation, reasons = classify_relation(entity, canonical_target, raw_candidates)
    if relation == "none":
        return 0, [], 0, relation

    exact_boost = RELATION_SCORE[relation]
    score = exact_boost

    object_id_norm = entity["object_id_norm"]
    friendly_name_norm = entity["friendly_name_norm"]
    domain = entity["domain"]

    # segnali lessicali aggiuntivi
    for cand in raw_candidates:
        if not cand:
            continue

        if cand in WEAK_LOCATION_TOKENS:
            if cand in object_id_norm:
                score += 4
                reasons.append(f"token locale debole in object_id={cand}")
            elif cand in friendly_name_norm:
                score += 3
                reasons.append(f"token locale debole in friendly_name={cand}")
            continue

        if cand.isdigit():
            if f"_{cand}" in object_id_norm or object_id_norm.endswith(cand):
                score += 8
                reasons.append(f"numero debole in object_id={cand}")
            continue

        if len(cand) >= 5:
            if cand in object_id_norm:
                score += 55
                reasons.append(f"token forte in object_id={cand}")
            if cand in entity["entity_id"]:
                score += 48
                reasons.append(f"token forte in entity_id={cand}")
            if cand in friendly_name_norm:
                score += 38
                reasons.append(f"token forte in friendly_name={cand}")
        else:
            if cand in object_id_norm:
                score += 10
                reasons.append(f"token breve in object_id={cand}")
            if cand in friendly_name_norm:
                score += 8
                reasons.append(f"token breve in friendly_name={cand}")

    if action == "find":
        score += domain_bonus_for_find(entity)
        reasons.append(f"priorità dominio find={domain}")

        if domain in {"light", "switch"} and relation == "exact":
            score += 60
            reasons.append("bonus core entity controllabile")

        if domain in {"sensor", "binary_sensor"} and relation == "direct_child":
            score += 25
            reasons.append("bonus figlia diretta leggibile")

        if domain in {"light", "switch", "sensor", "binary_sensor"}:
            if canonical_target and (canonical_target in object_id_norm or canonical_target in friendly_name_norm):
                score += 20
                reasons.append("bonus leggibilità target")

        if domain == "climate":
            if canonical_target and canonical_target not in friendly_name_norm:
                score -= 260
                reasons.append("penalità climate con friendly_name poco esplicativo")

        if domain == "update":
            score -= 140
            reasons.append("penalità dominio update")

        if domain == "button":
            score -= 80
            reasons.append("penalità button in query find")

        if relation == "composite_related":
            score -= 90
            reasons.append("penalità entità composita")

    if is_operational_action(action):
        if domain == "light":
            score += 80
            reasons.append("priorità azione su light")
        elif domain == "switch":
            score += 55
            reasons.append("priorità azione su switch")
        elif domain == "input_boolean":
            score += 30
            reasons.append("priorità azione su input_boolean")
        elif not entity["is_controllable"]:
            score -= 35
            reasons.append("penalità dominio non attuabile")

        if entity["is_diagnostic"]:
            score -= 25
            reasons.append("penalità diagnostica/non target")

    return score, reasons[:8], exact_boost, relation


def find_matching_entities_v3(
    states: list[dict[str, Any]],
    message: str,
    action: str,
    limit: int = 30,
) -> tuple[list[dict[str, Any]], str | None, list[str]]:
    canonical_target, raw_candidates = resolve_canonical_target(message)

    matches: list[dict[str, Any]] = []
    for state_obj in states:
        entity = enrich_entity(state_obj)
        score, reasons, exact_boost, relation = score_entity_v3(
            entity, canonical_target, raw_candidates, action
        )
        if score <= 0:
            continue

        matches.append(
            {
                "entity_id": entity["entity_id"],
                "state": entity["state"],
                "friendly_name": entity["friendly_name"],
                "domain": entity["domain"],
                "score": score,
                "exact_boost": exact_boost,
                "relation": relation,
                "relation_rank": relation_rank_value(relation),
                "group": entity["group"],
                "is_controllable": entity["is_controllable"],
                "is_composite": entity["is_composite"],
                "composite_parts": entity["composite_parts"],
                "reasons": reasons,
            }
        )

    def sort_key(item: dict[str, Any]):
        core_bonus = 1 if item["domain"] in {"light", "switch"} and item["relation"] == "exact" else 0
        return (
            -item["relation_rank"],
            -core_bonus,
            -item["exact_boost"],
            -domain_bonus_for_find({"domain": item["domain"]}) if action == "find" else 0,
            -item["score"],
            item["entity_id"],
        )

    matches.sort(key=sort_key)
    return matches[:limit], canonical_target, raw_candidates


def keep_find_match(match: dict[str, Any]) -> bool:
    relation = match.get("relation", "none")
    score = match.get("score", 0)
    domain = match.get("domain", "")

    if relation == "exact":
        return True

    if relation == "direct_child":
        return True

    if relation == "composite_related":
        if domain in {"update", "button"}:
            return score >= 650
        return score >= 320

    if relation == "partial_related":
        return score >= 260

    if relation == "weak_related":
        return False

    return False


def choose_controllable_entity(matches: list[dict[str, Any]]) -> dict[str, Any] | None:
    for match in matches:
        if match.get("domain") in CONTROLLABLE_DOMAINS:
            return match
    return matches[0] if matches else None


def build_entity_summary(matches: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_matches": len(matches),
        "entities_preview": [
            {
                "entity_id": m["entity_id"],
                "friendly_name": m["friendly_name"],
                "state": m["state"],
                "domain": m["domain"],
                "group": m["group"],
                "relation": m["relation"],
                "score": m["score"],
                "reasons": m["reasons"],
            }
            for m in matches[:8]
        ],
    }


def build_grouped_entities(matches: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for m in matches:
        grouped.setdefault(m["group"], []).append(
            {
                "entity_id": m["entity_id"],
                "friendly_name": m["friendly_name"],
                "state": m["state"],
                "domain": m["domain"],
                "relation": m["relation"],
                "score": m["score"],
            }
        )
    return grouped


def execute_ha_action(action: str, target_entity_id: str) -> dict[str, Any]:
    domain = target_entity_id.split(".", 1)[0]

    if not _allow_ha_actions:
        return {
            "executed": False,
            "allowed": False,
            "note": (
                f"Azione riconosciuta ({action}) su {target_entity_id}, ma "
                "HA Actions disabilitato. Nessun comando è stato eseguito."
            ),
        }

    if action == "turn_on":
        service = "turn_on"
    elif action == "turn_off":
        service = "turn_off"
    else:
        return {
            "executed": False,
            "allowed": False,
            "note": f"Azione '{action}' non eseguibile direttamente.",
        }

    try:
        result = ha_post(f"/services/{domain}/{service}", {"entity_id": target_entity_id})
        return {
            "executed": True,
            "allowed": True,
            "note": f"Comando eseguito: {domain}.{service} su {target_entity_id}.",
            "service_result": result,
        }
    except Exception as exc:
        return {
            "executed": False,
            "allowed": True,
            "note": f"Errore esecuzione comando su {target_entity_id}: {exc}",
        }


def build_operational_note(action: str, target: dict[str, Any] | None, action_note: str) -> str:
    if target is None:
        return action_note
    friendly = target.get("friendly_name") or target.get("entity_id")
    return f"Entità selezionata per l'azione: {target['entity_id']} ({friendly}). {action_note}"


def should_include_config_context(action: str, matches: list[dict[str, Any]]) -> bool:
    if is_operational_action(action):
        return False
    if action == "find" and matches:
        return False
    return True


def build_semantic_query(message: str, action: str, matches: list[dict[str, Any]]) -> str:
    if not matches:
        return message

    top = matches[0]
    parts = [
        top["entity_id"],
        str(top.get("friendly_name") or ""),
    ]
    for m in matches[1:4]:
        parts.append(m["entity_id"])

    if not is_operational_action(action):
        parts.append(message)

    return "\n".join([p for p in parts if p])


def format_config_context(config_hits: list[dict[str, Any]], max_chars: int = 700) -> str:
    if not config_hits:
        return "Nessun contesto config rilevante trovato."

    return "\n\n---\n\n".join(
        f"[Config {idx} | {hit['source']} | score={hit['score']:.0%}]\n{hit['text'][:max_chars]}"
        for idx, hit in enumerate(config_hits, start=1)
    )


def format_config_sample(config_hits: list[dict[str, Any]], max_chars: int = 1000) -> str:
    if not config_hits:
        return "Nessun campione disponibile."
    first = config_hits[0]
    return f"--- {first['source']} ---\n{first['text'][:max_chars]}"


# ── LLM response generation ─────────────────────────────────────────────────

_ACTIVE_STATES = {"on", "home", "open", "opening", "playing", "heating", "cooling",
                  "locked", "true", "active", "armed_home", "armed_away"}
_INACTIVE_STATES = {"off", "away", "closed", "idle", "unavailable", "unknown",
                    "false", "standby", "disarmed"}

_LIST_KEYWORDS = ["stampa", "elenca", "mostra", "lista", "visualizza", "dimmi",
                  "quante", "quanti", "quali sono", "vedi", "fammi vedere", "quali"]
_ENTITY_KEYWORDS = ["entità", "entita", "dispositivi", "dispositivo", "sensori",
                    "luci", "luce", "switch", "interruttori", "cover", "tapparelle",
                    "automazioni", "scene", "device", "entities", "prese"]
_ACTIVE_KEYWORDS = ["attive", "attivi", "accese", "accesi", "attivo", "attiva",
                    "acceso", "accesa", "on", "funzionanti", "aperte", "aperti"]
_INACTIVE_KEYWORDS = ["spente", "spenti", "off", "inattive", "inattivi",
                      "spento", "spenta", "disattivate", "ferme", "chiuse", "chiusi"]
_ALL_KEYWORDS = ["tutte", "tutti", "tutto", "all", "ogni"]

# Mappa parola chiave → dominio HA
_DOMAIN_KEYWORDS: dict[str, str] = {
    "luce": "light",
    "luci": "light",
    "light": "light",
    "lights": "light",
    "switch": "switch",
    "interruttore": "switch",
    "interruttori": "switch",
    "presa": "switch",
    "prese": "switch",
    "cover": "cover",
    "tapparella": "cover",
    "tapparelle": "cover",
    "sensore": "sensor",
    "sensori": "sensor",
    "sensor": "sensor",
    "binary_sensor": "binary_sensor",
    "automazione": "automation",
    "automazioni": "automation",
    "automation": "automation",
    "scena": "scene",
    "scene": "scene",
    "script": "script",
    "media_player": "media_player",
    "climate": "climate",
    "clima": "climate",
    "termostato": "climate",
}


_LOG_KEYWORDS = [
    "log", "errori nei log", "log di sistema", "error log", "errore home assistant",
    "cosa c'è nel log", "cosa dice il log", "mostra log", "mostra i log",
    "stampa log", "vedi log", "controlla log", "leggi log",
    "warning", "critical", "warning nel log", "problemi nel log",
    "errori di sistema", "log errori",
]

_DIAGNOSTIC_KEYWORDS = [
    "risolvi", "risolvere", "risolvilo", "risolviamo", "come risolvo", "come si risolve",
    "come fixo", "come faccio a risolvere", "aiutami a risolvere",
    "leggi la configurazione", "leggi config", "leggi la config", "leggi il file",
    "verifica tu", "controlla tu", "verifica lei", "controlla lei",
    "analizza", "analizzami", "dimmi cosa c'è", "cosa trovi", "cosa vedi",
    "spiegami", "spiega l'errore", "cosa significa l'errore",
    "dimmi come", "come posso", "come devo", "cosa devo fare",
]

_ENTITY_ID_RE = re.compile(r'\b([a-z_]+\.[a-z0-9_]+)\b')


def detect_diagnostic_intent(message: str) -> bool:
    msg = message.lower()
    return any(kw in msg for kw in _DIAGNOSTIC_KEYWORDS)


def extract_entity_ids_from_history(history: list[HistoryMessage]) -> list[str]:
    """Estrae pattern entity_id (dominio.nome) dalla cronologia."""
    seen: set[str] = set()
    result: list[str] = []
    for msg in reversed(history):  # messaggi più recenti prima
        for eid in _ENTITY_ID_RE.findall(msg.content):
            if eid not in seen and not eid.startswith("http"):
                seen.add(eid)
                result.append(eid)
        if len(result) >= 10:
            break
    return result


def gather_diagnostic_context(message: str, history: list[HistoryMessage]) -> str:
    """Raccoglie: log HA + configurazione (ChromaDB) + stati entità per una risposta diagnostica completa."""
    parts: list[str] = []

    # 1. Log di HA — sempre incluso in modalità diagnostica
    log_raw = fetch_ha_error_log(max_lines=50)
    parts.append(f"=== LOG DI HOME ASSISTANT (ultime righe) ===\n{log_raw}")

    # 2. Estrai entity ID da history + messaggio corrente
    all_history_text = message + " " + " ".join(h.content for h in history[-6:])
    entity_ids = list(dict.fromkeys(_ENTITY_ID_RE.findall(all_history_text)))
    # Filtra ID troppo generici (es. "v.1", "is.number")
    entity_ids = [e for e in entity_ids if "." in e and len(e.split(".")[0]) > 1 and len(e.split(".")[1]) > 2][:6]

    # 3. Cerca ChromaDB: ogni entity_id + parole chiave del messaggio
    config_hits: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()

    queries = entity_ids[:3] + [message]
    for query in queries:
        for hit in query_config_context(query, top_k=4):
            key = (hit["source"], hit["chunk_index"])
            if key not in seen_keys:
                seen_keys.add(key)
                config_hits.append(hit)

    config_hits.sort(key=lambda x: x["score"], reverse=True)

    if config_hits:
        chunks = []
        for hit in config_hits[:5]:
            chunks.append(
                f"[File: {hit['source']} | rilevanza={hit['score']:.0%}]\n{hit['text'][:800]}"
            )
        parts.append("=== CONFIGURAZIONE TROVATA IN CHROMADB ===\n\n" + "\n\n---\n\n".join(chunks))
    else:
        parts.append("=== CONFIGURAZIONE ===\nNessun file di configurazione trovato per le entità coinvolte.")

    # 4. Stato attuale delle entità da HA
    if entity_ids:
        state_lines: list[str] = []
        for eid in entity_ids[:6]:
            try:
                sd = ha_get(f"/states/{eid}")
                name = (sd.get("attributes") or {}).get("friendly_name", eid)
                state = sd.get("state", "unknown")
                attrs = {k: v for k, v in list((sd.get("attributes") or {}).items())[:8]}
                state_lines.append(f"  - {eid} ({name}): stato={state!r}, attr={attrs}")
            except Exception as exc:
                state_lines.append(f"  - {eid}: non raggiungibile ({exc})")
        parts.append("=== STATO ATTUALE ENTITÀ ===\n" + "\n".join(state_lines))

    return "\n\n".join(parts)


def detect_log_intent(message: str) -> bool:
    msg = message.lower()
    return any(kw in msg for kw in _LOG_KEYWORDS)


def fetch_ha_error_log(max_lines: int = 80) -> str:
    try:
        raw = ha_get_text("/error_log")
    except Exception as exc:
        return f"Impossibile recuperare il log: {exc}"
    lines = [l for l in raw.splitlines() if l.strip()]
    if not lines:
        return "Il log di Home Assistant è vuoto."
    recent = lines[-max_lines:]
    return "\n".join(recent)


def detect_list_intent(message: str) -> tuple[str | None, str | None, str | None]:
    """Rileva query di tipo 'mostra/elenca [luci/switch/...] [attive/spente/tutte]'.
    Restituisce (intent, state_filter, domain_filter).
    intent in {'list_all','list_by_state'}, state_filter in {'on','off',None}, domain_filter in {'light','switch',...,None}."""
    msg = message.lower()
    has_list = any(kw in msg for kw in _LIST_KEYWORDS)
    has_entity = any(kw in msg for kw in _ENTITY_KEYWORDS)

    if not (has_list or has_entity):
        return None, None, None

    # Rileva il dominio specifico dalla query
    domain_filter: str | None = None
    for kw, domain in _DOMAIN_KEYWORDS.items():
        if kw in msg:
            domain_filter = domain
            break

    if any(kw in msg for kw in _ACTIVE_KEYWORDS):
        return "list_by_state", "on", domain_filter
    if any(kw in msg for kw in _INACTIVE_KEYWORDS):
        return "list_by_state", "off", domain_filter
    if has_list and has_entity:
        return "list_all", None, domain_filter
    return None, None, None


def filter_entities_by_state(
    states: list[dict[str, Any]],
    state_filter: str | None,
    domain_filter: str | None = None,
) -> list[dict[str, Any]]:
    result = states
    # Filtra per dominio se specificato
    if domain_filter:
        result = [s for s in result if (s.get("entity_id") or "").split(".")[0] == domain_filter]
    # Filtra per stato
    if state_filter == "on":
        result = [s for s in result if s.get("state", "").lower() in _ACTIVE_STATES]
    elif state_filter == "off":
        result = [s for s in result if s.get("state", "").lower() in _INACTIVE_STATES]
    return result


def format_entity_list_for_llm(
    entities: list[dict[str, Any]], state_filter: str | None, limit: int = 80
) -> str:
    label = {
        "on": "attive/accese",
        "off": "spente/inattive",
        None: "totali",
    }.get(state_filter, "")

    lines = [f"Entità {label}: {len(entities)} (mostrando prime {min(limit, len(entities))})"]
    by_domain: dict[str, list[str]] = {}
    for s in entities[:limit]:
        domain = s.get("entity_id", "").split(".")[0]
        name = (s.get("attributes", {}) or {}).get("friendly_name") or s.get("entity_id", "")
        state = s.get("state", "")
        by_domain.setdefault(domain, []).append(f"  - {name}: {state}")

    for domain, items in sorted(by_domain.items()):
        lines.append(f"\n[{domain.upper()}]")
        lines.extend(items[:20])
        if len(items) > 20:
            lines.append(f"  ... e altri {len(items) - 20}")

    return "\n".join(lines)


def build_llm_context(
    message: str,
    matches: list[dict[str, Any]],
    action: str,
    action_note: str,
    config_context: str,
) -> str:
    parts: list[str] = []

    if matches:
        match_lines = [f"Entità trovate per la richiesta ({len(matches)}):"]
        for m in matches[:15]:
            name = m.get("friendly_name") or m.get("entity_id", "")
            match_lines.append(
                f"  - {m['entity_id']} ({name}): stato={m['state']}, relazione={m['relation']}"
            )
        parts.append("\n".join(match_lines))

    if action_note and action_note != "Nessuna azione eseguita.":
        parts.append(f"Azione eseguita: {action_note}")

    if config_context and "omesso" not in config_context and "Nessun contesto" not in config_context:
        parts.append(f"Contesto configurazione:\n{config_context[:1500]}")

    return "\n\n".join(parts) if parts else ""


_SYSTEM_PROMPT_DEFAULT = (
    "Sei un assistente esperto di Home Assistant con accesso diretto alla configurazione e agli stati del sistema. "
    "Rispondi SEMPRE in italiano.\n\n"
    "REGOLE:\n"
    "1. Usa TUTTI i dati nel contesto e nella cronologia per rispondere nel modo più completo possibile.\n"
    "2. Sii conciso e diretto. Usa elenchi puntati quando hai più elementi.\n"
    "3. Non inventare entità, stati o configurazioni non presenti nei dati.\n"
    "4. Tieni conto della cronologia per rispondere a domande di follow-up."
)

_SYSTEM_PROMPT_DIAGNOSTIC = """Sei un esperto di Home Assistant. Hai accesso al log di sistema, alla configurazione YAML indicizzata e agli stati delle entità. Rispondi SEMPRE in italiano.

OBIETTIVO: analizza il log e la configurazione, poi per ogni errore dai la causa esatta e il codice corretto.

ESEMPIO DI RISPOSTA CORRETTA:
---
### Problema 1: TemplateError su sensor.aria_qualita
**Causa:** il template usa `| round(0)` su un valore che può essere `None` quando `input_select.aria` non ha opzione corrispondente.
**Codice attuale (in `sensors/aria.yaml`):**
```jinja2
{{ opt[sel] | round(0) }}
```
**Fix da applicare:**
```jinja2
{{ (opt[sel] | default(0)) | round(0) }}
```
**File da modificare:** `config/sensors/aria.yaml`
---

REGOLE:
1. Usa SEMPRE il formato sopra: Causa → Codice attuale → Fix → File.
2. NON scrivere "verifica tu", "assicurati di", "controlla tu" — hai i dati, analizzali direttamente.
3. Se il codice originale è nei dati di configurazione, copialo esattamente prima di mostrare il fix.
4. Se l'entità non è nella configurazione indicizzata, scrivilo esplicitamente e indica il tipo di file dove cercare (es. "il file non è in ChromaDB — cercalo in `config/sensors/` o `config/templates/`").
5. Per warning di unità mancante: mostra il blocco YAML del sensore e aggiungi `unit_of_measurement: °C`.
6. Per entità mancanti: elenca i file ChromaDB che la referenziano e proponi come aggiornare o ricreare l'entità."""


def generate_llm_answer(
    message: str,
    context: str,
    history: list[HistoryMessage] | None = None,
    diagnostic: bool = False,
) -> str:
    try:
        client = ollama_lib.Client(host=OLLAMA_HOST)
        system_prompt = _SYSTEM_PROMPT_DIAGNOSTIC if diagnostic else _SYSTEM_PROMPT_DEFAULT
        if context:
            user_prompt = f"CONTESTO:\n{context}\n\nDOMANDA: {message}"
        else:
            user_prompt = f"DOMANDA: {message}"

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if history:
            max_history = 6 if diagnostic else 10
            for h in history[-max_history:]:
                messages.append({"role": h.role, "content": h.content})
        messages.append({"role": "user", "content": user_prompt})

        ctx_size = 8192 if diagnostic else 6144
        max_tokens = 900 if diagnostic else 800
        response = client.chat(
            model=LLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.1, "num_predict": max_tokens, "num_ctx": ctx_size},
        )
        return response["message"]["content"].strip()
    except Exception as exc:
        return f"(Risposta LLM non disponibile: {exc})"


@app.get("/health")
def health() -> dict[str, Any]:
    out: dict[str, Any] = {
        "api": "ok",
        "ha_host": HA_HOST,
        "ha_token_configured": bool(HA_TOKEN),
        "allow_ha_actions": _allow_ha_actions,
        "enable_web_search": _enable_web_search,
        "config_collection": CONFIG_COLLECTION,
        "embed_model": EMBED_MODEL,
        "matcher_version": "v3.1",
        "aliases_file": ALIASES_FILE,
        "aliases_loaded": len(load_area_aliases()),
    }

    try:
        if HA_HOST and HA_TOKEN:
            _ = ha_get("/")
            out["home_assistant"] = "ok"
        else:
            out["home_assistant"] = "not_configured"
    except Exception as exc:
        out["home_assistant"] = f"error: {exc}"

    try:
        collection = get_collection(CONFIG_COLLECTION)
        out["config_chunks"] = collection.count()
    except Exception as exc:
        out["config_chunks"] = f"error: {exc}"

    return out


@app.post("/settings/ha-actions/toggle")
def toggle_ha_actions() -> dict[str, Any]:
    global _allow_ha_actions
    _allow_ha_actions = not _allow_ha_actions
    return {"allow_ha_actions": _allow_ha_actions}


@app.post("/settings/web-search/toggle")
def toggle_web_search() -> dict[str, Any]:
    global _enable_web_search
    _enable_web_search = not _enable_web_search
    return {"enable_web_search": _enable_web_search}


@app.post("/chat")
def agent_chat(req: AgentChatRequest) -> dict[str, Any]:
    message = req.message.strip()
    history = req.history
    if not message:
        raise HTTPException(status_code=400, detail="Messaggio vuoto.")

    try:
        states = ha_get("/states")
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Errore lettura stati Home Assistant: {exc}",
        ) from exc

    # ── Gestione query "log / errori di sistema" ────────────────────────────
    if detect_log_intent(message):
        # Usa gather_diagnostic_context che include log + config + stati
        diag_context = gather_diagnostic_context(message, history or [])
        answer = generate_llm_answer(message, diag_context, history, diagnostic=True)
        return {
            "message": message,
            "answer": answer,
            "detected_action": "log",
            "ha_summary": {"total_entities": len(states)},
            "matched_entities": [],
            "grouped_entities": {},
            "selected_target": None,
            "config_context": diag_context[:500],
            "config_sample": "",
            "note": "Analisi log + configurazione HA.",
            "canonical_target": None,
            "query_candidates": [],
        }

    # ── Gestione query diagnostica (risolvi/verifica/leggi config) ──────────
    if detect_diagnostic_intent(message):
        diag_context = gather_diagnostic_context(message, history or [])
        answer = generate_llm_answer(message, diag_context, history, diagnostic=True)
        return {
            "message": message,
            "answer": answer,
            "detected_action": "diagnostic",
            "ha_summary": {"total_entities": len(states)},
            "matched_entities": [],
            "grouped_entities": {},
            "selected_target": None,
            "config_context": diag_context[:500] if diag_context else "",
            "config_sample": "",
            "note": "Risposta diagnostica basata su configurazione + stati HA.",
            "canonical_target": None,
            "query_candidates": [],
        }

    # ── Gestione query "lista entità [attive/spente/tutte]" ──────────────────
    list_intent, state_filter, domain_filter = detect_list_intent(message)
    if list_intent:
        filtered = filter_entities_by_state(states, state_filter, domain_filter)
        entity_list_context = format_entity_list_for_llm(filtered, state_filter)
        answer = generate_llm_answer(message, entity_list_context, history)
        return {
            "message": message,
            "answer": answer,
            "detected_action": "list",
            "ha_summary": {
                "total_entities": len(states),
                "filtered_entities": len(filtered),
                "entities_preview": [
                    {
                        "entity_id": s.get("entity_id"),
                        "friendly_name": (s.get("attributes") or {}).get("friendly_name"),
                        "state": s.get("state"),
                        "domain": (s.get("entity_id") or "").split(".")[0],
                    }
                    for s in filtered[:8]
                ],
            },
            "matched_entities": [],
            "grouped_entities": {},
            "selected_target": None,
            "config_context": entity_list_context,
            "config_sample": "",
            "note": f"Query lista: {list_intent}, dominio: {domain_filter or 'tutti'}, stato: {state_filter or 'qualsiasi'}",
            "canonical_target": None,
            "query_candidates": [],
        }

    # ── Entity matching standard ─────────────────────────────────────────────
    action = detect_action(message)

    matches, canonical_target, raw_candidates = find_matching_entities_v3(
        states=states,
        message=message,
        action=action,
        limit=30,
    )

    if is_operational_action(action):
        matches = [
            m for m in matches
            if m.get("relation") in {"exact", "direct_child", "composite_related", "partial_related"}
        ]

    if action == "find":
        matches = [m for m in matches if keep_find_match(m)]

    matches = matches[:20]

    entity_summary = build_entity_summary(matches)
    grouped_entities = build_grouped_entities(matches)

    target = None
    action_note = "Nessuna azione eseguita."

    if is_operational_action(action):
        target = choose_controllable_entity(matches)
        if target is None:
            action_note = (
                f"Ho rilevato un intento di tipo '{action}', ma non ho trovato "
                "un'entità Home Assistant sufficientemente pertinente."
            )
        else:
            exec_result = execute_ha_action(action, target["entity_id"])
            action_note = exec_result["note"]

    include_config = should_include_config_context(action, matches)
    config_hits: list[dict[str, Any]] = []
    config_context = ""
    config_sample = ""

    if include_config:
        semantic_query = build_semantic_query(message, action, matches)
        config_hits = query_config_context(semantic_query)
        config_context = format_config_context(config_hits)
        config_sample = format_config_sample(config_hits)
    else:
        if action == "find" and matches:
            config_context = "Contesto config omesso: ricerca entità già risolta tramite matching diretto."
            config_sample = ""
        else:
            config_context = "Contesto config omesso: richiesta operativa già risolta."
            config_sample = ""

    final_note = (
        build_operational_note(action, target, action_note)
        if is_operational_action(action)
        else action_note
    )

    # ── Genera risposta in linguaggio naturale ───────────────────────────────
    llm_context = build_llm_context(message, matches, action, final_note, config_context)
    answer = generate_llm_answer(message, llm_context, history)

    return {
        "message": message,
        "answer": answer,
        "detected_action": action,
        "ha_summary": {
            "total_entities": len(states),
            **entity_summary,
        },
        "matched_entities": matches,
        "grouped_entities": grouped_entities,
        "selected_target": target,
        "config_hits": config_hits,
        "config_context": config_context,
        "config_sample": config_sample,
        "note": final_note,
        "semantic_search_enabled": True,
        "config_context_included": include_config,
        "query_candidates": raw_candidates,
        "canonical_target": canonical_target,
    }


@app.post("/chat/stream")
def agent_chat_stream(req: AgentChatRequest) -> StreamingResponse:
    """Versione streaming di /chat via SSE: invia token LLM man mano che vengono generati."""
    message = req.message.strip()
    history = req.history
    if not message:
        raise HTTPException(status_code=400, detail="Messaggio vuoto.")

    def _sse(data: dict) -> str:
        return f"data: {_json.dumps(data, ensure_ascii=False)}\n\n"

    def generate():
        yield _sse({"type": "status", "text": "Connessione stabilita..."})

        try:
            states = ha_get("/states")
        except Exception as exc:
            yield _sse({"type": "error", "text": f"Errore lettura stati HA: {exc}"})
            return

        yield _sse({"type": "status", "text": f"Stati HA: {len(states)} entità"})

        diagnostic = False
        context = ""
        detected_action = "find"
        matched_count = 0

        if detect_log_intent(message):
            yield _sse({"type": "status", "text": "Recupero log di Home Assistant..."})
            context = gather_diagnostic_context(message, history or [])
            detected_action = "log"
            diagnostic = True

        elif detect_diagnostic_intent(message):
            yield _sse({"type": "status", "text": "Analisi configurazione in corso..."})
            context = gather_diagnostic_context(message, history or [])
            detected_action = "diagnostic"
            diagnostic = True

        else:
            list_intent, state_filter, domain_filter = detect_list_intent(message)
            if list_intent:
                filtered = filter_entities_by_state(states, state_filter, domain_filter)
                context = format_entity_list_for_llm(filtered, state_filter)
                detected_action = "list"
                matched_count = len(filtered)
            else:
                detected_action = detect_action(message)
                matches, canonical_target, raw_candidates = find_matching_entities_v3(
                    states=states, message=message, action=detected_action, limit=30
                )
                if is_operational_action(detected_action):
                    matches = [
                        m for m in matches
                        if m.get("relation") in {"exact", "direct_child", "composite_related", "partial_related"}
                    ]
                if detected_action == "find":
                    matches = [m for m in matches if keep_find_match(m)]
                matches = matches[:20]
                matched_count = len(matches)

                action_note = "Nessuna azione eseguita."
                target = None
                if is_operational_action(detected_action):
                    target = choose_controllable_entity(matches)
                    if target:
                        exec_result = execute_ha_action(detected_action, target["entity_id"])
                        action_note = exec_result["note"]
                    else:
                        action_note = f"Nessuna entità trovata per '{detected_action}'."

                include_config = should_include_config_context(detected_action, matches)
                config_hits: list[dict[str, Any]] = []
                if include_config:
                    semantic_query = build_semantic_query(message, detected_action, matches)
                    config_hits = query_config_context(semantic_query)
                    config_str = format_config_context(config_hits)
                else:
                    config_str = ""

                final_note = (
                    build_operational_note(detected_action, target, action_note)
                    if is_operational_action(detected_action) else action_note
                )
                context = build_llm_context(message, matches, detected_action, final_note, config_str)

        yield _sse({"type": "status", "text": "Generazione risposta LLM..."})

        try:
            llm_client = ollama_lib.Client(host=OLLAMA_HOST)
            system_prompt = _SYSTEM_PROMPT_DIAGNOSTIC if diagnostic else _SYSTEM_PROMPT_DEFAULT
            user_prompt = f"CONTESTO:\n{context}\n\nDOMANDA: {message}" if context else f"DOMANDA: {message}"

            llm_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
            max_hist = 6 if diagnostic else 10
            for h in (history or [])[-max_hist:]:
                llm_messages.append({"role": h.role, "content": h.content})
            llm_messages.append({"role": "user", "content": user_prompt})

            ctx_size = 8192 if diagnostic else 6144
            max_tokens = 900 if diagnostic else 800

            for chunk in llm_client.chat(
                model=LLAMA_MODEL,
                messages=llm_messages,
                stream=True,
                options={"temperature": 0.1, "num_predict": max_tokens, "num_ctx": ctx_size},
            ):
                token = chunk["message"]["content"]
                if token:
                    yield _sse({"type": "token", "content": token})

        except Exception as exc:
            yield _sse({"type": "error", "text": f"Errore LLM: {exc}"})
            return

        yield _sse({"type": "done", "detected_action": detected_action, "matched_count": matched_count})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )