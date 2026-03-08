# HA Assistant — RAG + HA Agent Completamente Locale

Sistema RAG con agente specializzato per Home Assistant.
Analizza la configurazione HA, legge i log, interroga i dispositivi e risponde in italiano.
Tutto locale — nessun dato inviato a servizi esterni.

## Stack
- **Ollama** — LLM locale (llama3.1:8b di default)
- **ChromaDB** — Vector database con documentazione HA (~41k chunks)
- **FastAPI** — Backend RAG
- **FastAPI** — HA Agent (porta 8100)
- **Nginx** — Frontend web

## Avvio rapido

```bash
# 1. Copia e configura l'env
cp .env.example .env
# Modifica HA_TOKEN e HA_HOST nel .env

# 2. Avvia i container
docker compose up -d

# 3. Aspetta che Ollama scarichi il modello (~5 min prima volta)
docker logs rag-ollama-init -f

# 4. Apri il browser
open http://localhost:3000
```

## Indicizzare la documentazione HA

```bash
# Scarica documentazione ufficiale HA
bash scripts/download_ha_docs.sh

# Indicizza (repo scaricata e depurata da "rumore")
docker exec rag-backend python ingest.py \
  --folder /app/data/ha_docs/ha_docs_clean \
  --chunk-size 1200 --chunk-overlap 300

# Indicizza integrazioni specifiche
bash scripts/download_integrations.sh
docker exec rag-backend python ingest.py \
  --folder /app/data/ha_integrations \
  --ext .md --chunk-size 1000 --chunk-overlap 150
```

## Configurazione .env

| Variabile | Default | Descrizione |
|-----------|---------|-------------|
| `LLAMA_MODEL` | `llama3.1:8b` | Modello Ollama da usare |
| `HA_TOKEN` | — | Long-lived token di Home Assistant |
| `HA_HOST` | — | URL di Home Assistant (es: `http://192.168.1.1:8123`) |
| `ENABLE_WEB_SEARCH` | `false` | Abilita ricerca web (DuckDuckGo) |
| `TRANSLATE_QUERY` | `false` | Traduce query in inglese prima del retrieval |

## Struttura

```
rag-ha-agent/
├── backend/          # FastAPI + ChromaDB + ingest
├── frontend/         # HTML + Nginx
├── agent/            # HA Agent FastAPI (porta 8100)
├── scripts/          # Script download documentazione HA
├── data/             # Documenti e DB (gitignored)
├── docker-compose.yml
└── .env.example
```

## Funzionalità HA Agent

- 📁 Legge tutti i file YAML della configurazione HA
- 🏠 Interroga l'API HA in tempo reale (stati, log, config)
- 📚 Cerca nella documentazione ufficiale HA (ChromaDB)
- 🔧 Può controllare dispositivi (con conferma)
- 🌐 Web search opzionale (DuckDuckGo, disabilitato di default)
