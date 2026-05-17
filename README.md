# RAG HA Agent

Assistente AI locale per **Home Assistant** basato su RAG (Retrieval-Augmented Generation).  
Risponde a domande sulla tua configurazione HA, indicizza i tuoi file di configurazione, e può eseguire azioni su Home Assistant con un flusso di conferma esplicito.

Tutto gira in locale — nessun dato esce dalla rete.

---

## Indice

- [Come funziona](#come-funziona)
- [Architettura](#architettura)
- [Prerequisiti](#prerequisiti)
- [Installazione e primo avvio](#installazione-e-primo-avvio)
- [Configurazione](#configurazione)
- [Flusso operativo dettagliato](#flusso-operativo-dettagliato)
  - [Indicizzazione documenti](#1-indicizzazione-documenti)
  - [Indicizzazione configurazione HA](#2-indicizzazione-configurazione-ha)
  - [Chat RAG](#3-chat-rag)
  - [Azioni su Home Assistant](#4-azioni-su-home-assistant)
- [API Reference](#api-reference)
- [Area aliases](#area-aliases)
- [Gestione modelli Ollama](#gestione-modelli-ollama)
- [Troubleshooting](#troubleshooting)

---

## Come funziona

```
                    ┌─────────────────────────────────────────────────────┐
                    │                     Frontend :13001                 │
                    │          (chat RAG + chat agent + upload)           │
                    └──────────────────┬──────────────────┬───────────────┘
                                       │                  │
                         ┌─────────────▼──────┐  ┌───────▼──────────────┐
                         │  Backend :18005     │  │  HA Agent :18103     │
                         │  (RAG engine)       │  │  (azioni su HA)      │
                         └──────┬──────┬───────┘  └──────┬───────┬───────┘
                                │      │                  │       │
                    ┌───────────▼──┐  ┌▼──────────────┐  │   ┌───▼──────┐
                    │  ChromaDB    │  │  Ollama        │  │   │  HA API  │
                    │  (vettori)   │  │  (LLM + embed) │◄─┘   │          │
                    └──────────────┘  └────────────────┘      └──────────┘
                         │
              ┌──────────┴──────────┐
              │ collection:         │ collection:
              │ "documents"         │ "ha_config"
              │ (PDF, MD, YAML...)  │ (config HA locale)
              └─────────────────────┘
```

Il sistema separa nettamente **conoscenza documentale generica** (manuali, PDF, guide) dalla **configurazione reale della tua istanza HA** (YAML, automazioni, script). Il retrieval interroga entrambe le fonti, deduplica i risultati e li reordina con un algoritmo ibrido BM25 + similarità semantica prima di passarli al modello LLM.

---

## Architettura

| Servizio | Porta | Descrizione |
|---|---|---|
| `ollama` | interno | LLM locale (llama3.1:8b) + embedding (all-MiniLM-L6-v2) |
| `chromadb` | interno | Database vettoriale persistente |
| `backend` | **18005** | API RAG: upload, indicizzazione, chat |
| `ha-agent` | **18103** | API agente HA: query stato, esecuzione azioni |
| `frontend` | **13001** | Interfaccia web |

I servizi comunicano su una rete Docker interna (`rag-v2-network`). Solo le porte elencate sono esposte all'host.

---

## Prerequisiti

- **Docker** e **Docker Compose** v2+
- Accesso alla rete locale dove gira Home Assistant
- Un **long-lived access token** di Home Assistant
- Almeno **8 GB di RAM** (il modello llama3.1:8b richiede ~5 GB)
- La cartella di configurazione HA accessibile dall'host (es. `/home/andrea/homeassistant`)

---

## Installazione e primo avvio

**1. Clona il repository**

```bash
git clone https://github.com/redsolo78/rag-ha-agent.git
cd rag-ha-agent
```

**2. Crea il file `.env`**

```bash
cp .env.example .env
```

Modifica `.env` con i tuoi valori (vedi sezione [Configurazione](#configurazione)).

**3. Avvia i container**

```bash
docker compose up -d --build
```

Al primo avvio Docker scarica le immagini e compila i container: può richiedere alcuni minuti.

**4. Scarica il modello LLM**

Dopo che Ollama è in esecuzione, scarica il modello:

```bash
docker exec -it rag-ha-agent-ollama-1 ollama pull llama3.1:8b
docker exec -it rag-ha-agent-ollama-1 ollama pull all-MiniLM-L6-v2
```

> Puoi usare qualsiasi modello disponibile su Ollama. Aggiorna `LLAMA_MODEL` e `EMBED_MODEL` nel `.env` di conseguenza.

**5. Verifica che tutto funzioni**

```bash
curl http://localhost:18005/health
curl http://localhost:18103/agent/health
```

Entrambi devono rispondere `"api": "ok"`.

**6. Indicizza la configurazione HA**

```bash
curl -X POST http://localhost:18005/admin/reindex-ha-config \
  -H "X-Admin-Token: <tuo-admin-token>" \
  -H "Content-Type: application/json" \
  -d '{}'
```

**7. Apri l'interfaccia web**

```
http://localhost:13001
```

---

## Configurazione

Tutte le variabili si impostano nel file `.env`.

### Home Assistant

| Variabile | Default | Descrizione |
|---|---|---|
| `HA_HOST` | `http://homeassistant:8123` | URL dell'istanza HA |
| `HA_TOKEN` | _(obbligatorio)_ | Long-lived access token HA |
| `HA_CONFIG_HOST_PATH` | `/home/andrea/homeassistant` | Percorso **sull'host** della cartella config HA — viene montata in sola lettura |

### Modelli

| Variabile | Default | Descrizione |
|---|---|---|
| `LLAMA_MODEL` | `llama3.1:8b` | Modello LLM da usare per le risposte |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Modello di embedding per i vettori |

### Retrieval

| Variabile | Default | Descrizione |
|---|---|---|
| `TOP_K` | `10` | Numero di chunk da passare al modello nella chat RAG |
| `AGENT_TOP_K` | `8` | Numero di chunk usati dall'agente HA |
| `MIN_RETRIEVAL_SCORE` | `0.30` | Score minimo (0–1) sotto il quale un chunk viene scartato |
| `RETRIEVAL_OVERSAMPLE` | `3` | Fattore di oversampling prima del reranking (TOP_K × 3) |

### Sicurezza

| Variabile | Default | Descrizione |
|---|---|---|
| `ALLOWED_ORIGINS` | `http://localhost:3000,...` | Origins CORS autorizzati (lista separata da virgole) |
| `ADMIN_TOKEN` | `change-me` | Token per gli endpoint admin (upload, delete, reindex) |
| `ACTION_CONFIRM_TOKEN` | `change-me` | Token da fornire per confermare un'azione su HA |
| `ALLOW_HA_ACTIONS` | `false` | Abilita l'esecuzione di azioni su HA (`true`/`false`) |
| `ENABLE_WEB_SEARCH` | `false` | Abilita la ricerca web nell'agente |
| `MAX_UPLOAD_MB` | `50` | Dimensione massima dei file caricabili |

---

## Flusso operativo dettagliato

### 1. Indicizzazione documenti

Puoi caricare documenti nella **collection documentale** (`documents`) tramite l'interfaccia web o l'API.

Formati supportati: `.txt`, `.md`, `.yaml`, `.yml`, `.json`, `.pdf`

```
File caricato
    │
    ▼
Lettura testo (UTF-8, tollerante agli errori)
    │
    ▼
Chunking intelligente
  - dimensione: 1200 caratteri
  - overlap: 200 caratteri
  - sentence-boundary aware (preferisce spezzare su ". " o "\n")
    │
    ▼
Embedding via Ollama (all-MiniLM-L6-v2)
    │
    ▼
Upsert in ChromaDB — collection "documents"
  - ID stabile: upload::<filename>::<idx>
  - Metadati: source, page, source_kind
```

### 2. Indicizzazione configurazione HA

La cartella config HA viene montata in sola lettura in `/ha_config` dentro il container `backend`.

Triggerabile via API:
```bash
curl -X POST http://localhost:18005/admin/reindex-ha-config \
  -H "X-Admin-Token: <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"config_root": "/ha_config", "collection_name": "ha_config"}'
```

```
Scansione ricorsiva /ha_config
    │
    ▼
Filtro file
  Estensioni ammesse: .yaml, .yml, .txt, .md
  JSON: solo file utili (.storage/core.*, automations.json, ...)
  Directory escluse: .git, __pycache__, deps, backups, tts, www,
                     translations, node_modules, build, .storage, ...
    │
    ▼
Per ogni file ammesso:
  ├── Lettura contenuto
  ├── Chunking (1200 char, overlap 200)
  ├── ID stabile: ha_config::<rel_path>::<idx>::<sha1[:12]>
  └── Metadati: source, path, extension, file_kind, chunk_index
    │
    ▼
Batch upsert in ChromaDB — collection "ha_config"
  (batch da max 4000 chunk per chiamata)
```

File categorizzati automaticamente (`file_kind`): `automation`, `script`, `scene`, `template`, `root_config`, `yaml`, `json`, `markdown`, `text`.

### 3. Chat RAG

Ogni messaggio nella chat RAG percorre questo flusso:

```
Domanda utente
    │
    ▼
Retrieval parallelo da entrambe le collection
  ├── query "documents"  → top_k × RETRIEVAL_OVERSAMPLE risultati
  └── query "ha_config"  → top_k × RETRIEVAL_OVERSAMPLE risultati
    │
    ▼
Filtro per score minimo (MIN_RETRIEVAL_SCORE = 0.30)
  (score = 1 - distanza_cosine / 2)
    │
    ▼
Deduplica
  (stessa collection + stesso source + stesso prefisso testuale)
    │
    ▼
Reranking ibrido BM25 + semantico
  score_finale = 0.6 × score_semantico + 0.4 × score_BM25_normalizzato
    │
    ▼
Selezione top_k chunk
    │
    ▼
Se nessun chunk supera il filtro → risposta onesta senza chiamare il modello
    │
    ▼
Costruzione prompt grounded
  - Ogni chunk etichettato: [Fonte N | nomefile | score% | tipo | pag.]
  - Regole esplicite: cita sempre la fonte, non inventare, usa ```yaml```
    │
    ▼
Chiamata Ollama (temperature=0.1, top_p=0.9)
    │
    ▼
Risposta con: testo + lista chunk usati + flag grounded
```

### 4. Azioni su Home Assistant

L'agente HA gestisce un flusso a tre fasi per garantire che le azioni siano intenzionali.

> Richiede `ALLOW_HA_ACTIONS=true` nel `.env`.

```
Utente: "Spegni le luci del salotto"
    │
    ▼
Risoluzione entità e area
  - Lettura alias da area_aliases.yaml
  - Query ChromaDB collection "ha_config" per trovare entità rilevanti
  - Ranking corrispondenze:
      exact → direct_child → composite_related → partial_related → weak_related
  - Le corrispondenze "weak" vengono filtrate nelle azioni operative
    │
    ▼
Proposta strutturata
  {
    "action": "light.turn_off",
    "entities": ["light.salotto_principale"],
    "confirm_required": true,
    "confirm_token_hint": "usa ACTION_CONFIRM_TOKEN per confermare"
  }
    │
    ▼
Utente conferma con token (ACTION_CONFIRM_TOKEN)
    │
    ▼
Esecuzione servizio HA via REST API
  POST /api/services/<domain>/<service>
    │
    ▼
Risposta con esito
```

Se `ALLOW_HA_ACTIONS=false`, l'agente descrive l'azione che eseguirebbe senza effettuarla.

---

## API Reference

### Backend — porta 18005

| Metodo | Endpoint | Auth | Descrizione |
|---|---|---|---|
| `GET` | `/health` | — | Stato di tutti i servizi |
| `GET` | `/config` | — | Configurazione attiva |
| `GET` | `/stats` | — | Statistiche chunk e documenti |
| `POST` | `/chat` | — | Chat RAG |
| `POST` | `/upload-document` | Admin | Carica documento nella collection docs |
| `POST` | `/upload-pdf` | Admin | Alias di `/upload-document` |
| `POST` | `/admin/reindex-ha-config` | Admin | Indicizza la config HA |
| `DELETE` | `/documents/all` | Admin | Svuota tutte le collection |
| `DELETE` | `/document/{name}` | Admin | Elimina un singolo documento |
| `DELETE` | `/admin/reset-documents-index` | Admin | Reset pulito collection docs |
| `DELETE` | `/admin/reset-ha-config-index` | Admin | Reset pulito collection ha_config |

Gli endpoint **Admin** richiedono l'header `X-Admin-Token: <ADMIN_TOKEN>`.

**Payload `/chat`:**
```json
{
  "question": "Come è configurato il sensore di temperatura?",
  "top_k": 10
}
```

### HA Agent — porta 18103

| Metodo | Endpoint | Descrizione |
|---|---|---|
| `GET` | `/agent/health` | Stato dell'agente e connessione HA |
| `POST` | `/agent/query` | Invia una query all'agente |
| `POST` | `/agent/confirm` | Conferma un'azione proposta |

---

## Area aliases

Il file `agent/area_aliases.yaml` mappa i nomi delle aree HA con i loro alias in linguaggio naturale.  
Questo permette all'agente di capire frasi come "spegni le luci del bagno di sopra" anche se l'area in HA si chiama `bagno_su`.

```yaml
bagno_su:
  - bagno_su
  - bagno su
  - bagno superiore

salotto:
  - salotto

cucina:
  - cucina
```

Modifica questo file aggiungendo le aree presenti nella tua installazione HA.

---

## Gestione modelli Ollama

Elenco modelli disponibili:
```bash
docker exec rag-ha-agent-ollama-1 ollama list
```

Scaricare un modello alternativo (es. llama3.2:3b per hardware meno potente):
```bash
docker exec rag-ha-agent-ollama-1 ollama pull llama3.2:3b
```

Poi aggiorna `LLAMA_MODEL=llama3.2:3b` nel `.env` e riavvia:
```bash
docker compose restart backend ha-agent
```

---

## Troubleshooting

**Il backend risponde ma Ollama non è pronto**  
Al primo avvio Ollama impiega qualche minuto. Il `healthcheck` di Docker aspetta automaticamente — verifica con `docker compose ps` che lo stato sia `healthy`.

**Nessun chunk trovato nelle risposte**  
Significa che le collection sono vuote. Carica almeno un documento o esegui il reindex della config HA.

**Score troppo basso, il modello non risponde**  
Abbassa `MIN_RETRIEVAL_SCORE` nel `.env` (es. `0.20`) e riavvia il backend.

**Errore CORS dal browser**  
Aggiungi l'origine del browser a `ALLOWED_ORIGINS` nel `.env` (es. `http://192.168.1.50:13001`).

**Agente non trova entità HA**  
Verifica che `HA_HOST` e `HA_TOKEN` siano corretti (`/agent/health` mostrerà l'errore).  
Aggiungi gli alias mancanti in `area_aliases.yaml` e riavvia `ha-agent`.

**Reindex lento o che si blocca**  
La cartella config HA contiene molti file. Verifica che `HA_CONFIG_HOST_PATH` punti alla cartella giusta e non all'intera home. Controlla i log con `docker compose logs backend -f`.
