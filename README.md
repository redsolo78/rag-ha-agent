# RAG HA Agent — Versione 1.0

RAG HA Agent è un sistema AI locale basato su **Retrieval-Augmented Generation (RAG)** progettato per analizzare, interrogare e assistere un'istanza reale di **Home Assistant**.

Questa repository rappresenta la **versione 1.0**: una baseline stabile, già utilizzabile, pensata per validare l'integrazione end-to-end tra:

- interfaccia web
- backend RAG
- modello LLM locale via Ollama
- vector database ChromaDB
- agente specializzato per Home Assistant

L'obiettivo della v1 non è ancora l'automazione completa, ma la costruzione di una piattaforma locale, sicura e presentabile, su cui evolvere una v2 più intelligente e operativa.

---

## 1. Obiettivo del progetto

Il progetto nasce per trasformare Home Assistant da piattaforma configurata manualmente a sistema **assistito da AI**, capace di:

- leggere configurazioni e stato reale dell'ambiente
- rispondere in modo contestualizzato
- supportare il debug tecnico
- usare documentazione indicizzata come base di conoscenza locale
- preparare il terreno per azioni controllate nella versione successiva

---

## 2. Architettura della versione 1.0

La versione 1.0 è composta logicamente da 5 blocchi.

```text
Utente
  ↓
Frontend Web
  ↓
Backend RAG ─────────────┐
  ↓                      │
ChromaDB                 │
  ↓                      │
Ollama (LLM locale)      │
                         │
HA Agent ────────────────┘
  ↓
Home Assistant
```

### Componenti principali

- **Frontend**: interfaccia web per chat, upload documenti e diagnostica di base.
- **Backend**: API FastAPI che gestisce retrieval, prompt building e interrogazione del modello.
- **ChromaDB**: database vettoriale persistente per i chunk indicizzati.
- **Ollama**: runtime locale per LLM.
- **HA Agent**: API specializzata in Home Assistant per leggere configurazione, stati e log.

---

## 3. Cosa fa la v1.0

### Capacità già disponibili

- esecuzione completamente locale
- integrazione con Ollama
- integrazione con ChromaDB
- health endpoint del backend
- accesso all'istanza reale di Home Assistant tramite token API
- interrogazione di stati, log e configurazione di HA
- indicizzazione documentale generale nel backend
- CORS ristretto a origin esplicite
- protezione base degli endpoint amministrativi

### Limiti noti della v1.0

- la knowledge base non viene popolata automaticamente dalla config HA
- il retrieval è ancora semplice
- la raccolta del contesto nell'agent è utile ma non ancora “collection-aware”
- la piattaforma è pensata come **internal tool**, non ancora come prodotto generalizzato
- le azioni operative su HA devono restare controllate e disabilitate di default

---

## 4. Sicurezza e configurazione

La v1 usa variabili ambiente per separare configurazione locale e codice.

### Variabili principali

- `HA_HOST`: endpoint HTTP di Home Assistant
- `HA_TOKEN`: Long-Lived Access Token di Home Assistant
- `ALLOWED_ORIGINS`: origin browser autorizzate per il frontend
- `ADMIN_TOKEN`: protezione endpoint amministrativi del backend
- `ACTION_CONFIRM_TOKEN`: protezione endpoint di conferma azioni
- `ALLOW_HA_ACTIONS`: abilita/disabilita le azioni su Home Assistant
- `HA_CONFIG_HOST_PATH`: path locale della cartella config di Home Assistant da montare nel container

### Importante

Il file `.env` **non va committato**.  
Il file `.env.example` **va committato** con valori segnaposto.

---

## 5. Avvio rapido

### Prerequisiti

- Docker
- Docker Compose
- Home Assistant raggiungibile via rete
- token API di Home Assistant

### Setup

```bash
cp .env.example .env
# compilare .env con i valori reali
```

### Avvio stack

```bash
docker compose up -d --build
```

### Endpoint tipici

- frontend: `http://<IP_HOST>:3000`
- backend: `http://<IP_HOST>:8000`
- ha-agent docs: `http://<IP_HOST>:8100/docs`
- backend health: `http://<IP_HOST>:8000/health`
- agent health: `http://<IP_HOST>:8100/agent/health`

---

## 6. Test minimi consigliati

### Test backend

```bash
curl http://<IP_HOST>:8000/health
```

### Test agent

```bash
curl http://<IP_HOST>:8100/agent/health
```

### Test connettività Home Assistant dal container agent

```bash
docker exec -i rag-ha-agent python - <<'PY'
import os, requests

ha_host = os.getenv("HA_HOST")
ha_token = os.getenv("HA_TOKEN")

r = requests.get(
    f"{ha_host}/api/",
    headers={"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"},
    timeout=10,
)

print(r.status_code)
print(r.text[:200])
PY
```

Risultato atteso: `200` e `{"message":"API running."}`

---

## 7. Stato della v1.0

La versione 1.0 va considerata come:

- **stabile** per uso interno e demo tecnica
- **adatta** a presentazioni aziendali su architettura e direzione del progetto
- **non ancora completa** sul fronte ingest automatico, retrieval avanzato e automazione operativa

In altre parole, la v1 chiude bene il perimetro di:

- architettura locale
- integrazione tecnica
- sicurezza di base
- separazione dei componenti

---

## 8. Roadmap sintetica verso la v2.0

La direzione di evoluzione naturale del progetto è:

1. ingest automatico della configurazione HA
2. retrieval separato tra documenti e config
3. ranking migliore dei chunk
4. flusso esplicito propose → confirm → execute
5. audit trail delle azioni
6. agente più “instance-aware”

Per questa evoluzione è stata creata la directory [`v2/`](./v2), che contiene una proposta concreta di implementazione della versione 2.0 senza creare una nuova repository.

---

## 9. Commit consigliati per chiudere la v1.0

File da committare:

- `README.md`
- `backend/api.py`
- `agent/agent_api.py`
- `docker-compose.yml`
- `.env.example`

File da **non** committare:

- `.env`
- token reali
- path locali sensibili

---

## 10. Posizionamento finale

La versione 1.0 dimostra che è possibile integrare in modo credibile:

- LLM locale
- retrieval documentale
- introspezione dell'ambiente Home Assistant
- sicurezza minima e isolamento configurativo

La v2.0, contenuta nella cartella dedicata, è il passo successivo per trasformare il progetto da **PoC avanzato** a **internal platform più matura e realmente context-aware**.
