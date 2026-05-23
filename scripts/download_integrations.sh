#!/bin/bash
# Scarica SOLO le pagine di integrazione esatte per la tua installazione HA
# Zero rumore — solo documentazione rilevante al 100%
# Uso: bash scripts/download_integrations.sh

DOCS_DIR="./data/ha_integrations"
BASE_URL="https://raw.githubusercontent.com/home-assistant/home-assistant.io/current/source/_integrations"
BASE_URL_DOCS="https://raw.githubusercontent.com/home-assistant/home-assistant.io/current/source"

mkdir -p "$DOCS_DIR"
cd "$DOCS_DIR"

# Lista integrazioni installate nella tua HA
INTEGRATIONS=(
  # Dispositivi smart
  "shelly"
  "esphome"
  "hue"
  "netatmo"
  "fritzbox"
  "tplink"
  "yeelight"
  "ezviz"
  "onkyo"
  "androidtv_remote"
  "vesync"
  "meross_lan"

  # Network & sistema
  "adguard"
  "glances"
  "synology_dsm"
  "fritz"
  "upnp"
  "systemmonitor"

  # Energia & meteo
  "influxdb"
  "forecast_solar"
  "co2signal"
  "met"
  "zcsazzurro"

  # Automazione & logica
  "automation"
  "script"
  "scene"
  "template"
  "blueprint"
  "schedule"
  "command_line"
  "rest"
  "nodered"

  # Helper & input
  "input_boolean"
  "input_number"
  "input_select"
  "input_datetime"
  "input_text"
  "counter"
  "timer"

  # Media & voce
  "alexa_media"
  "google_assistant"
  "google_translate"
  "cast"
  "webostv"
  "onkyo"

  # Sensori & monitor
  "rd200_ble"
  "dpc"
  "recorder"
  "logbook"
  "history"
  "statistics"

  # Sicurezza
  "alarm_control_panel"
  "manual"
)

echo "📥 Download pagine integrazione HA (${#INTEGRATIONS[@]} integrazioni)..."
echo "================================================================"

OK=0
FAIL=0

for integration in "${INTEGRATIONS[@]}"; do
  # Prova .markdown prima, poi .md
  URL="$BASE_URL/${integration}.markdown"
  OUTFILE="${integration}.md"

  HTTP_CODE=$(curl -s -o "$OUTFILE" -w "%{http_code}" "$URL")

  if [ "$HTTP_CODE" = "200" ] && [ -s "$OUTFILE" ]; then
    echo "  ✅ $integration"
    OK=$((OK+1))
  else
    rm -f "$OUTFILE"
    # Prova variante .md
    URL2="$BASE_URL/${integration}.md"
    HTTP_CODE2=$(curl -s -o "$OUTFILE" -w "%{http_code}" "$URL2")
    if [ "$HTTP_CODE2" = "200" ] && [ -s "$OUTFILE" ]; then
      echo "  ✅ $integration (md)"
      OK=$((OK+1))
    else
      rm -f "$OUTFILE"
      echo "  ⚠️  $integration (non trovata)"
      FAIL=$((FAIL+1))
    fi
  fi
done

# Pagine extra: automazioni avanzate, best practice
echo ""
echo "📥 Download guide avanzate..."

GUIDES=(
  "docs/automation/index.markdown|automation_guide.md"
  "docs/automation/trigger.markdown|automation_trigger.md"
  "docs/automation/condition.markdown|automation_condition.md"
  "docs/automation/action.markdown|automation_action.md"
  "docs/scripts.markdown|scripts_guide.md"
  "docs/template.markdown|template_guide.md"
  "docs/energy/index.markdown|energy_guide.md"
)

for guide in "${GUIDES[@]}"; do
  SRC="${guide%%|*}"
  DST="${guide##*|}"
  HTTP_CODE=$(curl -s -o "$DST" -w "%{http_code}" "$BASE_URL_DOCS/$SRC")
  if [ "$HTTP_CODE" = "200" ] && [ -s "$DST" ]; then
    echo "  ✅ $DST"
    OK=$((OK+1))
  else
    rm -f "$DST"
    echo "  ⚠️  $DST (non trovata)"
  fi
done

# Pulizia: rimuovi file troppo piccoli (stub/redirect)
echo ""
echo "🧹 Rimuovo file vuoti o stub (<500 bytes)..."
find . -name "*.md" -size -500c -delete

echo ""
echo "================================================================"
echo "✅ Scaricati: $OK  |  ⚠️  Non trovati: $FAIL"
echo "File pronti:"
find . -name "*.md" | wc -l
echo ""
echo "📊 Per indicizzare:"
echo "docker exec rag-backend python ingest.py \\"
echo "  --folder /app/data/ha_integrations \\"
echo "  --ext .md \\"
echo "  --chunk-size 1000 --chunk-overlap 150"
