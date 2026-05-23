#!/bin/bash
# Script per scaricare documentazione specifica per la tua installazione HA
# Esegui da: ~/Cybersec/rag-chatbot-docker/rag-chatbot/
# Uso: bash scripts/download_ha_docs.sh

DOCS_DIR="./data/ha_docs_extra"
mkdir -p "$DOCS_DIR"
cd "$DOCS_DIR"

echo "📥 Download documentazione specifica per la tua HA..."
echo "=============================================="

# ── 1. SHELLY ──────────────────────────────────────────────────────
echo "📦 [1/12] Shelly integration docs..."
git clone --depth=1 --filter=blob:none --sparse \
  https://github.com/home-assistant/home-assistant.io shelly_docs 2>/dev/null || true
cd shelly_docs
git sparse-checkout set source/_integrations/shelly.markdown
cp source/_integrations/shelly.markdown ../shelly.md 2>/dev/null || true
cd ..
rm -rf shelly_docs

# ── 2. ESPHOME ─────────────────────────────────────────────────────
echo "📦 [2/12] ESPHome docs..."
git clone --depth=1 https://github.com/esphome/esphome-docs esphome_docs 2>/dev/null || \
  git -C esphome_docs pull
find esphome_docs -name "*.rst" -o -name "*.md" | \
  grep -E "components|automations|cookbook" | head -50 | \
  xargs -I{} cp {} ./ 2>/dev/null || true

# ── 3. NODE-RED HA ─────────────────────────────────────────────────
echo "📦 [3/12] Node-RED HA integration docs..."
git clone --depth=1 \
  https://github.com/zachowj/hass-node-red nodered_ha 2>/dev/null || \
  git -C nodered_ha pull
find nodered_ha -name "*.md" | xargs -I{} cp {} ./ 2>/dev/null || true

# ── 4. MEROSS LAN ──────────────────────────────────────────────────
echo "📦 [4/12] Meross LAN docs..."
git clone --depth=1 \
  https://github.com/krahabb/meross_lan meross_lan 2>/dev/null || \
  git -C meross_lan pull
find meross_lan -name "*.md" | xargs -I{} cp {} ./ 2>/dev/null || true

# ── 5. ALEXA MEDIA PLAYER ──────────────────────────────────────────
echo "📦 [5/12] Alexa Media Player docs..."
git clone --depth=1 \
  https://github.com/alandtse/alexa_media_player alexa_media 2>/dev/null || \
  git -C alexa_media pull
find alexa_media -name "*.md" | xargs -I{} cp {} ./ 2>/dev/null || true

# ── 6. ZCS AZZURRO (solare) ────────────────────────────────────────
echo "📦 [6/12] ZCS Azzurro (impianto solare) docs..."
git clone --depth=1 \
  https://github.com/alexdelprete/ha-abb-powerone-pvi-sunspec zcs_docs 2>/dev/null || true
find zcs_docs -name "*.md" | xargs -I{} cp {} ./ 2>/dev/null || true

# ── 7. FRITZBOX ────────────────────────────────────────────────────
echo "📦 [7/12] Fritz!Box integration docs..."
git clone --depth=1 \
  https://github.com/mib1185/homeassistant-addons fritz_docs 2>/dev/null || true

# ── 8. INFLUXDB + GRAFANA ──────────────────────────────────────────
echo "📦 [8/12] InfluxDB + Grafana con HA..."
curl -s "https://raw.githubusercontent.com/home-assistant/home-assistant.io/current/source/_integrations/influxdb.markdown" \
  -o influxdb_ha.md 2>/dev/null || true

# ── 9. FORECAST SOLAR ─────────────────────────────────────────────
echo "📦 [9/12] Forecast Solar docs..."
curl -s "https://raw.githubusercontent.com/home-assistant/home-assistant.io/current/source/_integrations/forecast_solar.markdown" \
  -o forecast_solar_ha.md 2>/dev/null || true

# ── 10. ADGUARD ───────────────────────────────────────────────────
echo "📦 [10/12] AdGuard docs..."
curl -s "https://raw.githubusercontent.com/home-assistant/home-assistant.io/current/source/_integrations/adguard.markdown" \
  -o adguard_ha.md 2>/dev/null || true

# ── 11. BLUEPRINTS HA ─────────────────────────────────────────────
echo "📦 [11/12] Blueprint ufficiali HA..."
git clone --depth=1 --filter=blob:none --sparse \
  https://github.com/home-assistant/home-assistant.io blueprint_docs 2>/dev/null || true
if [ -d blueprint_docs ]; then
  cd blueprint_docs
  git sparse-checkout set source/blueprints 2>/dev/null || true
  find source/blueprints -name "*.yaml" -exec cp {} ../ \; 2>/dev/null || true
  cd ..
  rm -rf blueprint_docs
fi

# ── 12. HACS ──────────────────────────────────────────────────────
echo "📦 [12/12] HACS docs..."
if [ ! -d hacs_docs ]; then
  git clone --depth=1 https://github.com/hacs/integration hacs_docs 2>/dev/null || true
else
  git -C hacs_docs pull 2>/dev/null || true
fi
find hacs_docs -name "*.md" -exec cp {} ./ \; 2>/dev/null || true

# ── Pulizia ───────────────────────────────────────────────────────
echo ""
echo "🧹 Pulizia file non necessari..."
find . -name "*.py" -delete
find . -name "*.js" -delete
find . -name "*.json" -delete
find . -name "*.png" -delete
find . -name "*.jpg" -delete
find . -name "LICENSE" -delete

echo ""
echo "✅ Download completato!"
echo "File scaricati:"
find . -maxdepth 1 -type f | wc -l
echo ""
echo "📊 Per indicizzare esegui:"
echo "docker exec rag-backend python ingest.py \\"
echo "  --folder /app/data/ha_docs_extra \\"
echo "  --ext .md .markdown .yaml .rst \\"
echo "  --chunk-size 1000 --chunk-overlap 200"
