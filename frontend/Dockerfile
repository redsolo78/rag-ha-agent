# ── Frontend Dockerfile ───────────────────────────────────────
# Nginx Alpine: immagine minimale (~25MB)
FROM nginx:alpine

LABEL description="RAG Chatbot Frontend"

# Rimuovi la config default di nginx
RUN rm /etc/nginx/conf.d/default.conf

# Copia la nostra config nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copia i file statici
COPY index.html /usr/share/nginx/html/index.html

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=3s \
  CMD wget -q --spider http://localhost/ || exit 1
