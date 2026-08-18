#!/bin/sh
set -e

cd /app

OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:11434}"

wait_for_ollama() {
  echo "[entrypoint] Waiting for Ollama at $OLLAMA_URL ..."
  i=0
  until curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -ge 30 ]; then
      echo "[entrypoint] WARNING: Ollama not reachable after 60s - continuing without it."
      return 1
    fi
    sleep 2
  done
  echo "[entrypoint] Ollama is up."
  return 0
}

has_model() {
  curl -sf "$OLLAMA_URL/api/tags" 2>/dev/null | grep -q "\"name\":\"$1"
}

# --- GTFS engine ---
if [ ! -f kmrl_agent/gtfs.db ]; then
  echo "[entrypoint] gtfs.db missing - building from Knowledge Base/OpenData ..."
  python kmrl_agent/data_ingestion/build_gtfs_db.py
else
  echo "[entrypoint] gtfs.db found - skipping."
fi

# --- Lost & Found engine ---
if [ ! -f kmrl_agent/lost_found.db ]; then
  echo "[entrypoint] lost_found.db missing - building from Lost and Found.json ..."
  python kmrl_agent/data_ingestion/build_lost_found_db.py
else
  echo "[entrypoint] lost_found.db found - skipping."
fi

# --- Vision / OCR engine ---
if [ ! -f kmrl_agent/processed_images.json ]; then
  echo "[entrypoint] processed_images.json missing - running OCR ingestion ..."
  python kmrl_agent/data_ingestion/process_images.py
else
  echo "[entrypoint] processed_images.json found - skipping."
fi

# --- Hybrid KB index (needs Ollama + nomic-embed-text) ---
need_kb_index=0
[ -f kmrl_agent/bm25_index.pkl ] || need_kb_index=1
if [ ! -d kmrl_agent/chroma_db ] || [ -z "$(ls -A kmrl_agent/chroma_db 2>/dev/null)" ]; then
  need_kb_index=1
fi

if [ "$need_kb_index" -eq 1 ]; then
  if wait_for_ollama && has_model "nomic-embed-text"; then
    echo "[entrypoint] KB index missing - embedding every chunk, this can take a while ..."
    # index_documents.py degrades gracefully if Knowledge Base/PDFs isn't mounted -
    # it just skips PDFs and indexes the JSONs/images that are present.
    python kmrl_agent/data_ingestion/index_documents.py
  else
    echo "[entrypoint] WARNING: KB index missing but Ollama / nomic-embed-text isn't ready."
    echo "[entrypoint] Skipping KB ingestion - kb_search will report an unbuilt index until you"
    echo "[entrypoint] pull nomic-embed-text and re-run: python kmrl_agent/data_ingestion/index_documents.py"
  fi
else
  echo "[entrypoint] chroma_db/ and bm25_index.pkl found - skipping KB ingestion."
fi

echo "[entrypoint] Startup checks complete. Launching app ..."
exec "$@"
