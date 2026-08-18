# KMRL Agentic RAG Assistant

A multi-tool, agentic Retrieval-Augmented Generation (RAG) chatbot for **Kochi Metro Rail Limited (KMRL)**, built entirely on open-source, locally-run models — no paid APIs, no external cloud LLM calls.

It answers passenger questions about train fares, routes, station timings, lost & found items, parking rates, feeder bus schedules, and official KMRL policies/financial reports, by routing each query to the correct specialized tool rather than relying on a single model's general knowledge.

## How it works

```
User Query
    │
    ▼
Intent Router  (ollama.chat: llama3.1:8b, JSON mode)
    │
    ├── "gtfs"        ──▶ GTFSTool        ──▶ gtfs.db (SQLite + NetworkX fallback)
    ├── "lost_found"  ──▶ LostFoundTool   ──▶ lost_found.db (SQLite FTS5, 31,626 records)
    ├── "vision"      ──▶ VisionTool      ──▶ processed_images.json (pre-OCR'd infographics)
    ├── "kb_search"   ──▶ KBSearchTool    ──▶ ChromaDB (dense) + BM25 (sparse), RRF-fused
    ├── "multi"       ──▶ LostFoundTool + KBSearchTool (fixed pairing)
    └── "chitchat"    ──▶ handled directly, no tool call (greetings/thanks/farewells)
    │
    ▼
Context Assembly  (tag + join each tool's results)
    │
    ▼
Response Synthesizer  (ollama.chat: llama3.1:8b)
    │
    ▼
Final Answer  ──▶ CLI (main.py) or Streamlit chat UI (kmrl_agent/ui/app.py)
```

Every query costs exactly **two** LLM calls — one to route, one to synthesize the final answer. Everything in between (fare lookups, route pathfinding, lost-item search, document retrieval) is resolved deterministically via SQL, FTS5, keyword scoring, or hybrid vector search — not the LLM's own knowledge. If Ollama's routing call itself fails, a keyword-based fallback heuristic picks the tool instead.

**Known limitations** (by design, not yet built):
- Routing is single-shot — there's no retry or re-routing if a tool comes back empty.
- The pipeline is stateless — no conversation memory carries between messages.
- The `multi` branch always pairs `lost_found` + `kb_search`, regardless of which policy the query actually needs.

## Tools

| Tool | Triggers on | Backend |
|---|---|---|
| `gtfs_tool.py` | Fares, routes, station-to-station stops, next train times | SQL joins on `fare_rules`/`fare_attributes`/`stop_times`; NetworkX shortest-path fallback for cross-line routes |
| `lost_found_tool.py` | Lost/found items, LFID lookups | SQLite FTS5 full-text search + exact LFID match |
| `vision_tool.py` | Parking rates, feeder bus timetables, system map | Keyword-overlap scoring over OCR-extracted markdown tables |
| `kb_search_tool.py` | Annual report financials, policies, tenders, vigilance, general queries | ChromaDB (dense, `nomic-embed-text` 768-dim) + BM25 (sparse), fused with Reciprocal Rank Fusion |

## Tech stack

- **LLM / embeddings**: [Ollama](https://ollama.com) running `llama3.1:8b` (routing + synthesis) and `nomic-embed-text` (dense retrieval)
- **Vector store**: ChromaDB (persistent, HNSW cosine index)
- **Sparse retrieval**: `rank-bm25`
- **Relational data**: SQLite (`gtfs.db`, `lost_found.db`)
- **Graph fallback**: NetworkX
- **Ingestion**: `pandas` (GTFS CSVs), `pymupdf` (PDF text extraction), `rapidocr-onnxruntime` (infographic OCR)
- **UI**: Streamlit

## Project structure

```
KMRL/
├── main.py                          # CLI entrypoint — runs sample test queries
├── requirements.txt
├── Dockerfile / docker-compose.yml  # containerized app + Ollama service
├── Knowledge Base/
│   ├── OpenData/                    # 11 GTFS transit files (stops, routes, fares, ...)
│   ├── JSONs/                       # 37 scraped KMRL policy/web pages
│   ├── PDFs/                        # 17 Annual Reports (not committed — see .gitignore)
│   └── Images/                      # Parking, feeder bus, system map infographics
└── kmrl_agent/
    ├── agent/
    │   ├── router.py                 # IntentRouter — classifies queries into tool calls
    │   └── assistant.py              # KMRLAssistant — orchestrates route → execute → synthesize
    ├── tools/                        # gtfs_tool.py, lost_found_tool.py, vision_tool.py, kb_search_tool.py
    ├── data_ingestion/                # One-off scripts that build the databases/indices below
    │   ├── build_gtfs_db.py          # OpenData/*.txt        → gtfs.db
    │   ├── build_lost_found_db.py    # Lost and Found.json   → lost_found.db (+FTS5)
    │   ├── process_images.py         # Images/*              → processed_images.json (via OCR)
    │   └── index_documents.py        # JSONs + PDFs + images → chroma_db/ + bm25_index.pkl
    ├── ui/app.py                     # Streamlit chat interface
    ├── gtfs.db, lost_found.db, chroma_db/, bm25_index.pkl, processed_images.json  # generated (gitignored)
    └── kb_index.pkl                  # unused legacy artifact, safe to delete
```

## Setup

### Option A — Local

```powershell
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

The databases/indices under `kmrl_agent/` (`gtfs.db`, `lost_found.db`, `chroma_db/`, `bm25_index.pkl`, `processed_images.json`) are gitignored build artifacts. If they're missing, rebuild them once, in order:

```powershell
python kmrl_agent/data_ingestion/build_gtfs_db.py
python kmrl_agent/data_ingestion/build_lost_found_db.py
python kmrl_agent/data_ingestion/process_images.py
python kmrl_agent/data_ingestion/index_documents.py
```

Then run either entrypoint:

```powershell
python main.py                              # CLI — runs 5 sample test queries
streamlit run kmrl_agent/ui/app.py           # Web chat UI at http://localhost:8501
```

### Option B — Docker

```powershell
docker compose up --build
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text
```

Then open `http://localhost:8501`. The `app` container mounts `kmrl_agent/` from the host, so your local databases/indices (and any code edits) are used directly — no need to re-run ingestion inside the container if you've already built them locally.

## Notes on hardware

Both LLM calls run on CPU unless Ollama detects a supported GPU (NVIDIA CUDA / AMD ROCm) — integrated graphics (e.g. Intel Iris Xe) are not accelerated by Ollama, so expect each query to take longer on such machines. For faster inference, consider a smaller model for the routing step (`llama3.2` or `gemma3:4b`) while keeping `llama3.1:8b` for the final synthesis, or renting a cheap cloud GPU (RunPod/Vast.ai, both offer usable GPUs from roughly $0.10–0.30/hr).
