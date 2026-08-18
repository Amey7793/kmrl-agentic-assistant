# KMRL Agentic AI Assistant — Codebase Workflow

## System Architecture Overview

The project is a **multi-tool Agentic RAG (Retrieval-Augmented Generation)** assistant for Kochi Metro Rail Limited. It has two distinct phases: **Data Ingestion (offline)** and **Query Processing (runtime)**.

---

## Phase 1 — Offline: Data Ingestion & Index Building

```mermaid
flowchart TD
    A["📁 Knowledge Base/\nJSONs/ · PDFs/ · Images"] --> B

    subgraph data_ingestion["data_ingestion/"]
        B["build_gtfs_db.py\n─────────────────\nParses GTFS transit CSVs\n(routes, stops, trips,\n fares, stop_times)\nBuilds → gtfs.db (SQLite)"]
        C["build_lost_found_db.py\n─────────────────────\nParses Lost & Found\nJSON records (31,626)\nBuilds → lost_found.db\nwith FTS5 virtual table"]
        D["process_images.py\n────────────────\nRuns LLaVA vision LLM\non infographic images\n(parking, bus timetables,\n system map, fare chart)\nOutputs → processed_images.json"]
        E["index_documents.py\n───────────────────\nIngests 37 JSON policies\n+ 12 image markdowns\n+ 17 PDF Annual Reports\n\nChunks text (400 words,\n50-word overlap)\n\nGenerates nomic-embed-text\nvectors via Ollama\n\nPersists to:\n• chroma_db/ (ChromaDB)\n• bm25_index.pkl (BM25)"]
    end

    B --> F["💾 gtfs.db"]
    C --> G["💾 lost_found.db"]
    D --> H["📄 processed_images.json"]
    E --> I["💾 chroma_db/\n(ChromaDB HNSW)"]
    E --> J["📦 bm25_index.pkl\n(BM25 Sparse Index)"]
```

---

## Phase 2 — Runtime: Query Processing Pipeline

```mermaid
flowchart TD
    U(["👤 User Query"]) --> ENTRY

    subgraph ENTRY["Entry Points"]
        M["main.py\n(CLI)"]
        APP["ui/app.py\n(Streamlit UI)"]
    end

    ENTRY --> ASS

    subgraph agent["agent/"]
        ASS["assistant.py — KMRLAssistant\n─────────────────────────────\nOrchestrates the full pipeline:\n1. Route → 2. Tool Exec → 3. Synthesize"]
        ROT["router.py — IntentRouter\n──────────────────────────\nSends query to llama3.1:8b\nvia Ollama to classify intent\nOutputs: selected_tool + parameters\n\nFallback: keyword heuristics if LLM fails"]
    end

    ASS -->|"Step 1: Intent Classification"| ROT

    ROT -->|"gtfs"| GT
    ROT -->|"lost_found"| LF
    ROT -->|"vision"| VT
    ROT -->|"kb_search"| KB
    ROT -->|"multi"| BOTH

    subgraph tools["tools/"]
        GT["gtfs_tool.py — GTFSTool\n──────────────────────────\n• get_fare(origin, dest)\n• find_route(origin, dest)\n  (SQL + NetworkX fallback)\n• get_next_trains(station)\n• get_all_stations()\n\nData Source: gtfs.db (SQLite)"]

        LF["lost_found_tool.py — LostFoundTool\n──────────────────────────────────────\n• search_lost_items(query, cat, stn)\n  (FTS5 full-text search)\n• get_item_by_lfid(lfid)\n  (Exact LFID lookup)\n\nData Source: lost_found.db (SQLite)"]

        VT["vision_tool.py — VisionTool\n─────────────────────────────\n• query_visual_data(query)\n  (Keyword scoring over\n   pre-processed markdown)\n\nData Source: processed_images.json"]

        KB["kb_search_tool.py — KBSearchTool\n────────────────────────────────────\n• search(query, top_k)\n  1. Dense: ChromaDB HNSW query\n     (nomic-embed-text 768-dim)\n  2. Sparse: BM25Okapi ranking\n  3. Fuse: Reciprocal Rank Fusion\n\nData Sources:\n  chroma_db/ + bm25_index.pkl"]

        BOTH["MULTI → LostFoundTool\n              + KBSearchTool"]
    end

    GT --> CTX
    LF --> CTX
    VT --> CTX
    KB --> CTX
    BOTH --> CTX

    CTX["📦 Assembled Context Blocks\n[GTFS Info] / [Lost & Found DB]\n[Infographic Vision Data]\n[KMRL Knowledge Base]"]

    CTX -->|"Step 3: Synthesis"| LLM

    LLM["🤖 Ollama — llama3.1:8b\n(Response Synthesizer)\nSystem Prompt: KMRL official assistant rules\nGenerates structured, cited final answer"]

    LLM --> RES(["✅ Final Answer\nReturned to UI / CLI"])
```

---

## File-by-File Responsibility Map

| File | Layer | Responsibility |
|---|---|---|
| [main.py](file:///c:/Users/pooja.ingale/Projects/Kochi%20Metro/KMRL/main.py) | Entry | CLI runner; runs 5 test queries end-to-end |
| [ui/app.py](file:///c:/Users/pooja.ingale/Projects/Kochi%20Metro/KMRL/kmrl_agent/ui/app.py) | Entry | Streamlit web UI with sidebar widgets & chat interface |
| [agent/assistant.py](file:///c:/Users/pooja.ingale/Projects/Kochi%20Metro/KMRL/kmrl_agent/agent/assistant.py) | Agent Core | Orchestrates Route → Tool Exec → LLM Synthesis |
| [agent/router.py](file:///c:/Users/pooja.ingale/Projects/Kochi%20Metro/KMRL/kmrl_agent/agent/router.py) | Agent Core | LLM-based intent classifier → tool selector with fallback heuristics |
| [tools/gtfs_tool.py](file:///c:/Users/pooja.ingale/Projects/Kochi%20Metro/KMRL/kmrl_agent/tools/gtfs_tool.py) | Tool | Deterministic SQL + NetworkX graph queries on `gtfs.db` |
| [tools/lost_found_tool.py](file:///c:/Users/pooja.ingale/Projects/Kochi%20Metro/KMRL/kmrl_agent/tools/lost_found_tool.py) | Tool | FTS5 + SQL queries on `lost_found.db` (31,626 records) |
| [tools/vision_tool.py](file:///c:/Users/pooja.ingale/Projects/Kochi%20Metro/KMRL/kmrl_agent/tools/vision_tool.py) | Tool | Keyword-scored search over `processed_images.json` markdown |
| [tools/kb_search_tool.py](file:///c:/Users/pooja.ingale/Projects/Kochi%20Metro/KMRL/kmrl_agent/tools/kb_search_tool.py) | Tool | Hybrid Dense (ChromaDB) + Sparse (BM25) + RRF Fusion retrieval |
| [data_ingestion/build_gtfs_db.py](file:///c:/Users/pooja.ingale/Projects/Kochi%20Metro/KMRL/kmrl_agent/data_ingestion/build_gtfs_db.py) | Ingestion | Parses GTFS CSV files → SQLite `gtfs.db` |
| [data_ingestion/build_lost_found_db.py](file:///c:/Users/pooja.ingale/Projects/Kochi%20Metro/KMRL/kmrl_agent/data_ingestion/build_lost_found_db.py) | Ingestion | Parses Lost & Found JSON → SQLite `lost_found.db` with FTS5 |
| [data_ingestion/process_images.py](file:///c:/Users/pooja.ingale/Projects/Kochi%20Metro/KMRL/kmrl_agent/data_ingestion/process_images.py) | Ingestion | Uses LLaVA vision LLM to extract text from infographic images |
| [data_ingestion/index_documents.py](file:///c:/Users/pooja.ingale/Projects/Kochi%20Metro/KMRL/kmrl_agent/data_ingestion/index_documents.py) | Ingestion | Chunks + embeds JSON, PDF, Image docs → ChromaDB + BM25 index |

---

## Intent Routing Decision Table

| User Query Type | Selected Tool | Data Source |
|---|---|---|
| Fare / Route / Train timings | `gtfs` → `gtfs_tool.py` | `gtfs.db` |
| Lost or found items / LFID lookup | `lost_found` → `lost_found_tool.py` | `lost_found.db` |
| Parking charges / Feeder bus / System map | `vision` → `vision_tool.py` | `processed_images.json` |
| Annual reports / Policies / Financial data | `kb_search` → `kb_search_tool.py` | `chroma_db/` + `bm25_index.pkl` |
| Lost item + policy grievance combined | `multi` → both `lost_found` + `kb_search` | Both DBs |

