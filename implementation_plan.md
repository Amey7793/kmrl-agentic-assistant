# Implementation Plan: Track 1 - Agentic RAG with 100% Open Source Models (No Paid Services)

This implementation plan details the setup and execution for **Track 1: Agentic RAG with Router & Specialized Tools** for **Kochi Metro Rail Limited (KMRL)**, using **100% open-source local models** (Ollama, local embeddings, SQLite engines, PyMuPDF, NetworkX) and **zero paid cloud services/APIs**.

---

## 1. System Design & Open Source Tech Stack

```
                                  +--------------------------+
                                  |     User Query Input     |
                                  +------------+-------------+
                                               |
                                               v
                                  +--------------------------+
                                  |   Intent Router Agent    |
                                  | (Ollama: llama3.1:8b)    |
                                  +------------+-------------+
                                               |
              +------------------+-------------+-------------+------------------+
              |                  |                           |                  |
              v                  v                           v                  v
      +---------------+  +---------------+           +---------------+  +---------------+
      |  GTFS Engine  |  | Lost & Found  |           | Hybrid RAG KB |  |  Vision RAG   |
      | (SQLite+Graph)|  | (SQLite DB)   |           | (BM25 + Dense)|  | (OCR/Layout)  |
      +-------+-------+  +-------+-------+           +-------+-------+  +-------+-------+
              |                  |                           |                  |
              +------------------+-------------+-------------+------------------+
                                               |
                                               v
                                  +--------------------------+
                                  |   Context Synthesizer    |
                                  |  (Ollama: llama3.1:8b)   |
                                  +------------+-------------+
                                               |
                                               v
                                  +--------------------------+
                                  | Streamed Answer & Source |
                                  |   (Streamlit Web UI)     |
                                  +--------------------------+
```

### Open Source Technology Stack (No Paid Services)
- **Local LLM Engine**: **Ollama** running `llama3.1:8b` (or `gemma3:4b`) for intent classification, tool parameters extraction, and context synthesis.
- **Local Embedding Engine**: **`nomic-embed-text`** via Ollama API (or local `sentence-transformers`) producing 768-dim dense vectors.
- **Deterministic GTFS Engine**: **SQLite 3** database (`gtfs.db`) created from `OpenData/*.txt` + **NetworkX** transit graph for station-to-station shortest path, fares, transfer points, and train arrival timings.
- **Lost & Found Engine**: **SQLite 3** database (`lost_found.db`) with full-text search (`FTS5`) indexed over 15,000+ items from `Lost and Found.json`.
- **Hybrid Vector + Keyword Search (PDFs & Web JSONs)**:
  - Document parsing via `PyMuPDF` / `PyMuPDF4LLM` to preserve markdown tables in Annual Reports.
  - Hybrid retrieval using **BM25** (`rank_bm25`) + **Dense Cosine Similarity** (`nomic-embed-text`) with Reciprocal Rank Fusion (RRF).
- **Vision RAG**: Layout OCR and table transcription over infographic images (`Parking-Rate.jpeg`, `SYSTEM-MAP`, `Timings-Feeder-Bus`).
- **Interactive UI**: **Streamlit** dashboard displaying chat history, tool routing logs, journey planner, and lost item finder.

---

## 2. Proposed System Architecture & Modules

### Directory Structure to Create (`kmrl_agent/`):
```
kmrl_agent/
├── data_ingestion/
│   ├── build_gtfs_db.py       # Imports 11 GTFS text files into SQLite & builds NetworkX graph
│   ├── build_lost_found_db.py # Ingests 3.7MB Lost and Found.json into indexed SQLite table
│   ├── process_images.py      # Transcribes images/infographics into Markdown tables & metadata
│   └── index_documents.py     # Parses 17 PDFs & 37 web JSONs into dense+sparse search index
├── tools/
│   ├── gtfs_tool.py           # SQL & NetworkX queries for fares, routes, arrival schedules
│   ├── lost_found_tool.py     # Structured SQL & FTS search for lost items by date/station/type
│   ├── kb_search_tool.py      # Hybrid BM25 + nomic-embed-text vector search with RRF reranking
│   └── vision_tool.py         # Visual inquiry over parking rates, feeder bus timings, system map
├── agent/
│   ├── router.py              # Zero-cost Ollama router classifying intent into JSON tool calls
│   └── assistant.py           # Core agent orchestrating tools, executing calls, & generating answers
├── ui/
│   └── app.py                 # Streamlit web interface with real-time tool execution transparency
└── main.py                    # CLI entrypoint for testing end-to-end queries
```

---

## 3. Tool Routing Strategy (Track 1)

| Tool Name | Trigger Conditions / Example Queries | Backend Implementation |
| :--- | :--- | :--- |
| **`gtfs_tool`** | *"How much is ticket from Aluva to Pettah?"*, *"What are the stations between Edapally and MG Road?"*, *"When is the next train?"* | Executes SQL joins on `fare_rules`, `fare_attributes`, `stops`, `stop_times` & NetworkX pathfinding. |
| **`lost_found_tool`** | *"Did anyone find an iPhone at Kalamassery?"*, *"Show items lost on Aug 9, 2026"* | SQL query on `lost_items` with date range, category, and station filtering. |
| **`vision_tool`** | *"What are the two-wheeler parking fees?"*, *"Show feeder bus schedule for Kalamassery"* | Queries structured markdown extracted from `Images/` (Parking-Rate, Feeder Bus info). |
| **`kb_search_tool`** | *"What was KMRL's revenue in 2022-23?"*, *"Tell me about Celebration on Wheels policy"*, *"Who is the vigilance officer?"* | Hybrid BM25 + Dense vector search over 17 PDFs (Annual Reports) and 37 policy JSONs. |
| **`multi_tool`** | *"I lost my watch on the train, what is the procedure and who can I contact?"* | Combines `lost_found_tool` (search database) + `kb_search_tool` (Grievance / Lost Property policy). |

---

## 4. Phase-by-Phase Execution Plan

### Phase 1: Ingestion & Tool Build (Offline Processing)
1. **GTFS Relational Engine**:
   - Parse `OpenData/` files (`stops.txt`, `routes.txt`, `fare_rules.txt`, `fare_attributes.txt`, `stop_times.txt`).
   - Create `gtfs.db` SQLite database with indexed tables and build a `NetworkX` graph for station hops.
2. **Lost & Found Relational Engine**:
   - Parse `JSONs/Lost and Found.json` (3.7MB, ~15,000 records).
   - Populate `lost_found.db` with indexed columns (`id`, `lfid`, `category`, `date`, `found_station`, `current_station`, `description`).
3. **PDF & Web Knowledge Ingestion**:
   - Extract text and markdown tables from 17 PDFs using `PyMuPDF`.
   - Chunk document text into parent-child chunks (500 tokens chunk size, 50 token overlap).
   - Generate embeddings using Ollama's `nomic-embed-text` and index sparse BM25 terms.
4. **Infographic & Image Ingestion**:
   - Process image assets (`Parking-Rate.jpeg`, `SYSTEM-MAP`, feeder bus timetables) into markdown tables.

### Phase 2: Open Source Agentic Router & Assistant
1. **`router.py`**:
   - Implement structured JSON output prompt using `ollama.chat(model='llama3.1:8b', format='json')`.
   - Classify query into standard tool schemas: `gtfs`, `lost_found`, `vision`, `kb_search`, `multi`.
2. **`assistant.py`**:
   - Tool execution engine that runs tool methods in parallel or sequence.
   - Synthesizes retrieved evidence into clear, authoritative answers with citations.

### Phase 3: Streamlit Interface & Testing
1. Build Streamlit Web App (`ui/app.py`) with:
   - Interactive Chatbot Interface.
   - Sidebar with Quick Tools (GTFS Journey Planner, Lost Item Lookup).
   - Transparent Debug Drawer showing which tools were invoked and retrieved context.

---

## 5. Verification Plan

### Automated Verification:
1. **GTFS Route & Fare Accuracy Test**:
   - Test fare calculations (e.g. Aluva -> Pettah = ₹60) against raw GTFS `fare_rules.txt`.
2. **Lost & Found Query Test**:
   - Verify specific LFID searches return exact record details within < 50ms.
3. **Hybrid RAG Precision Test**:
   - Query Annual Report financial figures (e.g. FY 2022-23 solar energy output) and verify accurate chunk retrieval.

### Manual Verification:
- Launch Streamlit interface locally and test multi-turn conversations, edge-case queries, and UI components.
