"""
===============================================================================
MODULE: Document Indexing Script (index_documents.py)
PURPOSE:
  Ingests three content sources and builds a persistent hybrid search index:
    1. 37 web-scraped policy JSON files from 'Knowledge Base/JSONs/'
    2. 12 visual infographic markdown tables from 'processed_images.json'
    3. 17 PDF Annual Reports from 'Knowledge Base/PDFs/'

VECTOR DATABASE: ChromaDB (Persistent on disk at 'kmrl_agent/chroma_db/')
  - Dense vectors: Ollama 'nomic-embed-text' (768-dim)
  - ChromaDB stores: document chunks + embeddings + metadata

SPARSE INDEX: BM25Okapi (saved to 'kmrl_agent/bm25_index.pkl')
  - Used alongside ChromaDB dense retrieval for hybrid RRF fusion

CHUNKING STRATEGY:
  - Chunk size: 400 words with 50-word overlap for contextual continuity
===============================================================================
"""

import os
import json
import glob
import pickle
import pymupdf
import ollama
from rank_bm25 import BM25Okapi
import chromadb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(os.path.dirname(BASE_DIR), "Knowledge Base")
JSONS_DIR = os.path.join(KB_DIR, "JSONs")
PDFS_DIR = os.path.join(KB_DIR, "PDFs")
IMAGES_JSON = os.path.join(BASE_DIR, "processed_images.json")

# ChromaDB will persist to this directory on disk
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# BM25 sparse index saved separately (ChromaDB doesn't handle keyword ranking natively)
BM25_INDEX_PATH = os.path.join(BASE_DIR, "bm25_index.pkl")

COLLECTION_NAME = "kmrl_knowledge_base"


def chunk_text(text, chunk_size=400, overlap=50):
    """Split text into overlapping word-level chunks for contextual retrieval."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks


def tokenize_text(text):
    """Tokenize text for BM25 sparse indexing."""
    return [w.lower() for w in text.split() if w.isalnum()]


def build_knowledge_base_index():
    print("Building Hybrid ChromaDB + BM25 Knowledge Base Index...")

    documents = []   # List of dicts: {content, source_type, source_name, title, heading, url}

    # ---------------------------------------------------------
    # STEP 1: Ingest Scraped Web Policy JSON files (37 files)
    # ---------------------------------------------------------
    print("Ingesting JSON web policy documents...")
    json_files = glob.glob(os.path.join(JSONS_DIR, "*.json"))
    for jf in json_files:
        if "Lost and Found.json" in jf:
            continue  # Skip - handled separately in lost_found.db
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            doc_title = data.get("title") or os.path.basename(jf)
            doc_url = data.get("url", "")
            sections = data.get("sections", [])

            for sec in sections:
                heading = sec.get("heading") or ""
                text = sec.get("text") or ""
                full_content = f"{doc_title}\n{heading}\n{text}".strip()
                if len(full_content) < 20:
                    continue

                for chunk in chunk_text(full_content):
                    documents.append({
                        "source_type": "JSON",
                        "source_name": os.path.basename(jf),
                        "title": doc_title,
                        "heading": heading,
                        "url": doc_url,
                        "content": chunk
                    })
        except Exception as e:
            print(f"Error reading JSON {jf}: {e}")

    # ---------------------------------------------------------
    # STEP 2: Ingest Visual Infographic Markdown Tables
    # ---------------------------------------------------------
    print("Ingesting Image Markdown Infographics...")
    if os.path.exists(IMAGES_JSON):
        try:
            with open(IMAGES_JSON, "r", encoding="utf-8") as f:
                images_data = json.load(f)
            for img in images_data:
                content = f"{img['title']}\nCategory: {img['category']}\n\n{img['content_markdown']}"
                documents.append({
                    "source_type": "IMAGE",
                    "source_name": img["filename"],
                    "title": img["title"],
                    "heading": img["category"],
                    "url": img["filepath"],
                    "content": content
                })
        except Exception as e:
            print(f"Error reading {IMAGES_JSON}: {e}")

    # ---------------------------------------------------------
    # STEP 3: Ingest PDF Annual Reports (17 PDFs, up to 150MB each)
    # ---------------------------------------------------------
    print("Ingesting PDF Documents & Annual Reports...")
    pdf_files = glob.glob(os.path.join(PDFS_DIR, "*.pdf"))
    for pf in pdf_files:
        pdf_name = os.path.basename(pf)
        try:
            doc = pymupdf.open(pf)
            for page_num, page in enumerate(doc):
                text = page.get_text().strip()
                if not text or len(text) < 30:
                    continue

                full_page_content = f"Document: {pdf_name} (Page {page_num + 1})\n{text}"
                for chunk in chunk_text(full_page_content, chunk_size=400, overlap=50):
                    documents.append({
                        "source_type": "PDF",
                        "source_name": pdf_name,
                        "title": pdf_name,
                        "heading": f"Page {page_num + 1}",
                        "url": pf,
                        "content": chunk
                    })
        except Exception as e:
            print(f"Error processing PDF {pf}: {e}")

    print(f"Total document chunks created: {len(documents)}")

    # ---------------------------------------------------------
    # STEP 4: Persist to ChromaDB with nomic-embed-text vectors
    # ---------------------------------------------------------
    print(f"Initializing ChromaDB persistent store at: {CHROMA_DB_DIR}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Delete and recreate collection for a clean rebuild
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity for dense retrieval
    )

    # Generate embeddings and add to ChromaDB in batches
    print("Generating embeddings with nomic-embed-text and inserting into ChromaDB...")
    bm25_corpus = []
    batch_size = 50

    for batch_start in range(0, len(documents), batch_size):
        batch = documents[batch_start:batch_start + batch_size]
        batch_ids = []
        batch_embeddings = []
        batch_documents = []
        batch_metadatas = []

        for local_idx, doc in enumerate(batch):
            global_idx = batch_start + local_idx
            try:
                response = ollama.embeddings(model="nomic-embed-text", prompt=doc["content"])
                vec = response["embedding"]
            except Exception as e:
                print(f"Embedding error at chunk {global_idx}: {e}")
                vec = [0.0] * 768

            batch_ids.append(f"chunk_{global_idx}")
            batch_embeddings.append(vec)
            batch_documents.append(doc["content"])
            batch_metadatas.append({
                "source_type": doc["source_type"],
                "source_name": doc["source_name"],
                "title": doc["title"],
                "heading": doc["heading"],
                "url": doc["url"]
            })
            bm25_corpus.append(tokenize_text(doc["content"]))

        # Insert batch into ChromaDB
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )

        print(f"Inserted chunks {batch_start + 1} to {min(batch_start + batch_size, len(documents))} into ChromaDB...")

    # ---------------------------------------------------------
    # STEP 5: Save BM25 sparse index separately as a pickle
    # ---------------------------------------------------------
    print("Building and saving BM25 sparse index...")
    bm25_index = BM25Okapi(bm25_corpus)
    bm25_data = {
        "bm25_index": bm25_index,
        "bm25_corpus": bm25_corpus,
        "doc_ids": [f"chunk_{i}" for i in range(len(documents))],
        "doc_count": len(documents)
    }
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"BM25 index saved to: {BM25_INDEX_PATH}")
    print(f"ChromaDB index saved to: {CHROMA_DB_DIR}")
    print(f"Knowledge Base Index build complete! Total chunks: {len(documents)}")


if __name__ == "__main__":
    build_knowledge_base_index()
