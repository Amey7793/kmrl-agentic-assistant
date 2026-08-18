"""
===============================================================================
TOOL MODULE: Hybrid Dense + Sparse RAG Knowledge Base Search (kb_search_tool.py)
PURPOSE:
  Executes Hybrid RAG retrieval over 1,586+ document chunks from 17 PDF Annual Reports,
  37 policy JSON web pages, and visual infographics.

VECTOR DATABASE: ChromaDB (Persistent, HNSW cosine similarity index on disk)
  - Path: 'kmrl_agent/chroma_db/'
  - Dense query: Generates nomic-embed-text embedding then calls collection.query()

RETRIEVAL ARCHITECTURE:
  1. Dense Retrieval: ChromaDB HNSW query with nomic-embed-text 768-dim embeddings.
  2. Sparse Retrieval: BM25Okapi term frequency ranking (from 'bm25_index.pkl').
  3. RRF Fusion: Reciprocal Rank Fusion to merge dense + sparse results into final ranking.
===============================================================================
"""

import os
import pickle
import numpy as np
import ollama
import chromadb
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths for ChromaDB persistent store and BM25 sparse index
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
BM25_INDEX_PATH = os.path.join(BASE_DIR, "bm25_index.pkl")
COLLECTION_NAME = "kmrl_knowledge_base"


def tokenize_text(text):
    """Tokenize text for BM25 sparse scoring."""
    return [w.lower() for w in text.split() if w.isalnum()]


class KBSearchTool:
    def __init__(self):
        self.collection = None
        self.bm25_data = None
        self._load_index()

    def _load_index(self):
        """
        Load ChromaDB persistent collection and BM25 sparse index from disk.
        ChromaDB is loaded from 'chroma_db/' directory.
        BM25 index is loaded from 'bm25_index.pkl'.
        """
        # Load ChromaDB persistent collection
        chroma_path = CHROMA_DB_DIR
        if not os.path.exists(chroma_path):
            # Try alternate path relative to cwd
            chroma_path = os.path.join("kmrl_agent", "chroma_db")

        if os.path.exists(chroma_path):
            try:
                client = chromadb.PersistentClient(path=chroma_path)
                self.collection = client.get_collection(COLLECTION_NAME)
                print(f"ChromaDB collection loaded: '{COLLECTION_NAME}' with {self.collection.count()} chunks.")
            except Exception as e:
                print(f"Warning: Could not load ChromaDB collection: {e}")
        else:
            print(f"Warning: ChromaDB directory not found at {chroma_path}.")

        # Load BM25 sparse index
        bm25_path = BM25_INDEX_PATH
        if not os.path.exists(bm25_path):
            bm25_path = os.path.join("kmrl_agent", "bm25_index.pkl")

        if os.path.exists(bm25_path):
            try:
                with open(bm25_path, "rb") as f:
                    self.bm25_data = pickle.load(f)
                print(f"BM25 sparse index loaded with {self.bm25_data['doc_count']} document entries.")
            except Exception as e:
                print(f"Warning: Could not load BM25 index: {e}")
        else:
            print(f"Warning: BM25 index file not found at {bm25_path}.")

    def search(self, query: str, top_k: int = 5):
        """
        Hybrid RRF retrieval combining:
          - ChromaDB HNSW dense search (nomic-embed-text embeddings)
          - BM25 sparse keyword ranking
        Returns top_k fused results ranked by Reciprocal Rank Fusion score.
        """
        if not self.collection:
            self._load_index()
            if not self.collection:
                return {"error": "Knowledge Base index is not yet built. Run index_documents.py first."}

        # ---- 1. Dense Retrieval via ChromaDB ----
        try:
            response = ollama.embeddings(model="nomic-embed-text", prompt=query)
            q_vec = response["embedding"]

            # Query ChromaDB with the query vector, fetch top_k * 3 candidates
            dense_results = self.collection.query(
                query_embeddings=[q_vec],
                n_results=min(top_k * 3, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            dense_ids = dense_results["ids"][0]       # chunk IDs in ranked order
            dense_docs = dense_results["documents"][0]
            dense_metas = dense_results["metadatas"][0]
        except Exception as e:
            print(f"ChromaDB query error: {e}")
            dense_ids, dense_docs, dense_metas = [], [], []

        # ---- 2. Sparse BM25 Retrieval ----
        sparse_top_ids = []
        if self.bm25_data:
            tokens = tokenize_text(query)
            bm25_scores = self.bm25_data["bm25_index"].get_scores(tokens)
            top_sparse_indices = np.argsort(bm25_scores)[::-1][:top_k * 3]
            sparse_top_ids = [self.bm25_data["doc_ids"][i] for i in top_sparse_indices]

        # ---- 3. Reciprocal Rank Fusion (RRF) ----
        # rrf_score = sum over each ranking list of: 1 / (k + rank)
        k_const = 60
        rrf_scores = {}

        for rank, chunk_id in enumerate(dense_ids):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k_const + rank + 1))

        for rank, chunk_id in enumerate(sparse_top_ids):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k_const + rank + 1))

        # Sort by combined RRF score descending
        top_ids_sorted = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        # ---- 4. Assemble Final Results ----
        # Build lookup map from dense results
        id_to_doc = {cid: (doc, meta) for cid, doc, meta in zip(dense_ids, dense_docs, dense_metas)}

        results = []
        for chunk_id in top_ids_sorted:
            if chunk_id in id_to_doc:
                doc_content, meta = id_to_doc[chunk_id]
                results.append({
                    "chunk_id": chunk_id,
                    "source_type": meta.get("source_type", ""),
                    "source_name": meta.get("source_name", ""),
                    "title": meta.get("title", ""),
                    "heading": meta.get("heading", ""),
                    "url": meta.get("url", ""),
                    "content": doc_content,
                    "rrf_score": float(rrf_scores[chunk_id])
                })

        formatted_summary = "\n\n---\n\n".join([
            f"**Source**: [{r['source_type']}] {r['source_name']} ({r['heading']})\n"
            f"**Title**: {r['title']}\n"
            f"**Content**: {r['content']}"
            for r in results
        ])

        return {
            "query": query,
            "results_count": len(results),
            "results": results,
            "formatted": formatted_summary
        }


if __name__ == "__main__":
    tool = KBSearchTool()
    if tool.collection:
        res = tool.search("What was KMRL revenue and profit in annual report?")
        print(f"Results: {res['results_count']}")
        print(res["formatted"][:500])
