import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from rank_bm25 import BM25Okapi

from sbp_regulations_rag.embeddings import EMBEDDINGS_PATH, embed_texts, normalize

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
QUERY_CACHE_PATH = Path("data/processed/query_embeddings.json")


def load_chunks():
    return pd.read_json(CHUNKS_PATH, lines=True)


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class KeywordSearch:
    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(text) for text in chunks["text"]])

    def search(self, query, k=10):
        scores = self.bm25.get_scores(tokenize(query))
        top = scores.argsort()[::-1][:k]
        return self.chunks.iloc[top].assign(score=scores[top])


class QueryEmbedder:
    def __init__(self):
        load_dotenv()
        self.client = genai.Client()
        self.cache = json.loads(QUERY_CACHE_PATH.read_text()) if QUERY_CACHE_PATH.exists() else {}

    def embed(self, query, dims=3072):
        if query not in self.cache:
            vector = embed_texts(self.client, [query], "RETRIEVAL_QUERY")[0]
            self.cache[query] = vector.tolist()
            QUERY_CACHE_PATH.write_text(json.dumps(self.cache))
        vector = np.array([self.cache[query]], dtype=np.float32)
        return normalize(vector[:, :dims])[0]


class VectorSearch:
    def __init__(self, chunks, dims=3072):
        self.embedder = QueryEmbedder()
        self.chunks = chunks
        self.dims = dims
        vectors = np.load(EMBEDDINGS_PATH)
        if len(vectors) != len(chunks):
            raise ValueError("chunk_embeddings.npy doesn't match chunks.jsonl - run embeddings again")
        self.vectors = normalize(vectors[:, :dims])

    def search(self, query, k=10):
        scores = self.vectors @ self.embedder.embed(query, self.dims)
        top = scores.argsort()[::-1][:k]
        return self.chunks.iloc[top].assign(score=scores[top])


class HybridSearch:
    def __init__(self, searchers, rrf_k=60, candidates=50):
        self.searchers = searchers
        self.rrf_k = rrf_k
        self.candidates = candidates
        self.chunks = searchers[0].chunks.set_index("chunk_id", drop=False)

    def search(self, query, k=10):
        scores = {}
        for searcher in self.searchers:
            results = searcher.search(query, k=self.candidates)
            for rank, chunk_id in enumerate(results["chunk_id"], start=1):
                scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (self.rrf_k + rank)
        top = sorted(scores, key=scores.get, reverse=True)[:k]
        return self.chunks.loc[top].assign(score=[scores[c] for c in top])