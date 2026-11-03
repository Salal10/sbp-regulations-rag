import os

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from pgvector import HalfVector
from pgvector.psycopg import register_vector

from sbp_regulations_rag.embeddings import EMBEDDINGS_PATH, normalize
from sbp_regulations_rag.evaluate import evaluate
from sbp_regulations_rag.search import QueryEmbedder, load_chunks

CREATE_TABLE = """
DROP TABLE IF EXISTS chunks;
CREATE TABLE chunks (
    chunk_id   text PRIMARY KEY,
    doc        text NOT NULL,
    page_start int NOT NULL,
    page_end   int NOT NULL,
    text       text NOT NULL,
    embedding  halfvec(3072) NOT NULL
);
"""
CREATE_INDEX = "CREATE INDEX ON chunks USING hnsw (embedding halfvec_cosine_ops)"


def connect():
    load_dotenv()
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def load(conn):
    chunks = load_chunks()
    vectors = normalize(np.load(EMBEDDINGS_PATH))
    conn.execute(CREATE_TABLE)
    rows = [
        (c.chunk_id, c.doc, c.page_start, c.page_end, c.text, HalfVector(v))
        for c, v in zip(chunks.itertuples(), vectors)
    ]
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO chunks VALUES (%s, %s, %s, %s, %s, %s)", rows)
    conn.execute(CREATE_INDEX)
    conn.commit()
    print(f"loaded {len(rows)} chunks")


class PgVectorSearch:
    def __init__(self, conn):
        self.conn = conn
        self.embedder = QueryEmbedder()

    def search(self, query, k=10):
        vector = HalfVector(self.embedder.embed(query))
        rows = self.conn.execute(
            """
            SELECT chunk_id, doc, page_start, page_end, text, 1 - (embedding <=> %s) AS score
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (vector, vector, k),
        ).fetchall()
        return pd.DataFrame(rows, columns=["chunk_id", "doc", "page_start", "page_end", "text", "score"])


if __name__ == "__main__":
    with connect() as conn:
        load(conn)
        evaluate(PgVectorSearch(conn), "pgvector_hnsw")
