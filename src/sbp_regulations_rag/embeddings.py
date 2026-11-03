import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

MODEL = "gemini-embedding-001"
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/chunk_embeddings.npy")
BATCH_SIZE = 90
SECONDS_BETWEEN_BATCHES = 62


def embed_texts(client, texts, task_type, attempts=3, wait=60):
    for attempt in range(attempts):
        try:
            result = client.models.embed_content(
                model=MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return np.array([e.values for e in result.embeddings], dtype=np.float32)
        except errors.APIError as e:
            if e.code != 429 and e.code < 500:
                raise
            print(f"  error {e.code}, waiting {wait} s")
            time.sleep(wait)
    raise RuntimeError(f"embedding failed {attempts} times")


def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def main():
    load_dotenv()
    client = genai.Client()
    texts = pd.read_json(CHUNKS_PATH, lines=True)["text"].tolist()
    parts = []
    for start in range(0, len(texts), BATCH_SIZE):
        if start > 0:
            time.sleep(SECONDS_BETWEEN_BATCHES)
        batch = texts[start:start + BATCH_SIZE]
        parts.append(embed_texts(client, batch, "RETRIEVAL_DOCUMENT"))
        print(f"embedded {start + len(batch)}/{len(texts)}")
    vectors = np.vstack(parts)
    np.save(EMBEDDINGS_PATH, vectors)
    print(f"saved {vectors.shape} to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()