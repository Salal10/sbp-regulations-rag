import json
import re
from pathlib import Path

PAGES_PATH = Path("data/processed/pages.jsonl")
OUT_PATH = Path("data/processed/chunks.jsonl")
TARGET_WORDS = 200
MIN_WORDS = 80
OVERLAP_WORDS = 40
HEADING = re.compile(r"^(\d+(\.\d+)*\.?|[IVX]+\.|CHAPTER\s+[IVX]+)(\s|$)")


def load_pages():
    with PAGES_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def count_words(lines):
    return sum(len(text.split()) for text, _ in lines)


def overlap_tail(lines):
    tail = []
    for text, page in reversed(lines):
        if count_words(tail) + len(text.split()) > OVERLAP_WORDS:
            break
        tail.insert(0, (text, page))
    return tail


def make_chunk(doc, lines):
    return {
        "doc": doc,
        "page_start": lines[0][1],
        "page_end": lines[-1][1],
        "text": "\n".join(text for text, _ in lines),
    }


def chunk_document(doc, pages):
    lines = [(line, p["page"]) for p in pages for line in p["text"].splitlines()]
    chunks, current = [], []
    for text, page in lines:
        words = count_words(current)
        if HEADING.match(text) and words >= MIN_WORDS:
            chunks.append(make_chunk(doc, current))
            current = []
        elif words >= TARGET_WORDS:
            chunks.append(make_chunk(doc, current))
            current = overlap_tail(current)
        current.append((text, page))
    if current:
        chunks.append(make_chunk(doc, current))
    return chunks


def main():
    pages = load_pages()
    docs = sorted({p["doc"] for p in pages})
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, [p for p in pages if p["doc"] == doc])
        for i, chunk in enumerate(chunks):
            chunk["chunk_id"] = f"{doc}#{i:03d}"
        all_chunks.extend(chunks)
        print(f"{doc:40} chunks={len(chunks)}")
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for chunk in all_chunks:
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"total chunks: {len(all_chunks)}")


if __name__ == "__main__":
    main()