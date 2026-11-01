import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/pages.jsonl")
REPEAT_SHARE = 0.5


def normalize(line):
    return re.sub(r"\d+", "#", line.strip())


def find_repeated_lines(pages):
    counts = Counter()
    for text in pages:
        counts.update({normalize(line) for line in text.splitlines() if line.strip()})
    min_pages = max(2, len(pages) * REPEAT_SHARE)
    return {line for line, n in counts.items() if n >= min_pages}

def is_contents_page(text):
    return len(re.findall(r"\.{5,}", text)) >= 5

def clean_page(text, repeated):
    if is_contents_page(text):
        return ""
    kept = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if normalize(line) in repeated:
            continue
        if re.fullmatch(r"\s*\d+\s*", line):
            continue
        kept.append(line.strip())
    return "\n".join(kept)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for path in sorted(RAW_DIR.glob("*.pdf")):
            pages = [page.get_text() for page in pymupdf.open(path)]
            repeated = find_repeated_lines(pages)
            for number, text in enumerate(pages, start=1):
                record = {"doc": path.stem, "page": number, "text": clean_page(text, repeated)}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{path.name:42} pages={len(pages):3}  removed as header/footer: {sorted(repeated)}")


if __name__ == "__main__":
    main()