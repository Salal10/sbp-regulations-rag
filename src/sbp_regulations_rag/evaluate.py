from pathlib import Path

import pandas as pd

from sbp_regulations_rag.search import KeywordSearch, VectorSearch, load_chunks

QUESTIONS_PATH = Path("data/eval/questions.csv")
RESULTS_DIR = Path("results")
K = 10


def is_hit(chunk, question):
    return (
        chunk.doc == question.doc
        and chunk.page_start <= question.page_end
        and chunk.page_end >= question.page_start
    )


def first_hit_rank(results, question):
    for rank, chunk in enumerate(results.itertuples(), start=1):
        if is_hit(chunk, question):
            return rank
    return None


def evaluate(searcher, name):
    questions = pd.read_csv(QUESTIONS_PATH)
    questions = questions[questions["answerable"]]
    rows = []
    for q in questions.itertuples():
        results = searcher.search(q.question, k=K)
        rank = first_hit_rank(results, q)
        rows.append({"qid": q.qid, "source": q.source, "question": q.question, "rank": rank})
    df = pd.DataFrame(rows)
    rank = df["rank"]
    print(f"{name}: questions={len(df)}")
    for k in (1, 3, 5, 10):
        print(f"  recall@{k:<2} = {(rank <= k).mean():.1%}")
    print(f"  MRR@{K}    = {(1 / rank).fillna(0).mean():.3f}")
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / f"retrieval_{name}.csv", index=False)
    return df


if __name__ == "__main__":
    chunks = load_chunks()
    bm25 = evaluate(KeywordSearch(chunks), "bm25")
    vector_3072 = evaluate(VectorSearch(chunks, dims=3072), "vector_3072")
    vector_768 = evaluate(VectorSearch(chunks, dims=768), "vector_768")

    ranks = pd.DataFrame({
        "question": bm25["question"].str[:70],
        "bm25": bm25["rank"],
        "vec_3072": vector_3072["rank"],
        "vec_768": vector_768["rank"],
    })
    print("\nQuestions missed by at least one method (rank in top 10, blank = not found):")
    print(ranks[ranks[["bm25", "vec_3072", "vec_768"]].isna().any(axis=1)].to_string())