import logging
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from sbp_regulations_rag.answer import TOP_K, ask_model
from sbp_regulations_rag.db import PgVectorSearch, make_pool
from sbp_regulations_rag.search import QueryEmbedder

GEMINI_TIMEOUT_MS = 30_000
ATTEMPTS = 2
WAIT_SECONDS = 2

logger = logging.getLogger("uvicorn.error")
state = {}


@asynccontextmanager
async def lifespan(app):
    load_dotenv()
    client = genai.Client(http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS))
    embedder = QueryEmbedder(client, save_to_disk=False, attempts=ATTEMPTS, wait=WAIT_SECONDS)
    pool = make_pool()
    state["pool"] = pool
    state["client"] = client
    state["searcher"] = PgVectorSearch(pool, embedder)
    yield
    pool.close()


app = FastAPI(title="SBP regulations Q&A", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500)


class Citation(BaseModel):
    doc: str
    page_start: int
    page_end: int
    text: str


class AskResponse(BaseModel):
    found: bool
    answer: str
    citations: list[Citation]


@app.get("/health")
def health():
    with state["pool"].connection() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    start = time.perf_counter()
    try:
        results = state["searcher"].search(request.question, k=TOP_K)
        reply = ask_model(state["client"], request.question, results, ATTEMPTS, WAIT_SECONDS)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="The language model is unavailable, try again later")

    numbers = dict.fromkeys(n for n in reply.citations if 1 <= n <= len(results))
    citations = [
        Citation(
            doc=chunk.doc,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            text=chunk.text,
        )
        for chunk in (results.iloc[n - 1] for n in numbers)
    ]
    seconds = time.perf_counter() - start
    logger.info(f"ask: found={reply.found} citations={len(citations)} chars={len(request.question)} seconds={seconds:.2f}")
    return AskResponse(found=reply.found, answer=reply.answer, citations=citations)
