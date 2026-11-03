from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from google import genai
from pydantic import BaseModel, Field

from sbp_regulations_rag.answer import TOP_K, ask_model
from sbp_regulations_rag.db import PgVectorSearch, connect

state = {}


@asynccontextmanager
async def lifespan(app):
    conn = connect()
    state["conn"] = conn
    state["searcher"] = PgVectorSearch(conn)
    state["client"] = genai.Client()
    yield
    conn.close()


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
    state["conn"].execute("SELECT 1")
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    results = state["searcher"].search(request.question, k=TOP_K)
    try:
        reply = ask_model(state["client"], request.question, results)
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
    return AskResponse(found=reply.found, answer=reply.answer, citations=citations)