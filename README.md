# SBP regulations Q&A

Ask a question about State Bank of Pakistan rules on digital payments and customer protection, and get a short answer that cites the document and page it came from. If the documents don't cover the question, it says so instead of guessing.

```
POST /ask  {"question": "Can an e-money company connect to Raast?"}

{"found": true,
 "answer": "Yes, SBP may at its discretion provide Raast membership and connectivity to EMIs
            that have been granted go-live or commercial launch approval.",
 "citations": [{"doc": "emi_regulations_2023", "page_start": 23, "page_end": 23, "text": "..."}]}
```

## Documents

Six public PDFs from SBP's website, split into 448 chunks:

- Regulations for Electronic Money Institutions, 2023
- Licensing and regulatory framework for digital banks, 2022
- Branchless Banking Regulations, 2019
- Rules for Payment System Operators and Payment Service Providers, 2014
- Consumer Grievances Handling Mechanism, 2016
- Payment Systems and Electronic Fund Transfers Act, 2007

## How it works

1. Extract text from the PDFs (pymupdf), remove repeated headers and footers, skip contents pages.
2. Split into chunks of about 200 words at section headings, keeping the page numbers.
3. Embed the chunks with `gemini-embedding-001` and store them in Postgres with pgvector (`halfvec(3072)`, HNSW index).
4. For a question: find the 5 closest chunks, give them to `gemini-3.5-flash-lite` as numbered sources, and ask for JSON with `found`, `answer` and the source numbers it used. The code turns the numbers into document and page, so the model can't invent a page number.

## Evaluation

I built the test set before the search: 42 questions labelled with the document and page that answers them, plus 10 questions the documents can't answer (policy rate, zakat, ATM fees...). An LLM drafted 38 of the answerable questions from random chunks; 4 were written by hand. Every question was checked against the PDF, and 4 drafts were dropped and 2 edited.

**Finding the right chunk** (42 questions):

| Method | Right chunk in top 5 | MRR |
|---|---|---|
| Keyword search (BM25) | 71.4% | 0.620 |
| **Vector search, 3,072 dims** | **95.2%** | **0.846** |
| Vector search, 768 dims | 95.2% | 0.795 |
| Hybrid (BM25 + vector, RRF) | 88.1% | 0.771 |

Keyword search failed when the question used different words from the regulation ("Bitcoin" vs "virtual currencies", "EMI" vs "electronic money institution"). Hybrid search was worse than vector search alone here: keyword search was much weaker and pulled correct chunks down, so I kept vector search. Moving from numpy to Postgres changed nothing: all 42 questions ranked the same.

**Answers:**

| | Result |
|---|---|
| Answerable questions answered | 40 of 42 |
| Answers citing the right page | 39 of 40 |
| Answers correct (my grading) | 38 of 40, 2 partial |
| Unanswerable questions refused | **10 of 10** |

What went wrong:

- **q42** ("below what age is a customer a minor?"): the right chunk was 10th, so the model never saw it. It said "not found" rather than guessing, which is what I want when search misses.
- **q14**: the right chunk was 6th. The answer said "meet the minimum capital requirement" without the amount (PKR 4 billion).
- **q21**: the right chunk was there, but it says records are kept "for a period as may be determined by the State Bank", and the model refused because there was no number. I didn't tune the prompt for this one question.
- **q11**: gave the 15-day limit but left out the 7-working-day final reply.
- **q31**: the EMI regulations give two different limits for family members on the board (25% on p.16, 50% on p.42). The model reported both, with citations.

**LLM judge:** I also tried grading the answers with Gemini. It agreed with my grades on 92.5% of answers, but mostly on the easy ones: it accepted the vague q14 answer and marked q31 down although both rules are in the PDF. After tightening its grading rules it agreed on 95%. Those rules were written after seeing its mistakes on these same answers, so 95% is optimistic. It's useful for spotting a drop after a prompt change, not for final grading.

All results are CSVs in `results/`.

## Run it

You need Docker and a free Gemini API key from Google AI Studio. Create `.env`:

```
GEMINI_API_KEY=your-key
POSTGRES_PASSWORD=any-local-password
DATABASE_URL=postgresql://rag:any-local-password@localhost:6543/rag
```

Build the data (embedding takes about 5 minutes on the free tier):

```
uv sync
uv run python -m sbp_regulations_rag.download
uv run python -m sbp_regulations_rag.extract
uv run python -m sbp_regulations_rag.chunk
uv run python -m sbp_regulations_rag.embeddings
docker compose up -d db
uv run python -m sbp_regulations_rag.db
```

Start the API:

```
docker compose up -d --build
```

Then open http://127.0.0.1:8200/docs. An answer takes about 1.6 seconds.

## Limitations

- 42 test questions: one question is 2.4 points, so small differences are noise.
- No login or rate limit on the API, so anyone who can reach it can use your Gemini quota.
- The free Gemini tier may use prompts for training, so don't use it with private documents.
- Not legal advice. Regulations get amended, and these are the versions listed above.

