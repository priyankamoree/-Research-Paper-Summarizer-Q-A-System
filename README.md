# Research Paper Summarizer + Q&A (RAG-powered)

An app that lets you upload a research paper PDF, get an AI-generated summary,
and ask follow-up questions that are answered using retrieval-augmented
generation (RAG) — so answers are grounded in the paper's actual text, not
just the model's general knowledge.

## How it works

1. **PDF extraction** (`pdf_utils.py`) — pulls raw text out of the uploaded PDF, page by page.
2. **Chunking** — splits the text into overlapping ~1000-character chunks so relevant
   passages can be retrieved individually instead of stuffing the whole paper into every prompt.
3. **Embedding + retrieval** (`rag_engine.py`) — each chunk is embedded locally with
   `sentence-transformers` (`all-MiniLM-L6-v2`, runs on CPU, no API cost). A question is
   embedded the same way, and cosine similarity finds the most relevant chunks.
4. **Generation** — the top chunks are passed to **Gemini** along with the question (or, for
   summaries, a large slice of the paper) to produce the final answer/summary.
5. **Streamlit UI** (`app.py`) — handles upload, tabs for Summary / Q&A / Raw text, and a
   chat interface with source citations shown per answer.

## File structure

```
paper_qa_app/
├── app.py                          # Streamlit UI — run this
├── rag_engine.py                    # Embeddings, retrieval, Gemini calls
├── pdf_utils.py                      # PDF text extraction + chunking
├── requirements.txt
└── .streamlit/secrets.toml.example   # copy to secrets.toml with your real key
```

## Setup

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and paste in your real Gemini key
streamlit run app.py
```

Get a Gemini API key at https://aistudio.google.com/apikey — Google AI Studio has a free
tier (rate-limited), which is a solid default if you're covering usage cost for a public deployment.

The API key is read from `st.secrets`, never entered by the user in the UI (see
"Deploying this publicly" below for why).

The first run will download the small embedding model (~80MB), which is cached locally afterward.

## Using Gemini as the LLM backend

`rag_engine.py` calls Gemini directly:

```python
genai.configure(api_key=api_key)
self.model = genai.GenerativeModel(GENERATION_MODEL)
...
response = self.model.generate_content(prompt)
```

- **Model selection**: set by the `GENERATION_MODEL` constant at the top of `rag_engine.py`.
  Defaults to `"gemini-2.0-flash"` — fast and cheap. Swap to `"gemini-1.5-pro"` for stronger
  reasoning on dense papers, at higher cost/latency. Full list at
  https://ai.google.dev/gemini-api/docs/models
- **Embeddings stay local** regardless of LLM provider — sentence-transformers runs on your
  machine/server, so switching providers only affects generation cost, not retrieval.

## Deploying this publicly (recommended setup)

If you want strangers to be able to upload a paper and get a summary/Q&A — without seeing
each other's data, and without you handing your API key to every visitor — use this setup:

**1. Hold the key server-side, not in the UI.**
`app.py` reads `st.secrets["GEMINI_API_KEY"]` — visitors never see or enter it.
**Never commit `secrets.toml` to git** (a `.gitignore` is included that excludes it).

If deploying on **Streamlit Community Cloud**: don't upload the secrets file at all — paste its
contents into your app's dashboard under Settings → Secrets instead.

**2. Data isolation is automatic, not something you need to build.**
Streamlit gives each browser session its own `st.session_state` on the server. One visitor's
uploaded PDF, chunks, summary, and chat history are never visible to another visitor, as long as
nothing is written to a shared location — and this app never writes uploads to disk, so that
holds true by default.

**3. Usage caps protect your budget.**
Since you're now paying for every visitor's usage, `app.py` includes per-session limits
(`MAX_SUMMARIES_PER_SESSION`, `MAX_QUESTIONS_PER_SESSION` — 3 and 15 by default). Tune these to
your budget. For heavier abuse protection (e.g. one person refreshing to reset limits
repeatedly), consider IP-based rate limiting at the deployment layer, since `st.session_state`
alone resets on every new session.

**4. The embedding model is shared safely, per-user data is not.**
`rag_engine.py`'s `load_embedder()` is wrapped in `@st.cache_resource`, so the ~80MB
sentence-transformers model loads once per server process and is reused across all visitors —
it holds no user data, only a stateless text→vector function. Each visitor's actual paper
embeddings still live in their own session's `PaperQAEngine` instance, never shared.

**5. Be upfront about the privacy model.**
The sidebar includes a privacy note explaining data is session-only and cleared on refresh —
keep this visible since people are trusting the app with possibly-unpublished research.

## Implementation notes / things to customize

- **Chunk size**: 1000 chars with 150 overlap is a good default for academic prose. For
  papers with lots of equations/tables, consider larger chunks (1500–2000) since formulas
  get fragmented easily.
- **top_k retrieval**: exposed as a slider (2–10) in the sidebar. More chunks = more
  context but higher cost and slightly more noise in the answer.
- **Scanned PDFs**: this only extracts text layers. If a paper is a scanned image with no
  embedded text, extraction will fail — you'd need an OCR step (e.g. `pytesseract`) added
  to `pdf_utils.py` first.
- **Cost control**: the summary function truncates to ~60k characters to avoid runaway
  token costs on very long papers — for papers longer than that, consider a map-reduce
  summary (summarize chunks first, then summarize the summaries).

## Extending it further

- Multi-paper support: let users upload several papers and compare/synthesize across them.
- Citation-aware Q&A: parse the references section and let users ask "what does paper X say about Y."
- Export: add a "download Q&A session as PDF/Markdown" button for research notes.
- Persistence: if you want returning visitors to keep a paper's index without re-uploading,
  add a real vector store (e.g. Chroma) keyed by a document hash — with clear consent, since
  that changes the "nothing is saved" privacy story above.

## How to present this project

1. **Problem framing (30 sec)** — Reading papers is slow; researchers want fast, trustworthy
   answers rather than skimming 15 pages, and generic chatbots hallucinate about papers they
   haven't actually read.
2. **Live demo** — Upload a real paper live. Show the summary tab, then ask 2-3 questions and
   **expand the "Sources used" panel** — this proves answers are grounded, not hallucinated,
   which is the detail that differentiates it from "ChatGPT with a PDF pasted in."
3. **Architecture slide** — extract → chunk → embed → retrieve → generate. Have this ready;
   it's the most commonly asked technical question.
4. **Why RAG instead of pasting the whole paper into the prompt** — cost (no re-sending the
   whole paper per question), accuracy (retrieval surfaces the most relevant section rather
   than relying on the model to find it in a huge context), and it scales past the model's
   context window.
5. **Limitations, said upfront** — no OCR for scanned PDFs, no persistence across sessions,
   summary truncates very long papers. Naming these proactively reads as engineering maturity.
