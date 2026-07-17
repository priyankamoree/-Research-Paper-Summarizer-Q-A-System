# Research Paper Summarizer + Q&A (RAG-powered)

An app that lets you upload a research paper PDF, get an AI-generated summary,
and ask follow-up questions that are answered using retrieval-augmented
generation (RAG) — so answers are grounded in the paper's actual text, not
just the model's general knowledge.

## 🌐 Live Demo

**Project Link:** research-paper-summarizer-q-a-system
.streamlit.app

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



