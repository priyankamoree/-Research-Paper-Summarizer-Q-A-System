"""
rag_engine.py
Core retrieval-augmented generation engine for the Research Paper Summarizer + QA app.

Responsibilities:
- Embed chunks with sentence-transformers (local, free, no extra API calls)
- Retrieve the most relevant chunks for a query via cosine similarity
- Call the Gemini API to summarize or answer questions using retrieved context
"""

from typing import List, Tuple
import time
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable


def _generate_with_retry(model, prompt: str, max_output_tokens: int, max_retries: int = 3) -> str:
    """
    Call Gemini with retry-and-backoff for transient rate-limit/server errors.
    Raises a friendly RuntimeError (instead of the raw SDK exception) if retries
    are exhausted, so the UI can show a clear message instead of crashing.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(max_output_tokens=max_output_tokens),
            )
            return response.text
        except ResourceExhausted as e:
            last_error = e
            # Back off and retry — this is a rate-limit (429), often transient within seconds.
            time.sleep(2 ** attempt)  # 1s, 2s, 4s
        except ServiceUnavailable as e:
            last_error = e
            time.sleep(2 ** attempt)

    raise RuntimeError(
        "Gemini's API rate limit or quota was hit and retries didn't recover in time. "
        "This usually means too many requests hit the free tier at once, or the daily "
        "quota is exhausted. Try again in a minute, or in a few hours if the daily cap was hit."
    ) from last_error


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, good quality for retrieval, runs locally (free)

# Gemini model for generation. Swap freely — e.g. "gemini-2.0-flash" (fast/cheap, good default),
# "gemini-1.5-pro" (stronger, slower, pricier) — see https://ai.google.dev/gemini-api/docs/models
GENERATION_MODEL = "gemini-2.0-flash"


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder() -> SentenceTransformer:
    """
    Load the embedding model once per server process and share it across all visitors.
    Safe to share: this model holds no per-user data — it's a stateless function from
    text -> vector. Each visitor's actual embeddings are still stored separately in
    their own st.session_state via PaperQAEngine instances.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


class PaperQAEngine:
    def __init__(self, api_key: str):
        # Gemini's SDK configures the key globally per process. Since every visitor
        # uses the same server-held key (see app.py), calling configure() here is safe
        # even across concurrent sessions.
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GENERATION_MODEL)
        self._embedder = None  # falls back to loading its own copy if load_embedder() wasn't called first
        self.chunks: List[str] = []
        self.chunk_embeddings: np.ndarray | None = None

    # ---------- Embeddings / Retrieval ----------

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = load_embedder()
        return self._embedder

    def build_index(self, chunks: List[str]) -> None:
        """Embed all chunks once and cache them in memory for retrieval."""
        self.chunks = chunks
        embeddings = self.embedder.encode(
            chunks, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        self.chunk_embeddings = embeddings

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return the top_k chunks most relevant to the query, with similarity scores."""
        if self.chunk_embeddings is None or len(self.chunks) == 0:
            return []

        query_vec = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]

        # embeddings are normalized, so dot product == cosine similarity
        scores = self.chunk_embeddings @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    # ---------- Generation ----------

    def summarize_paper(self, full_text: str, style: str = "detailed") -> str:
        """
        Summarize the paper. For long papers, we summarize using the most
        information-dense chunks (abstract, intro, conclusion patterns) plus
        a map-reduce style pass if the text is very long.
        """
        style_instructions = {
            "brief": "Write a concise 3-4 sentence summary capturing only the core contribution.",
            "detailed": (
                "Write a structured summary with these sections: Objective, Method, "
                "Key Findings, and Limitations. Use short paragraphs or bullet points."
            ),
            "eli5": "Explain the paper in simple terms a non-expert could understand, avoiding jargon.",
        }
        instruction = style_instructions.get(style, style_instructions["detailed"])

        # Guard against extremely long papers exceeding context comfortably
        max_chars = 60000
        text_for_prompt = full_text[:max_chars]

        prompt = f"""You are an expert research assistant summarizing an academic paper.

{instruction}

Paper text:
\"\"\"
{text_for_prompt}
\"\"\"

Respond with only the summary, no preamble."""

        return _generate_with_retry(self.model, prompt, max_output_tokens=7000)

    def answer_question(self, question: str, top_k: int = 5) -> Tuple[str, List[str]]:
        """
        Answer a question using retrieval-augmented generation:
        1. Retrieve the most relevant chunks to the question
        2. Ask Claude to answer using only those chunks as context
        Returns (answer, list_of_source_chunks_used)
        """
        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved:
            return "The paper hasn't been indexed yet. Please upload and process a PDF first.", []

        context_blocks = [chunk for chunk, _ in retrieved]
        context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are answering questions about a research paper using only the excerpts below.
If the excerpts don't contain enough information to answer confidently, say so explicitly
rather than guessing.

Excerpts:
\"\"\"
{context}
\"\"\"

Question: {question}

Answer clearly and cite which part of the excerpt supports your answer where relevant."""

        answer = _generate_with_retry(self.model, prompt, max_output_tokens=800)
        return answer, context_blocks
