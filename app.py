"""
app.py
Streamlit front end for the AI-powered Research Paper Summarizer + Question Answering System.

Public-deployment version:
- API key is held server-side (st.secrets), never shown to or entered by visitors.
- Each visitor's upload, summary, and chat history live only in their own
  st.session_state, which Streamlit keeps isolated per browser session.
- Nothing is written to disk, so no visitor's data is ever persisted or shared.
- Per-session usage caps protect against runaway API cost from a single visitor.

Run with:
    streamlit run app.py
"""

import streamlit as st
from pdf_utils import extract_text_from_pdf, chunk_text, estimate_token_count
from rag_engine import PaperQAEngine, load_embedder

st.set_page_config(page_title="Research Paper AI", page_icon="📄", layout="wide")

# ---------- Usage caps (protects your API budget on a public deployment) ----------
MAX_SUMMARIES_PER_SESSION = 3
MAX_QUESTIONS_PER_SESSION = 15


# ---------- Session State Setup ----------

def init_session_state():
    defaults = {
        "engine": None,
        "paper_text": None,
        "paper_name": None,
        "chunks": [],
        "summary": None,
        "chat_history": [],  # list of (role, message) tuples
        "indexed": False,
        "summary_count": 0,
        "question_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# Shared embedding model, loaded once per server process (not per visitor).
# This is safe to share: the model itself holds no user data, only the
# resulting embeddings (which live in each visitor's own session_state) do.
load_embedder()

# Server-held API key — visitors never see or enter this.
API_KEY = st.secrets.get("GEMINI_API_KEY")

if not API_KEY:
    st.error(
        "Server is missing a GEMINI_API_KEY in st.secrets. "
        "This is a deployment configuration issue, not something a visitor can fix."
    )
    st.stop()


# ---------- Sidebar: settings only, no key entry ----------

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Summary style")
    summary_style = st.radio(
        "Choose how the summary should be written",
        options=["detailed", "brief", "eli5"],
        format_func=lambda x: {"detailed": "Detailed (structured)", "brief": "Brief (3-4 sentences)",
                                "eli5": "Explain like I'm 5"}[x],
    )

    st.divider()
    st.subheader("Retrieval settings")
    top_k = st.slider("Chunks retrieved per question", min_value=2, max_value=10, value=5)

    st.divider()
    st.caption(
        f"Summaries used: {st.session_state.summary_count}/{MAX_SUMMARIES_PER_SESSION}  \n"
        f"Questions asked: {st.session_state.question_count}/{MAX_QUESTIONS_PER_SESSION}"
    )

    st.divider()
    if st.button("🗑️ Clear my session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.divider()
    st.caption(
        "🔒 **Privacy**: your upload, summary, and questions exist only for this "
        "browser session. Nothing is saved to a server or visible to other visitors. "
        "Refreshing or closing the tab clears everything."
    )


# ---------- Header ----------

st.title("📄 Research Paper Summarizer + Q&A")
st.caption("Upload a research paper (PDF). Get a structured summary, then ask follow-up questions grounded in the actual text.")


# ---------- File Upload & Processing ----------

uploaded_file = st.file_uploader("Upload a PDF research paper", type=["pdf"])

if uploaded_file is not None and uploaded_file.name != st.session_state.paper_name:
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    if len(text.strip()) < 200:
        st.error("Couldn't extract meaningful text from this PDF. It may be a scanned image without OCR text.")
        st.stop()

    st.session_state.paper_text = text
    st.session_state.paper_name = uploaded_file.name
    st.session_state.chunks = chunk_text(text, chunk_size=1000, overlap=150)
    st.session_state.summary = None
    st.session_state.chat_history = []
    st.session_state.indexed = False
    st.session_state.summary_count = 0
    st.session_state.question_count = 0

    with st.spinner("Building search index over the paper (embedding chunks)..."):
        engine = PaperQAEngine(api_key=API_KEY)
        engine.build_index(st.session_state.chunks)
        st.session_state.engine = engine
        st.session_state.indexed = True

    st.success(f"Processed '{uploaded_file.name}' — {len(st.session_state.chunks)} chunks indexed "
               f"(~{estimate_token_count(text):,} tokens).")

if st.session_state.paper_text and st.session_state.engine is None:
    engine = PaperQAEngine(api_key=API_KEY)
    engine.build_index(st.session_state.chunks)
    st.session_state.engine = engine
    st.session_state.indexed = True

if not st.session_state.paper_text:
    st.stop()


# ---------- Tabs: Summary | Q&A | Raw Text ----------

tab_summary, tab_qa, tab_raw = st.tabs(["📝 Summary", "💬 Ask Questions", "📄 Extracted Text"])

with tab_summary:
    col1, col2 = st.columns([1, 4])
    with col1:
        summary_limit_reached = st.session_state.summary_count >= MAX_SUMMARIES_PER_SESSION
        generate_clicked = st.button(
            "Generate Summary", type="primary", use_container_width=True,
            disabled=summary_limit_reached,
        )
    if summary_limit_reached:
        st.caption(f"You've reached the {MAX_SUMMARIES_PER_SESSION}-summary limit for this session.")

    if generate_clicked and not summary_limit_reached:
        with st.spinner("Summarizing paper..."):
            summary = st.session_state.engine.summarize_paper(
                st.session_state.paper_text, style=summary_style
            )
            st.session_state.summary = summary
            st.session_state.summary_count += 1

    if st.session_state.summary:
        st.markdown(st.session_state.summary)
        st.download_button(
            "Download summary as .txt",
            data=st.session_state.summary,
            file_name=f"{st.session_state.paper_name.rsplit('.', 1)[0]}_summary.txt",
        )
    else:
        st.caption("Click 'Generate Summary' to summarize the uploaded paper.")

with tab_qa:
    st.caption("Ask anything about the paper. Answers are grounded in retrieved excerpts, not general knowledge.")

    question_limit_reached = st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION
    if question_limit_reached:
        st.warning(f"You've reached the {MAX_QUESTIONS_PER_SESSION}-question limit for this session.")

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    question = st.chat_input(
        "Ask a question about this paper...",
        disabled=question_limit_reached,
    )

    if question and not question_limit_reached:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant sections and generating answer..."):
                answer, sources = st.session_state.engine.answer_question(question, top_k=top_k)
                st.markdown(answer)
                with st.expander(f"📎 Sources used ({len(sources)} excerpts)"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Excerpt {i}:**")
                        st.text(src[:500] + ("..." if len(src) > 500 else ""))

        st.session_state.chat_history.append(("assistant", answer))
        st.session_state.question_count += 1

with tab_raw:
    st.text_area("Full extracted text", st.session_state.paper_text, height=500)
