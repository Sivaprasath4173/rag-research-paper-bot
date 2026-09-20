"""
app.py
------
ArXivInsight AI — Research Paper Intelligence System (Streamlit Application)
Unique Capstone UI featuring PDF Uploader, Dynamic Chunking, Dual Embeddings,
Hybrid/Cosine/MMR Retrieval Strategies, and Citation Cards.
Run with: streamlit run app.py
"""

import os
import shutil
import numpy as np
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Import LangChain / RAG components
# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="ArXivInsight AI — Research Paper Assistant", page_icon="🔬", layout="wide")

# Unique Premium Dark Cyberpunk Theme CSS
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .stApp {
        background-color: #0b0f19;
    }
    .user-msg {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 14px 18px;
        color: #f8fafc;
        margin-bottom: 16px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .answer-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid #6366f1;
        border-radius: 12px;
        padding: 20px 24px;
        color: #f1f5f9;
        font-size: 0.96rem;
        line-height: 1.7;
        margin-bottom: 20px;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
    }
    .source-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .source-title {
        color: #38bdf8;
        font-weight: 700;
        font-size: 0.88rem;
        margin-bottom: 4px;
    }
    .source-page {
        color: #a78bfa;
        font-size: 0.8rem;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .source-text {
        color: #cbd5e1;
        font-size: 0.82rem;
        font-family: 'Fira Code', 'Cascadia Code', monospace;
        background: #030712;
        padding: 10px;
        border-radius: 6px;
        border: 1px solid #111827;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("🔬 ArXivInsight AI — Seminal Research Paper System")
st.caption("Curated AI/ML research paper intelligence pipeline. Grounded, zero-hallucination answers powered by Hybrid Search & Google Gemini.")
st.markdown("**Author:** Sivaprasath | **GenAI Pin:** Pinnacle Plus Capstone")

# --- Sidebar Configuration ---
st.sidebar.title("⚡ Control Panel")

# 1. Embedding Method
st.sidebar.markdown("### 🧬 Embedding Model")
embedding_option = st.sidebar.radio(
    "Select Model:",
    options=[
        "BAAI/bge-m3 (1024-dim, Multi-lingual)",
        "mxbai-embed-large-v1 (1024-dim, English)"
    ],
    index=0
)
embedding_model_name = "BAAI/bge-m3" if "bge-m3" in embedding_option else "mixedbread-ai/mxbai-embed-large-v1"
collection_name = "bge_m3" if "bge-m3" in embedding_option else "mxbai"

# 2. Upload Documents
st.sidebar.markdown("### 📚 Custom Corpus Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload Research Papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload your own research PDFs or query the pre-loaded seminal papers (Attention Is All You Need & BERT)."
)

# 3. Chunking Strategy
st.sidebar.markdown("### ✂️ Chunk Granularity")
chunking_choice = st.sidebar.radio(
    "Splitting Config:",
    options=[
        "Config A — 500 chars (50 overlap)",
        "Config B — 1000 chars (150 overlap)"
    ],
    index=0
)
chunk_size = 500 if "500" in chunking_choice else 1000
chunk_overlap = 50 if "500" in chunking_choice else 150

# 4. Retrieval Strategy
st.sidebar.markdown("### 🔎 Retrieval Strategy")
retrieval_strategy = st.sidebar.radio(
    "Strategy:",
    options=[
        "Hybrid Fusion (BM25 + Vector)",
        "Dense Vector Search (Cosine)",
        "Max Marginal Relevance (MMR)"
    ],
    index=0
)

# 5. Top-K Sources
top_k = st.sidebar.slider("Top Context Passages (k)", min_value=1, max_value=5, value=3)

# Save uploaded files if provided
PDF_FOLDER = Path("./pdfs")
PDF_FOLDER.mkdir(parents=True, exist_ok=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        target_path = PDF_FOLDER / uploaded_file.name
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

# --- Load Corpus & Vectorstores ---
@st.cache_resource(show_spinner=False)
def initialize_pipeline(chunk_size_val: int, chunk_overlap_val: int, model_name: str, col_name: str):
    """Loads documents, chunks them dynamically, initializes BM25 index and ChromaDB vectorstore."""
    load_dotenv(override=False)
    CHROMA_DB_DIR = Path("./chroma_db")
    
    pdf_files = sorted([p for p in PDF_FOLDER.iterdir() if p.suffix.lower() == ".pdf"])
    all_docs = []
    for p in pdf_files:
        try:
            pages = PyPDFLoader(str(p)).load()
            for doc in pages:
                doc.metadata["filename"] = p.stem
                doc.metadata["page_display"] = doc.metadata["page"] + 1
            all_docs.extend(pages)
        except Exception as e:
            pass
            
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_val,
        chunk_overlap=chunk_overlap_val,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    active_chunks = splitter.split_documents(all_docs)
    
    # BM25 Sparse Index
    tokenized_corpus = [c.page_content.lower().split() for c in active_chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    vectorstore = Chroma(
        collection_name=col_name,
        persist_directory=str(CHROMA_DB_DIR / col_name),
        embedding_function=embeddings,
    )
    
    return active_chunks, bm25_index, vectorstore

with st.spinner("Initializing Pipeline, Vectorstores & BM25 Index..."):
    active_chunks, bm25_index, active_vectorstore = initialize_pipeline(
        chunk_size, chunk_overlap, embedding_model_name, collection_name
    )

def retrieve_documents(query: str, k: int = 3, strategy: str = "Hybrid Fusion (BM25 + Vector)"):
    """Retrieves passages using Cosine Similarity, MMR, or Hybrid (BM25 + Dense) fusion."""
    if strategy == "Dense Vector Search (Cosine)":
        return active_vectorstore.similarity_search(query, k=k)
    elif strategy == "Max Marginal Relevance (MMR)":
        return active_vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=20)
    else:
        # Hybrid Search (50% BM25 + 50% Vector)
        tokenized_query = query.lower().split()
        bm25_raw = np.array(bm25_index.get_scores(tokenized_query), dtype=float)
        bm25_max = bm25_raw.max()
        bm25_norm = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw
        
        candidate_k = min(len(active_chunks), max(k * 10, 20))
        dense_raw = active_vectorstore.similarity_search_with_relevance_scores(query, k=candidate_k)
        
        dense_map = {doc.page_content: max(float(s), 0.0) for doc, s in dense_raw}
        if dense_map:
            dmax = max(dense_map.values()) or 1.0
            dense_map = {key: val / dmax for key, val in dense_map.items()}
            
        scored = []
        for i, chunk in enumerate(active_chunks):
            d = dense_map.get(chunk.page_content, 0.0)
            b = float(bm25_norm[i])
            scored.append((0.5 * d + 0.5 * b, chunk))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]

def format_context(docs):
    blocks = []
    for i, doc in enumerate(docs, 1):
        fname = doc.metadata.get("filename", "unknown").replace("_", " ").title()
        page = doc.metadata.get("page_display", "?")
        blocks.append(f"[Source {i}: {fname}, p.{page}]\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)

# --- LLM Authentication & Chain Setup ---
load_dotenv(override=True)
api_key = os.environ.get("GOOGLE_API_KEY", "").strip()

if not api_key or "your_google_api_key" in api_key:
    user_key = st.sidebar.text_input("Enter Google API Key:", type="password")
    api_key = user_key.strip()

if not api_key or "your_google_api_key" in api_key:
    st.warning("⚠️ **API Key Required**: Please update `GOOGLE_API_KEY` in `.env` or enter it in the sidebar.")
    st.stop()

SYSTEM_PROMPT = """You are a precise research assistant. Your job is to answer \
questions strictly and only based on the research paper excerpts provided in the \
context below.

RULES:
1. Base your answer ONLY on the provided context. Do not use any prior knowledge.
2. If the context does not contain enough information to answer the question, \
respond with exactly this phrase: "I don't know based on the provided context."
3. Keep your answer concise and factual.
4. After your answer, always include a 'Sources:' section listing the top sources \
you used, in this exact format:

Sources:
- [Source N — Paper Title, p.X]: one-line description of what this source contributes

--- CONTEXT START ---
{context}
--- CONTEXT END ---"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}"),
])

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key, temperature=0)
output_parser = StrOutputParser()

def rag_chain_fn(question: str) -> dict:
    source_docs = retrieve_documents(question, k=top_k, strategy=retrieval_strategy)
    context = format_context(source_docs)
    messages = prompt_template.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    answer = output_parser.invoke(response)
    return {"question": question, "answer": answer, "source_docs": source_docs}

rag_chain = RunnableLambda(rag_chain_fn)

def invoke_app_with_retry(question: str, max_retries: int = 5) -> dict:
    import time
    retries = 0
    while retries < max_retries:
        try:
            return rag_chain.invoke(question)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str or "UNAVAILABLE" in err_str:
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(5 * retries)
            else:
                raise e

# --- Main Query Input & Interface ---
st.markdown("##### 💡 Quick Sample Questions:")
col1, col2, col3 = st.columns(3)

default_query = ""
if col1.button("📌 Transformer d_model Dimension"):
    default_query = "What is the dimensionality of the embeddings (d_model) in the base Transformer model?"
if col2.button("📌 BERT-Base Architecture Parameters"):
    default_query = "What is the number of layers (L) and hidden size (H) in BERT-Base?"
if col3.button("📌 Why Self-Attention vs Recurrence"):
    default_query = "Why is self-attention faster than recurrent layers?"

query = st.chat_input("Ask a question about your uploaded research papers...")
if not query and default_query:
    query = default_query

if query:
    # Display user message
    st.markdown(f'<div class="user-msg">❓ {query}</div>', unsafe_allow_html=True)
    
    with st.spinner("Searching corpus & generating grounded response..."):
        try:
            result = invoke_app_with_retry(query)
            
            # Display answer card
            st.markdown(f'<div class="answer-card">🧠 {result["answer"]}</div>', unsafe_allow_html=True)
            
            # Display source expander matching screenshot style
            with st.expander(f"📖 Top {len(result['source_docs'])} Supporting Sources Cited", expanded=True):
                for i, doc in enumerate(result["source_docs"], 1):
                    fname = doc.metadata.get("filename", "unknown").replace("_", " ").title() + ".pdf"
                    page = doc.metadata.get("page_display", "?")
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">[{i}] {fname}</div>
                        <div class="source-page">📍 Page {page}</div>
                        <div class="source-text">{doc.page_content.strip()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                st.warning("⏱️ **Google API Daily Quota Exceeded (429)**: The free-tier API daily quota has been reached. Please generate a new free API key at https://aistudio.google.com/app/apikey or update `.env`.")
            elif "503" in err_str or "UNAVAILABLE" in err_str:
                st.warning("📡 **Google API Temporary Overload (503)**: Google servers are experiencing high traffic. Please retry in a few seconds.")
            else:
                st.error(f"An error occurred: {err_str}")
