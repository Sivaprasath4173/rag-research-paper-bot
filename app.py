"""
app.py
------
Streamlit UI for the RAG Capstone Project.
Provides a simple text input for querying the RAG pipeline built in Sections 4-5.
Run with: streamlit run app.py
"""

import os
import numpy as np
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Import LangChain / RAG components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="RAG Capstone Q&A", page_icon="📚", layout="wide")

st.title("📚 Research Paper Answer Bot")
st.markdown("**Author:** Sivaprasath | **GenAI Pin:** Pinnacle Plus Capstone")
st.markdown("Retrieval-Augmented Generation (RAG) Capstone Project")

st.sidebar.title("Configuration")
st.sidebar.info("Using LLM: Google Gemini")

@st.cache_resource(show_spinner=False)
def load_pipeline_base():
    """Loads the documents, vectorstore, and hybrid search (independent of LLM)."""
    load_dotenv(override=False)
    PDF_FOLDER = Path("./pdfs")
    CHROMA_DB_DIR = Path("./chroma_db")
    
    pdf_files = sorted([p for p in PDF_FOLDER.iterdir() if p.suffix.lower() == ".pdf"])
    all_docs = []
    for p in pdf_files:
        pages = PyPDFLoader(str(p)).load()
        for doc in pages:
            doc.metadata["filename"] = p.stem
            doc.metadata["page_display"] = doc.metadata["page"] + 1
        all_docs.extend(pages)
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    active_chunks = splitter.split_documents(all_docs)
    
    tokenized_corpus = [c.page_content.lower().split() for c in active_chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    bge_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name="bge_m3",
        persist_directory=str(CHROMA_DB_DIR / "bge_m3"),
        embedding_function=bge_embeddings,
    )
    
    def hybrid_search(query, k=3, alpha=0.5):
        tokenized_query = query.lower().split()
        bm25_raw = np.array(bm25_index.get_scores(tokenized_query), dtype=float)
        bm25_max = bm25_raw.max()
        bm25_norm = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw
        
        candidate_k = min(len(active_chunks), max(k * 10, 20))
        dense_raw = vectorstore.similarity_search_with_relevance_scores(query, k=candidate_k)
        
        dense_map = {doc.page_content: max(float(s), 0.0) for doc, s in dense_raw}
        if dense_map:
            dmax = max(dense_map.values()) or 1.0
            dense_map = {k: v / dmax for k, v in dense_map.items()}
            
        scored = []
        for i, chunk in enumerate(active_chunks):
            d = dense_map.get(chunk.page_content, 0.0)
            b = float(bm25_norm[i])
            scored.append((alpha * d + (1 - alpha) * b, chunk))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]
        
    def format_context(docs):
        blocks = []
        for i, doc in enumerate(docs, 1):
            fname = doc.metadata.get("filename", "unknown").replace("_", " ").title()
            page = doc.metadata.get("page_display", "?")
            blocks.append(f"[Source {i}: {fname}, p.{page}]\n{doc.page_content.strip()}")
        return "\n\n".join(blocks)
        
    return hybrid_search, format_context

with st.spinner("Initializing Retrieval Pipeline..."):
    retrieve, format_context = load_pipeline_base()

# --- Build LLM Chain Dynamically ---
load_dotenv(override=False)

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

load_dotenv(override=True)
api_key = os.environ.get("GOOGLE_API_KEY", "").strip()

if not api_key or "your_google_api_key" in api_key:
    # Optional sidebar fallback only if .env key is missing/placeholder
    user_key = st.sidebar.text_input("Enter Google API Key:", type="password")
    api_key = user_key.strip()

if not api_key or "your_google_api_key" in api_key:
    st.warning("⚠️ **API Key Required**: Please update `GOOGLE_API_KEY` in `.env` or enter it in the sidebar.")
    st.stop()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)

output_parser = StrOutputParser()

def rag_chain_fn(question: str) -> dict:
    source_docs = retrieve(question, k=3)
    context = format_context(source_docs)
    messages = prompt_template.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    answer = output_parser.invoke(response)
    return {"question": question, "answer": answer, "source_docs": source_docs}

rag_chain = RunnableLambda(rag_chain_fn)

query = st.text_input("Enter your question:")

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
                time.sleep(6 * retries)
            else:
                raise e

if query:
    with st.spinner("Searching and generating answer using Google Gemini..."):
        try:
            result = invoke_app_with_retry(query)
            
            st.markdown("### Answer")
            st.info(result["answer"])
            
            st.markdown("### Sources Retrieved")
            for i, doc in enumerate(result["source_docs"], 1):
                fname = doc.metadata.get("filename", "unknown").replace("_", " ").title()
                page = doc.metadata.get("page_display", "?")
                with st.expander(f"Source {i}: {fname} (Page {page})"):
                    st.write(doc.page_content)
                    
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                st.warning("⏱️ **Google API Daily Quota Exceeded (429)**: The free-tier API daily quota (GenerateRequestsPerDay) has been reached. Please generate a new free API key at https://aistudio.google.com/app/apikey or update `.env`.")
            elif "503" in err_str or "UNAVAILABLE" in err_str:
                st.warning("📡 **Google API Temporary Overload (503)**: Google servers are experiencing high traffic. Please retry in a few seconds.")
            else:
                st.error(f"An error occurred: {err_str}")
