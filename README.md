# 📚 Research Paper Answer Bot (RAG Capstone)

> **End-to-End Retrieval-Augmented Generation System for Scientific Literature**  
> *GenAI Pinnacle Plus Program — Final Capstone Project*

---

## 📌 Executive Summary

This project implements an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline designed to accurately answer questions grounded in complex AI research literature. The system indexes two foundational deep learning research papers (*Attention Is All You Need* and *BERT: Pre-training of Deep Bidirectional Transformers*) and employs a hybrid retrieval strategy combining dense semantic vector search with BM25 keyword matching to provide precise, citeable answers powered by Google Gemini.

---

## 🛠️ Architecture & Pipeline

```
[ PDF Documents ] 
       │
       ▼ (PyPDFLoader)
[ Raw Document Pages & Metadata ]
       │
       ▼ (RecursiveCharacterTextSplitter: chunk_size=500, overlap=50)
[ 246 Text Chunks ] ────┐
       │                 │
       ▼                 ▼
[ ChromaDB Vector Store ] [ BM25 Keyword Index ]
(BAAI/bge-m3 & mxbai)    (In-memory BM25Okapi)
       │                 │
       └──────┬──────────┘
              ▼ (Hybrid Fusion Search: alpha=0.5)
      [ Top-3 Retrieved Chunks ]
              │
              ▼
    [ Grounded Prompt Template ]
              │
              ▼
   [ Google Gemini LLM ] ──► [ Answer + Paper & Page Citations ]
```

---

## 🚀 Key Features & Implementation Details

1. **Corpus Ingestion & Metadata Tracking**:
   - Parses multi-page PDF research papers while extracting document titles, page numbers, and structural metadata.
   - 31 total pages ingested without extraction errors.

2. **Dual Embedding Model Comparison**:
   - Compares **`BAAI/bge-m3`** (1024-dim, multi-lingual, dense) against **`mixedbread-ai/mxbai-embed-large-v1`** (1024-dim, English).
   - `BAAI/bge-m3` was selected for production due to higher semantic alignment across scientific terminology.

3. **Hybrid Retrieval Strategy (BM25 + Dense)**:
   - Combines vector similarity scoring with sparse BM25 keyword matching using weighted score fusion ($\alpha = 0.5$).
   - Overcomes dense retrieval weaknesses on domain-specific acronyms, hyperparameter names, and exact math notation.

4. **Strict Grounded Prompt & Fallback**:
   - Enforces zero-hallucination generation.
   - Triggers explicit *"I don't know based on the provided context"* fallbacks when questions fall outside the ingested corpus.

5. **Interactive Streamlit Web Interface**:
   - Provides a clean UI for interactive querying, model parameter configuration, and source expanders detailing exact paper pages and chunk extracts.

---

## 📁 Repository Structure

```
rag_capstone/
├── rag_notebook.ipynb        # Main Capstone Jupyter Notebook (Sections 1-6)
├── app.py                    # Interactive Streamlit Web Application
├── presentation.html         # 14-slide HTML presentation deck (Print to PDF ready)
├── requirements.txt          # Python dependency specifications
├── .env                      # Environment configuration template (API keys)
├── data/                     # Source PDF research papers
│   ├── attention_is_all_you_need.pdf
│   └── bert_pretraining.pdf
└── chroma_db/                # Persistent ChromaDB vector databases
    ├── bge_m3/               # BAAI/bge-m3 collection
    └── mxbai/                # mxbai-embed-large-v1 collection
```

---

## 🧰 Setup & Installation

### 1. Environment Setup

```bash
# Clone repository and enter project directory
cd rag_capstone

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

Access the interface at `http://localhost:8501`.

---

## 📊 Evaluation Summary (10-Question Benchmark)

| Metric | Result | Note |
| :--- | :---: | :--- |
| **Grounded Answer Rate** | **8 / 10** | Accurate answers derived strictly from corpus context |
| **Out-of-Corpus Fallback** | **2 / 10** | Correctly triggered "I don't know" on un-ingested topics |
| **Hallucinated Answers** | **0 / 10** | Zero hallucination recorded |
| **Citation Precision** | **100%** | All answered questions included Paper Title + Page Number |

---

## 📄 License & Attribution

Developed for the **GenAI Pinnacle Plus Program Capstone Submission**.  
Research papers used for evaluation are public domain open-access publications (*Vaswani et al., 2017* and *Devlin et al., 2018*).
