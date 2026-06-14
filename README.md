# ca-rag-chatbot
# CA Final Paper 2 — RAG Chatbot

An AI-powered chatbot for CA Final students to query Paper 2 (Advanced Financial Reporting) content.  
Built with LangChain, LangGraph, FastAPI, ChromaDB, Groq (LLaMA 3.1 70B), Tavily, and Streamlit.

---

## Architecture

```
User (Streamlit UI)
      ↓
FastAPI /chat endpoint
      ↓
LangGraph Agent
  ├── Router Node      → decides: RAG or Web Search?
  ├── RAG Node         → ChromaDB retrieval → Groq LLM → answer with sources
  └── Web Search Node  → Tavily → for ICAI notifications, exam dates
      ↓
Response + Source Citations
```

---

## Project Structure

```
ca-rag-chatbot/
├── data/
│   ├── pdfs/               ← Place all CA Final Paper 2 PDFs here
│   └── vectorstore/        ← ChromaDB auto-created here after ingestion
├── ingestion/
│   └── ingest.py           ← Step 1: PDF → Chunks → Embeddings → ChromaDB
├── agent/                  ← Step 3: LangGraph agent with RAG + web search
├── api/                    ← Step 2: FastAPI backend
├── frontend/               ← Step 4: Streamlit chat UI
├── utils/                  ← Shared helpers
├── .env.example            ← Copy to .env and fill in API keys
└── requirements.txt
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo>
cd ca-rag-chatbot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY and TAVILY_API_KEY
```

Get free API keys:
- Groq: https://console.groq.com
- Tavily: https://tavily.com

### 3. Add your PDFs

```
Place all CA Final Paper 2 PDFs inside:  data/pdfs/
```

Expected files:
- `Final-group2-paper2-Initial pages.pdf`
- `Final-group2-paper2-chapter1.pdf` through `chapter15.pdf`
- `Final-group2-paper2-Appendix.pdf`

### 4. Run ingestion (Step 1)

```bash
python -m ingestion.ingest
```

This will:
- Extract text from all PDFs
- Split into chunks
- Download HuggingFace embedding model (~90MB, one-time)
- Store embeddings in ChromaDB locally

---

## Steps Roadmap

- [x] **Step 1** — PDF Ingestion Pipeline (ingest.py)
- [ ] **Step 2** — FastAPI backend with `/chat` and `/ingest` endpoints
- [ ] **Step 3** — LangGraph agent with RAG + Tavily web search nodes
- [ ] **Step 4** — Streamlit chat UI with source citations
- [ ] **Step 5** — Docker + HuggingFace Spaces deployment

---

## Tech Stack

| Layer | Technology |
|---|---|
| PDF Extraction | PyMuPDF (fitz) |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local, free) |
| Vector Store | ChromaDB (local persistence) |
| LLM | Groq — LLaMA 3.1 70B |
| Agent Orchestration | LangGraph |
| Web Search | Tavily API |
| Backend | FastAPI |
| Frontend | Streamlit |
| Deployment | Docker + HuggingFace Spaces |