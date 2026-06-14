"""
ingestion/ingest.py

Step 1 — PDF Ingestion Pipeline
--------------------------------
Reads all CA Final Paper 2 PDFs from data/pdfs/,
extracts text using PyMuPDF, chunks them intelligently,
embeds using HuggingFace sentence-transformers (free, local),
and persists to ChromaDB vector store.

Run this ONCE before starting the chatbot:
    python -m ingestion.ingest
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

import fitz  # PyMuPDF
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PDF_DIR         = Path(os.getenv("PDF_DIR", "data/pdfs"))
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "data/vectorstore"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 150))
COLLECTION_NAME = "ca_final_paper2"


# ── Step 1A: Extract text from PDFs using PyMuPDF ────────────────────────────
def extract_text_from_pdf(pdf_path: Path) -> list[Document]:
    """
    Extract text page-by-page from a PDF.
    Returns a list of LangChain Documents, one per page.
    Metadata includes: source filename, chapter name, page number.
    """
    documents = []
    chapter_name = pdf_path.stem  # e.g. "Final-group2-paper2-chapter3"

    try:
        pdf = fitz.open(str(pdf_path))
        logger.info(f"  Opening: {pdf_path.name} ({len(pdf)} pages)")

        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text")  # plain text extraction

            # Skip near-empty pages (headers/footers only)
            if len(text.strip()) < 50:
                continue

            documents.append(Document(
                page_content=text,
                metadata={
                    "source":      pdf_path.name,
                    "chapter":     chapter_name,
                    "page_number": page_num,
                    "total_pages": len(pdf),
                }
            ))

        pdf.close()
        logger.success(f"  ✓ Extracted {len(documents)} pages from {pdf_path.name}")

    except Exception as e:
        logger.error(f"  ✗ Failed to read {pdf_path.name}: {e}")

    return documents


# ── Step 1B: Load all PDFs in order ──────────────────────────────────────────
def load_all_pdfs(pdf_dir: Path) -> list[Document]:
    """
    Load all PDFs from the directory.
    Files are sorted so chapters load in order (chapter1 → chapter15 → appendix).
    """
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}. Please add your CA Final Paper 2 PDFs.")
        sys.exit(1)

    logger.info(f"\n📂 Found {len(pdf_files)} PDF files in {pdf_dir}")

    all_documents = []
    for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
        docs = extract_text_from_pdf(pdf_path)
        all_documents.extend(docs)

    logger.success(f"\n✓ Total pages extracted: {len(all_documents)}")
    return all_documents


# ── Step 1C: Chunk documents ──────────────────────────────────────────────────
def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split page-level documents into smaller overlapping chunks.
    
    Why RecursiveCharacterTextSplitter?
    - Tries to split on paragraphs → sentences → words in order
    - Preserves semantic context better than fixed-size splitting
    - Overlap ensures a concept spanning two chunks isn't lost
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],  # paragraph → line → sentence
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"\n✂️  Chunking complete:")
    logger.info(f"   Pages in  → {len(documents)}")
    logger.info(f"   Chunks out → {len(chunks)}")
    logger.info(f"   Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    return chunks


# ── Step 1D: Embed and store in ChromaDB ─────────────────────────────────────
def embed_and_store(chunks: list[Document]) -> Chroma:
    """
    Embed chunks using HuggingFace sentence-transformers (runs locally, free).
    Store in ChromaDB with persistence to disk.
    
    First run: downloads the embedding model (~90MB), takes ~2 min.
    Subsequent runs: loads from cache instantly.
    """
    logger.info(f"\n🤗 Loading embedding model: {EMBEDDING_MODEL}")
    logger.info("   (First run downloads ~90MB model — one time only)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info(f"\n💾 Creating ChromaDB vector store at: {VECTORSTORE_DIR}")
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    # Embed in batches with progress (chromadb handles batching internally)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
    )

    logger.success(f"\n✅ Vector store created successfully!")
    logger.success(f"   Collection : {COLLECTION_NAME}")
    logger.success(f"   Total chunks stored: {vectorstore._collection.count()}")
    logger.success(f"   Location: {VECTORSTORE_DIR.resolve()}")

    return vectorstore


# ── Step 1E: Verify retrieval works ──────────────────────────────────────────
def verify_retrieval(vectorstore: Chroma):
    """
    Quick sanity check — query the vector store with a sample question
    to confirm retrieval is working before moving to Step 2.
    """
    logger.info("\n🔍 Running retrieval sanity check...")

    test_queries = [
        "What is the meaning of financial instruments?",
        "Explain lease accounting under Ind AS 116",
        "What are the disclosure requirements for related party transactions?",
    ]

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    for query in test_queries:
        results = retriever.invoke(query)
        logger.info(f"\n  Query: '{query}'")
        logger.info(f"  Top result from: {results[0].metadata.get('source', 'unknown')} "
                    f"(page {results[0].metadata.get('page_number', '?')})")
        logger.info(f"  Preview: {results[0].page_content[:120].strip()}...")

    logger.success("\n✅ Retrieval working correctly!")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_ingestion():
    logger.info("=" * 60)
    logger.info("  CA Final Paper 2 — RAG Ingestion Pipeline")
    logger.info("  Step 1: PDF → Chunks → Embeddings → ChromaDB")
    logger.info("=" * 60)

    # Check if vector store already exists
    if VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir()):
        logger.warning(f"\n⚠️  Vector store already exists at {VECTORSTORE_DIR}")
        answer = input("  Re-ingest? This will overwrite existing data. (y/n): ").strip().lower()
        if answer != "y":
            logger.info("Skipping ingestion. Existing vector store will be used.")
            return

    # Pipeline
    documents = load_all_pdfs(PDF_DIR)
    chunks    = chunk_documents(documents)
    vs        = embed_and_store(chunks)
    verify_retrieval(vs)

    logger.info("\n🎉 Step 1 Complete! You can now run Step 2 (FastAPI + LangGraph agent).")


if __name__ == "__main__":
    run_ingestion()