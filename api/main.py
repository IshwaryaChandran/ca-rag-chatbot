"""
api/main.py

Step 2B — FastAPI Backend
--------------------------
Exposes two endpoints:

  POST /chat        → receives question, returns answer + sources
  GET  /pdfs/{filename} → serves PDF files so frontend can open them
  GET  /health      → health check
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent.rag_agent import ask

load_dotenv()

PDF_DIR = Path(os.getenv("PDF_DIR", "data/pdfs"))

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CA Final AFM RAG Chatbot",
    description="AI-powered study assistant for CA Final Advanced Financial Management",
    version="1.0.0",
)

# Allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the Black-Scholes model for option pricing?"
            }
        }


class SourceModel(BaseModel):
    type:         str            # "pdf" or "web"
    # PDF fields
    display_name: str | None = None
    page:         int | str | None = None
    filename:     str | None = None
    pdf_url:      str | None = None
    # Web fields
    title:        str | None = None
    url:          str | None = None
    snippet:      str | None = None


class ChatResponse(BaseModel):
    question: str
    route:    str
    answer:   str
    sources:  list[SourceModel]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "CA Final AFM RAG Chatbot"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Accepts a question, routes through LangGraph agent, returns answer + sources.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = ask(request.question)
        return ChatResponse(
            question=result["question"],
            route=result["route"],
            answer=result["answer"],
            sources=[SourceModel(**s) for s in result["sources"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pdfs/{filename}")
def serve_pdf(filename: str):
    """
    Serves PDF files from data/pdfs/ directory.
    Called by frontend when user clicks 'Open PDF' button.
    """
    # Security: prevent path traversal
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    pdf_path = PDF_DIR / filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {filename}")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=filename,
    )


@app.get("/pdfs")
def list_pdfs():
    """Lists all available PDF files."""
    pdfs = [f.name for f in PDF_DIR.glob("*.pdf")]
    return {"pdfs": sorted(pdfs), "count": len(pdfs)}