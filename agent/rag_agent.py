"""
agent/rag_agent.py

Step 2A — LangGraph Agent with Query Understanding
----------------------------------------------------
Architecture:

  [START] → router_node
               ├── "rag"        → query_understanding_node → rag_node → [END]
               ├── "websearch"  → websearch_node                       → [END]
               └── "general"    → general_node                         → [END]

Key improvement: query_understanding_node sits between router and RAG.
It uses an LLM call to intelligently rewrite the query, detect chapter/topic
filters, set retrieval depth, and identify the user's intent — so RAG always
searches for the RIGHT thing regardless of how the question is phrased.
"""

import os
import json
from pathlib import Path
from typing import TypedDict, Literal, Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "data/vectorstore"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMP       = float(os.getenv("GROQ_TEMPERATURE", 0))
COLLECTION_NAME = "ca_final_paper2"

CHAPTER_MAP = {
    "1":  ("chapter1",  "Introduction to AFM"),
    "2":  ("chapter2",  "Capital Budgeting"),
    "3":  ("chapter3",  "Financial Risk Management"),
    "4":  ("chapter4",  "Security Analysis"),
    "5":  ("chapter5",  "Security Valuation"),
    "6":  ("chapter6",  "Portfolio Management"),
    "7":  ("chapter7",  "Securitisation"),
    "8":  ("chapter8",  "Mutual Funds"),
    "9":  ("chapter9",  "Derivatives"),
    "10": ("chapter10", "Foreign Exchange"),
    "11": ("chapter11", "International Finance"),
    "12": ("chapter12", "Interest Rate Risk Management"),
    "13": ("chapter13", "Corporate Valuation"),
    "14": ("chapter14", "Mergers and Acquisitions"),
    "15": ("chapter15", "Business Valuation"),
}


# ── Agent State ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question:        str
    route:           str
    search_query:    str
    chapter_filter:  Optional[str]
    topic_name:      Optional[str]
    retrieval_k:     int
    intent:          str
    answer:          str
    sources:         list[dict]


# ── Load shared resources ─────────────────────────────────────────────────────
def _load_llm():
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=GROQ_TEMP,
        api_key=os.getenv("GROQ_API_KEY"),
    )

def _load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )

def _load_tavily():
    return TavilySearchResults(
        max_results=4,
        api_key=os.getenv("TAVILY_API_KEY"),
    )

llm         = _load_llm()
vectorstore = _load_vectorstore()
tavily      = _load_tavily()


# ── Node 1: Router ────────────────────────────────────────────────────────────
def router_node(state: AgentState) -> AgentState:
    system = """You are a router for a CA Final AFM study assistant.

Classify into exactly one:
- "rag"        → AFM syllabus: concepts, formulas, chapters, problems, exam prep
- "websearch"  → current info: exam dates, ICAI notifications, live news
- "general"    → greetings, what can you do, unrelated

Reply ONLY: rag, websearch, or general."""

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["question"]),
    ])

    route = response.content.strip().lower()
    if route not in ("rag", "websearch", "general"):
        route = "rag"

    return {**state, "route": route}


# ── Node 2: Query Understanding ───────────────────────────────────────────────
def query_understanding_node(state: AgentState) -> AgentState:
    """
    THE KEY NODE — intelligently interprets the user's question and produces
    a structured retrieval plan. This single node handles ALL edge cases:
    chapter requests, ordinal references, topic names, intent detection.
    No hardcoding needed anywhere else.
    """

    chapter_context = "\n".join(
        f"  Chapter {num}: {topic}" for num, (_, topic) in CHAPTER_MAP.items()
    )
    chapter_keys = "\n".join(
        f"  Chapter {num} ({topic}) → \"{key}\""
        for num, (key, topic) in CHAPTER_MAP.items()
    )

    system = f"""You are a query understanding system for a CA Final AFM RAG chatbot.

The knowledge base has these chapters:
{chapter_context}

Given a user question, return a JSON object:
{{
  "search_query": "<expanded technical search terms for vector similarity search>",
  "chapter_filter": "<exact chapter key from list below, or null>",
  "retrieval_k": <integer 4-20>,
  "intent": "<teach|explain|solve|list|answer>"
}}

Chapter filter keys (use ONLY these exact strings or null):
{chapter_keys}

Rules for search_query:
- Expand the question into rich technical keywords for semantic search
- "teach chapter 2" → "capital budgeting NPV IRR payback period cash flows investment appraisal techniques"
- "9th chapter" → "derivatives futures options swaps hedging speculation"
- "WACC" → "weighted average cost of capital WACC equity debt cost formula"
- Never use the original question as-is if it contains chapter references

Rules for chapter_filter:
- Set when user mentions chapter by NUMBER ("chapter 2", "2nd chapter", "second chapter")
- Set when user mentions topic NAME that maps to a chapter ("capital budgeting chapter", "derivatives chapter")  
- Set when user says ordinal ("ninth chapter" → chapter9, "third" → chapter3)
- null for specific concept questions that don't target one chapter

Rules for retrieval_k:
- 15-20: "teach me", "explain fully", "summarise the chapter", "cover everything"
- 10-14: "what topics", "key concepts", "overview", "what is covered"
- 4-8:   specific concept/formula/problem questions

Rules for intent:
- teach:   teach me, explain the chapter, cover this chapter, summarise
- explain: explain this concept, what is X, how does X work
- solve:   solve, calculate, work out, find the value, numerical
- list:    list all, what are the topics, give me all formulas
- answer:  specific factual question

Return ONLY valid JSON. No markdown, no explanation."""

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["question"]),
    ])

    try:
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        plan = json.loads(raw.strip())
    except Exception:
        plan = {
            "search_query":   state["question"],
            "chapter_filter": None,
            "retrieval_k":    6,
            "intent":         "answer",
        }

    # Resolve topic name
    topic_name = None
    chapter_filter = plan.get("chapter_filter")
    if chapter_filter:
        for num, (key, topic) in CHAPTER_MAP.items():
            if key == chapter_filter:
                topic_name = topic
                break

    return {
        **state,
        "search_query":   plan.get("search_query", state["question"]),
        "chapter_filter": chapter_filter,
        "topic_name":     topic_name,
        "retrieval_k":    min(max(int(plan.get("retrieval_k", 6)), 4), 20),
        "intent":         plan.get("intent", "answer"),
    }


# ── Node 3: RAG ───────────────────────────────────────────────────────────────
def rag_node(state: AgentState) -> AgentState:
    search_query   = state["search_query"]
    chapter_filter = state.get("chapter_filter")
    topic_name     = state.get("topic_name")
    k              = state.get("retrieval_k", 6)
    intent         = state.get("intent", "answer")
    question       = state["question"]

    # ── Retrieval ─────────────────────────────────────────────────────────────
    if chapter_filter:
        results = vectorstore.get(
            where={"chapter": {"$eq": f"Final-group1-paper2-AdvancedFinancialManagement-{chapter_filter}"}},
            include=["documents", "metadatas"],
            limit=k,
        )
        raw_docs  = results["documents"]
        raw_metas = results["metadatas"]
        context_parts = [
            f"[Source {i} | Page {m.get('page_number','?')}]\n{t}"
            for i, (t, m) in enumerate(zip(raw_docs, raw_metas), 1)
        ]
        sources_meta = raw_metas
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(search_query)
        context_parts = [
            f"[Source {i} | {d.metadata.get('chapter','?')} | Page {d.metadata.get('page_number','?')}]\n{d.page_content}"
            for i, d in enumerate(docs, 1)
        ]
        sources_meta = [d.metadata for d in docs]

    context = "\n\n---\n\n".join(context_parts)
    subject = topic_name or "Advanced Financial Management"

    # ── Intent-aware prompts ──────────────────────────────────────────────────
    intent_prompts = {
        "teach": f"""You are an expert CA Final AFM tutor teaching {subject} from ICAI material.
Structure your response:
## Overview
What this chapter covers and its exam importance.
## Key Concepts  
Explain each major concept clearly with examples.
## Important Formulas
All formulas with variable definitions.
## Exam Focus
Common question patterns, tricky areas, marks weightage tips.""",

        "explain": f"""You are an expert CA Final AFM tutor explaining a concept from {subject}.
Provide: Definition → How it works → Formula (if any) → Example → Exam relevance.
Use only provided ICAI material.""",

        "solve": f"""You are an expert CA Final AFM tutor solving a problem.
Show: Given information → Formula/approach → Step-by-step working → Final answer.
Reference ICAI material where relevant.""",

        "list": f"""You are an expert CA Final AFM tutor summarising {subject}.
Provide a well-organised list or table. Use bullet points for clarity.
Base everything strictly on the provided ICAI material.""",

        "answer": f"""You are an expert CA Final AFM tutor.
Answer using ONLY the provided ICAI study material.
Be concise, accurate, and cite [Source N] when referencing specific points.
If context doesn't fully answer, say so honestly.""",
    }

    system = intent_prompts.get(intent, intent_prompts["answer"])
    prompt = f"ICAI Study Material:\n{context}\n\nStudent Question: {question}\n\nAnswer:"

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ])

    # ── Build sources ─────────────────────────────────────────────────────────
    sources = []
    seen = set()
    for meta in sources_meta:
        source = meta.get("source", "")
        page   = meta.get("page_number", "?")
        chapter = meta.get("chapter", "")
        key = f"{source}_{page}"
        if key not in seen:
            seen.add(key)
            display = topic_name or chapter.replace(
                "Final-group1-paper2-AdvancedFinancialManagement-", ""
            ).replace("-", " ").title()
            sources.append({
                "type": "pdf", "display_name": display,
                "page": page, "filename": source,
                "pdf_url": f"/pdfs/{source}",
            })

    return {**state, "answer": response.content, "sources": sources}


# ── Node 4: Web Search ────────────────────────────────────────────────────────
def websearch_node(state: AgentState) -> AgentState:
    results = tavily.invoke(state["question"])
    context = "\n---\n".join(
        f"[Web Source {i}]\nTitle: {r.get('title','')}\nURL: {r.get('url','')}\nContent: {r.get('content','')}"
        for i, r in enumerate(results, 1)
    )
    system = """You are a helpful CA Final exam assistant. Answer using web search results.
Cite [Web Source N]. For exam dates, recommend verifying from official ICAI website."""

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Search Results:\n{context}\n\nQuestion: {state['question']}"),
    ])

    sources = [{
        "type": "web", "title": r.get("title", "Web Result"),
        "url": r.get("url", ""), "snippet": r.get("content", "")[:150] + "...",
    } for r in results]

    return {**state, "answer": response.content, "sources": sources}


# ── Node 5: General ───────────────────────────────────────────────────────────
def general_node(state: AgentState) -> AgentState:
    system = """You are a CA Final AFM study assistant.
Introduce yourself warmly. You can: answer AFM questions from ICAI material,
teach full chapters with structured notes, explain concepts, solve numericals,
and search the web for current exam info."""

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["question"]),
    ])
    return {**state, "answer": response.content, "sources": []}


# ── Routing ───────────────────────────────────────────────────────────────────
def route_question(state: AgentState) -> Literal["query_understanding", "websearch", "general"]:
    return {"rag": "query_understanding", "websearch": "websearch", "general": "general"}.get(
        state["route"], "query_understanding"
    )


# ── Build Graph ───────────────────────────────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("router",              router_node)
    graph.add_node("query_understanding", query_understanding_node)
    graph.add_node("rag",                 rag_node)
    graph.add_node("websearch",           websearch_node)
    graph.add_node("general",             general_node)
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_question, {
        "query_understanding": "query_understanding",
        "websearch": "websearch",
        "general": "general",
    })
    graph.add_edge("query_understanding", "rag")
    graph.add_edge("rag",       END)
    graph.add_edge("websearch", END)
    graph.add_edge("general",   END)
    return graph.compile()

agent = build_agent()

def ask(question: str) -> dict:
    return agent.invoke({
        "question": question, "route": "",
        "search_query": "", "chapter_filter": None,
        "topic_name": None, "retrieval_k": 6,
        "intent": "answer", "answer": "", "sources": [],
    })