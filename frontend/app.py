"""
frontend/app.py — CA Final AFM Study Assistant
Modern dark-academia UI with collapsible sources
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AFM Tutor · CA Final",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ─────────────────────────────────────────────────────────────
# Palette: deep navy + warm gold + soft ivory — evokes CA exam hall + serious study
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #0f1117;
    color: #e8e6e1;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #2a2f3e;
}
[data-testid="stSidebar"] * {
    color: #c9c5bc !important;
}

/* ── App title ── */
.app-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #f0c040;
    letter-spacing: -0.5px;
    margin-bottom: 2px;
}
.app-subtitle {
    font-size: 0.85rem;
    color: #8a8680;
    margin-bottom: 1.5rem;
}

/* ── Route badges ── */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
    text-transform: uppercase;
}
.badge-rag       { background: #1a3a5c; color: #60aaff; border: 1px solid #1e4a7a; }
.badge-websearch { background: #1a3a2a; color: #5dd88a; border: 1px solid #1e5a36; }
.badge-general   { background: #2d1f3d; color: #c084fc; border: 1px solid #4a2d6e; }

/* ── Answer text ── */
.answer-text {
    font-size: 0.95rem;
    line-height: 1.75;
    color: #e0ddd8;
}

/* ── PDF source card ── */
.pdf-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #1a2035;
    border: 1px solid #2a3555;
    border-left: 3px solid #f0c040;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 5px 0;
    font-size: 0.85rem;
}
.pdf-card-info { color: #c9c5bc; }
.pdf-card-info b { color: #f0e6c0; }

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: #1a1f2e !important;
    border: 1px solid #2a3555 !important;
    color: #e8e6e1 !important;
    border-radius: 12px !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #141820 !important;
    border: 1px solid #1e2436 !important;
    border-radius: 12px !important;
    padding: 4px 8px !important;
    margin-bottom: 10px !important;
}

/* ── Sample question buttons ── */
.stButton button {
    background: #1a2035 !important;
    border: 1px solid #2a3555 !important;
    color: #c9c5bc !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    text-align: left !important;
    padding: 8px 12px !important;
    transition: all 0.2s ease;
}
.stButton button:hover {
    border-color: #f0c040 !important;
    color: #f0e6c0 !important;
}

/* ── Expander (collapsible sources) ── */
[data-testid="stExpander"] {
    background: #141820 !important;
    border: 1px solid #2a3555 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #8a8680 !important;
    font-size: 0.82rem !important;
}

/* ── Divider ── */
hr { border-color: #2a2f3e !important; }

/* ── Success/error ── */
.stSuccess { background: #1a3a2a !important; }
.stError   { background: #3a1a1a !important; }

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📐 AFM Tutor")
    st.caption("CA Final · Advanced Financial Management")
    st.divider()

    st.markdown("**How it works**")
    st.markdown("""
- 📘 AFM topic → ICAI study material
- 🌐 Current info → Tavily web search  
- 💬 Greeting → direct response
    """)

    st.divider()
    st.markdown("**Chapters**")
    chapters = [
        "1. Introduction to AFM",       "2. Capital Budgeting",
        "3. Financial Risk Management",  "4. Security Analysis",
        "5. Security Valuation",         "6. Portfolio Management",
        "7. Securitisation",             "8. Mutual Funds",
        "9. Derivatives",                "10. Foreign Exchange",
        "11. International Finance",     "12. Interest Rate Risk",
        "13. Corporate Valuation",       "14. Mergers & Acquisitions",
        "15. Business Valuation",
    ]
    for ch in chapters:
        st.caption(ch)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        try:
            r = requests.get(f"{API_URL}/health", timeout=2)
            st.success("API ✓") if r.status_code == 200 else st.error("API ✗")
        except:
            st.error("API ✗")


# ── Helpers ───────────────────────────────────────────────────────────────────
def badge(route: str) -> str:
    config = {
        "rag":       ("📘 ICAI Material",  "badge-rag"),
        "websearch": ("🌐 Web Search",     "badge-websearch"),
        "general":   ("💬 General",        "badge-general"),
    }
    label, cls = config.get(route, ("❓", "badge-general"))
    return f'<span class="badge {cls}">{label}</span>'


def render_sources(sources: list, route: str):
    if not sources:
        return

    if route == "rag":
        st.markdown("---")
        st.markdown('<p style="font-size:0.8rem;color:#8a8680;margin-bottom:6px;">📚 SOURCES FROM ICAI STUDY MATERIAL</p>', unsafe_allow_html=True)
        for src in sources:
            name     = src.get("display_name", "Chapter")
            page     = src.get("page", "?")
            filename = src.get("filename", "")
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(
                    f'<div class="pdf-card">'
                    f'<span class="pdf-card-info">📄 <b>{name}</b> &nbsp;·&nbsp; Page {page}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col2:
                if filename:
                    st.link_button("Open ↗", f"{API_URL}/pdfs/{filename}", use_container_width=True)

    elif route == "websearch":
        st.markdown("---")
        with st.expander(f"🌐 Web Sources ({len(sources)} results)", expanded=False):
            for i, src in enumerate(sources, 1):
                title   = src.get("title", "Web Result")
                url     = src.get("url", "")
                snippet = src.get("snippet", "")
                st.markdown(f"**{i}. [{title}]({url})**")
                st.caption(snippet)
                if i < len(sources):
                    st.divider()


# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown('<p class="app-title">CA Final AFM Tutor</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Powered by ICAI study material · Groq LLaMA 3.3 · Tavily Search</p>', unsafe_allow_html=True)

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            route = msg.get("route", "general")
            st.markdown(badge(route), unsafe_allow_html=True)
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"], route)
        else:
            st.markdown(msg["content"])

# Sample questions (only when chat is empty)
if not st.session_state.messages:
    st.markdown('<p style="color:#8a8680;font-size:0.85rem;margin-top:2rem;">Try asking:</p>', unsafe_allow_html=True)
    samples = [
        "What is the Black-Scholes model?",
        "Explain portfolio diversification",
        "Difference between futures and forwards?",
        "How is firm value calculated under APV?",
        "When is CA Final Nov 2026 exam?",
        "What is capital rationing?",
    ]
    cols = st.columns(3)
    for i, q in enumerate(samples):
        with cols[i % 3]:
            if st.button(q, key=f"s{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()

# Chat input
if question := st.chat_input("Ask anything about CA Final AFM..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(""):
            try:
                resp = requests.post(f"{API_URL}/chat", json={"question": question}, timeout=60)
                data = resp.json()
                if resp.status_code == 200:
                    route   = data.get("route", "general")
                    answer  = data.get("answer", "")
                    sources = data.get("sources", [])
                    st.markdown(badge(route), unsafe_allow_html=True)
                    st.markdown(answer)
                    render_sources(sources, route)
                    st.session_state.messages.append({
                        "role": "assistant", "content": answer,
                        "route": route, "sources": sources,
                    })
                else:
                    st.error(f"Error: {data.get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API. Is `uvicorn api.main:app` running?")
            except Exception as e:
                st.error(f"Error: {e}")