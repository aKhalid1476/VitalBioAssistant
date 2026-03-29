"""
VitalBio Bloodborne Pathogen Compliance Assistant — FastAPI entrypoint.

Endpoints:
  GET  /health  — liveness check
  POST /chat    — compliance Q&A with optional session continuity
"""

from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import rag_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("VitalBio Assistant API started.")
    yield
    print("VitalBio Assistant API shutting down.")


app = FastAPI(
    title="VitalBio Bloodborne Pathogen Compliance Assistant",
    description=(
        "RAG-powered Q&A over OSHA 29 CFR § 1910.1030 – Bloodborne Pathogens. "
        "Supports multi-turn conversation via session_id."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- Request / Response Models ---

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The compliance question to ask",
        examples=["What PPE is required when handling blood?"],
    )
    session_id: Optional[str] = Field(
        default="default",
        description=(
            "Session identifier for conversation continuity. "
            "Omit or use 'default' for single-turn use."
        ),
        examples=["user-abc-123"],
    )


class SourceDocument(BaseModel):
    page: int
    source: str
    chunk_index: int
    regulation: str
    content_preview: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[SourceDocument]


# --- Endpoints ---

@app.get("/health", tags=["System"])
async def health():
    """Liveness check."""
    return {"status": "ok", "service": "VitalBioAssistant"}


@app.post("/chat", response_model=ChatResponse, tags=["Compliance"])
async def chat(request: ChatRequest):
    """
    Ask a compliance question about OSHA 29 CFR § 1910.1030.

    Supports multi-turn conversation via `session_id`. Each session maintains
    its own conversation history for the lifetime of the server process.
    """
    session_id = request.session_id or "default"

    try:
        result = await rag_chain.ainvoke(
            {"input": request.message},
            config={"configurable": {"session_id": session_id}},
        )
    except Exception as e:
        print(f"RAG chain error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred processing your request.")

    answer: str = result.get("answer", "")
    context_docs = result.get("context", [])

    sources = []
    for doc in context_docs:
        m = doc.metadata
        sources.append(SourceDocument(
            page=m.get("page", -1),
            source=m.get("source", "unknown"),
            chunk_index=m.get("chunk_index", -1),
            regulation=m.get("regulation", "29 CFR § 1910.1030"),
            content_preview=doc.page_content[:200],
        ))

    return ChatResponse(answer=answer, session_id=session_id, sources=sources)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
