"""GEM2 RAG — lightweight FastAPI chat server."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import Config
from src.gemini_embedder import GeminiEmbedder
from src.supabase_client import SupabaseVectorClient
from src.retrieval.query_engine import QueryEngine

# ---------------------------------------------------------------------------
# Shared resources (initialised once at startup)
# ---------------------------------------------------------------------------
_embedder: GeminiEmbedder | None = None
_engine: QueryEngine | None = None
_default_model: str = "anthropic/claude-sonnet-4"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _engine, _default_model
    cfg = Config()
    _default_model = cfg.OPENROUTER_MODEL
    _embedder = GeminiEmbedder(cfg)
    await _embedder.__aenter__()
    db = SupabaseVectorClient(cfg)
    _engine = QueryEngine(_embedder, db, cfg)
    yield
    if _embedder:
        await _embedder.close()


app = FastAPI(title="GEM2 RAG", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

TEMPLATE = Path(__file__).parent / "templates" / "index.html"


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    filter_type: str | None = None
    top_k: int = 5
    model: str | None = None


# ---------------------------------------------------------------------------
# Human-readable error translation
# ---------------------------------------------------------------------------
def _friendly_error(exc: Exception) -> str:
    msg = str(exc)

    # Supabase: RPC function missing (schema not applied yet)
    if "match_media_embeddings" in msg and ("PGRST202" in msg or "schema cache" in msg.lower()):
        return (
            "The database search function is missing. "
            "Please run schema/supabase_schema.sql in your Supabase SQL editor to set up the vector table, "
            "then restart the server."
        )

    # Supabase: table doesn't exist
    if "media_embeddings" in msg and ("42P01" in msg or "does not exist" in msg.lower()):
        return (
            "The media_embeddings table doesn't exist yet. "
            "Run schema/supabase_schema.sql in your Supabase dashboard first."
        )

    # Supabase: connection / auth failure
    if "supabase" in msg.lower() or "postgrest" in msg.lower() or "PGRST" in msg:
        return (
            "Could not reach the database. "
            "Check that SUPABASE_PROJECT_URL and SUPABASE_PROJECT_KEY are correct in your .env file."
        )

    # Gemini: bad API key
    if "GEMINI" in msg.upper() or ("generativelanguage" in msg and "401" in msg):
        return (
            "The Gemini API key was rejected. "
            "Check that GEMINI_API_KEY is set correctly in your .env file."
        )

    # Gemini: quota / rate limit
    if "429" in msg and "generativelanguage" in msg:
        return "Gemini API rate limit reached. Wait a moment and try again."

    # OpenRouter: bad key or quota
    if "openrouter" in msg.lower() or ("openrouter.ai" in msg and ("401" in msg or "403" in msg)):
        return (
            "OpenRouter rejected the request. "
            "Check that OPENROUTER_API_KEY is valid and has credits."
        )

    # No results found (not really an error — surface it clearly)
    if "no relevant" in msg.lower():
        return "No matching content was found in the knowledge base. Try ingesting some files first."

    # Generic network error
    if "connection" in msg.lower() or "timeout" in msg.lower() or "connect" in msg.lower():
        return "A network connection error occurred. Check your internet connection and try again."

    # Fallback — hide raw traceback, show a clean message with a hint
    return (
        "Something went wrong while processing your request. "
        "Check the server terminal for details."
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return TEMPLATE.read_text(encoding="utf-8")


@app.get("/config")
async def config():
    return {"default_model": _default_model}


@app.post("/chat")
async def chat(req: ChatRequest):
    assert _engine is not None, "Server not initialised"
    try:
        ftype = req.filter_type if req.filter_type and req.filter_type != "all" else None
        result = await _engine.query(
            question=req.message,
            match_count=req.top_k,
            filter_type=ftype,
            model=req.model or None,
        )
        sources = [
            {"source": s, "score": round(sc, 4)}
            for s, sc in zip(result.sources, result.similarity_scores)
        ]
        return {
            "answer": result.answer,
            "sources": sources,
            "model": result.model_used,
            "usage": result.usage,
        }
    except Exception as exc:
        return JSONResponse({"answer": _friendly_error(exc), "sources": []}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
