"""GEM2 RAG — lightweight FastAPI chat server."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _engine
    cfg = Config()
    _embedder = GeminiEmbedder(cfg)
    await _embedder.__aenter__()
    db = SupabaseVectorClient(cfg)
    _engine = QueryEngine(_embedder, db, cfg)
    yield
    if _embedder:
        await _embedder.close()


app = FastAPI(title="GEM2 RAG", lifespan=lifespan)

TEMPLATE = Path(__file__).parent / "templates" / "index.html"


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    filter_type: str | None = None
    top_k: int = 5


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return TEMPLATE.read_text(encoding="utf-8")


@app.post("/chat")
async def chat(req: ChatRequest):
    assert _engine is not None, "Server not initialised"
    try:
        ftype = req.filter_type if req.filter_type and req.filter_type != "all" else None
        result = await _engine.query(
            question=req.message,
            match_count=req.top_k,
            filter_type=ftype,
        )
        sources = [
            {"source": s, "score": round(sc, 4)}
            for s, sc in zip(result.sources, result.similarity_scores)
        ]
        return {"answer": result.answer, "sources": sources}
    except Exception as exc:
        return JSONResponse({"answer": f"Error: {exc}", "sources": []}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
