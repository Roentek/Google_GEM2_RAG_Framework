from dataclasses import dataclass, field

import httpx

from src.config import Config
from src.gemini_embedder import GeminiEmbedder
from src.supabase_client import SupabaseVectorClient


@dataclass
class QueryResult:
    answer: str
    sources: list[str]
    similarity_scores: list[float]
    model_used: str = ""
    usage: dict = field(default_factory=dict)   # prompt_tokens, completion_tokens, total_tokens
    raw_context: list[dict] = field(default_factory=list)


class QueryEngine:
    def __init__(self, embedder: GeminiEmbedder, db: SupabaseVectorClient, config: Config):
        self._embedder = embedder
        self._db = db
        self._config = config

    async def query(
        self,
        question: str,
        match_threshold: float = 0.5,
        match_count: int = 5,
        filter_type: str | None = None,
        model: str | None = None,
    ) -> QueryResult:
        # 1. Embed the query (RETRIEVAL_QUERY task type — different from ingestion)
        query_embedding = await self._embedder.embed_text(
            question, task_type="RETRIEVAL_QUERY"
        )

        # 2. Retrieve similar documents from Supabase
        results = self._db.similarity_search(
            query_embedding,
            match_threshold=match_threshold,
            match_count=match_count,
            filter_type=filter_type,
        )

        if not results:
            return QueryResult(
                answer="No relevant content found in the knowledge base for your query.",
                sources=[],
                similarity_scores=[],
                model_used=model or self._config.OPENROUTER_MODEL,
                usage={},
            )

        # 3. Build context string
        context_parts = []
        for i, row in enumerate(results, 1):
            ctype = row.get("content_type", "unknown")
            source = row.get("source", "")
            score = row.get("similarity", 0.0)

            if ctype == "text" and row.get("content_text"):
                context_parts.append(
                    f"[{i}] (text, similarity={score:.3f}, source={source})\n{row['content_text']}"
                )
            elif ctype == "image":
                meta = row.get("metadata", {})
                desc = f"Image file: {source}"
                if "width" in meta:
                    desc += f" ({meta['width']}x{meta['height']}px)"
                context_parts.append(f"[{i}] (image, similarity={score:.3f}) {desc}")
            elif ctype == "video":
                meta = row.get("metadata", {})
                desc = f"Video file: {source}"
                if "timestamp_s" in meta:
                    desc += f" at {meta['timestamp_s']}s"
                context_parts.append(f"[{i}] (video, similarity={score:.3f}) {desc}")
            else:
                context_parts.append(f"[{i}] (source={source}, similarity={score:.3f})")

        context = "\n\n".join(context_parts)

        # 4. Call OpenRouter for answer generation
        answer, model_used, usage = await self._call_openrouter(question, context, model)

        return QueryResult(
            answer=answer,
            sources=[r.get("source", "") for r in results],
            similarity_scores=[r.get("similarity", 0.0) for r in results],
            model_used=model_used,
            usage=usage,
            raw_context=results,
        )

    async def _call_openrouter(
        self, question: str, context: str, model: str | None = None
    ) -> tuple[str, str, dict]:
        selected_model = model or self._config.OPENROUTER_MODEL
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question using only "
                    "the provided context. If the context does not contain enough information "
                    "to answer, say so clearly. Cite the source numbers [1], [2], etc. when referencing specific content."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._config.OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._config.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": selected_model,
                    "messages": messages,
                },
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter API error {response.status_code}: {response.text}"
            )

        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        model_used = data.get("model", selected_model)
        usage = data.get("usage", {})

        return answer, model_used, usage
