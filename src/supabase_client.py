from supabase import create_client, Client

from src.config import Config


class SupabaseVectorClient:
    def __init__(self, config: Config):
        self._client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

    def upsert(
        self,
        content_type: str,
        source: str,
        content_text: str | None,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> dict:
        row = {
            "content_type": content_type,
            "source": source,
            "content_text": content_text,
            "embedding": embedding,
            "metadata": metadata or {},
        }
        result = (
            self._client.table("media_embeddings")
            .upsert(row, on_conflict="source")
            .execute()
        )
        return result.data[0] if result.data else {}

    def similarity_search(
        self,
        query_embedding: list[float],
        match_threshold: float = 0.5,
        match_count: int = 10,
        filter_type: str | None = None,
    ) -> list[dict]:
        params = {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count": match_count,
            "filter_type": filter_type,
        }
        result = self._client.rpc("match_media_embeddings", params).execute()
        return result.data or []

    def delete_by_source(self, source: str) -> None:
        self._client.table("media_embeddings").delete().eq("source", source).execute()

    def get_by_id(self, id: str) -> dict | None:
        result = (
            self._client.table("media_embeddings")
            .select("*")
            .eq("id", id)
            .single()
            .execute()
        )
        return result.data
