from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.gemini_embedder import GeminiEmbedder
from src.supabase_client import SupabaseVectorClient


@dataclass
class IngestResult:
    source: str
    content_type: str
    chunks_ingested: int
    errors: list[str] = field(default_factory=list)


class BaseIngestor(ABC):
    def __init__(self, embedder: GeminiEmbedder, db: SupabaseVectorClient):
        self.embedder = embedder
        self.db = db

    @abstractmethod
    async def ingest(self, path: str, metadata: dict | None = None) -> IngestResult:
        ...
