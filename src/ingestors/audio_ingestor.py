from pathlib import Path

from src.config import Config
from src.ingestors.base_ingestor import BaseIngestor, IngestResult

SUPPORTED = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".ogg": "audio/ogg"}


class AudioIngestor(BaseIngestor):
    def __init__(self, embedder, db, config: Config):
        super().__init__(embedder, db)
        self._config = config

    async def ingest(self, path: str, metadata: dict | None = None) -> IngestResult:
        p = Path(path)
        ext = p.suffix.lower()
        if ext not in SUPPORTED:
            raise ValueError(f"Unsupported audio format '{ext}'. Supported: {list(SUPPORTED)}")

        mime = SUPPORTED[ext]
        size_mb = p.stat().st_size / (1024 * 1024)

        try:
            embedding = await self.embedder.embed_audio(p)
            self.db.upsert(
                content_type="audio",
                source=str(path),
                content_text=None,
                embedding=embedding,
                metadata={
                    **(metadata or {}),
                    "mime_type": mime,
                    "file": p.name,
                    "size_mb": round(size_mb, 2),
                },
            )
            return IngestResult(source=str(path), content_type="audio", chunks_ingested=1)
        except Exception as e:
            return IngestResult(source=str(path), content_type="audio", chunks_ingested=0, errors=[str(e)])
