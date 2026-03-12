from pathlib import Path

from src.config import Config
from src.ingestors.base_ingestor import BaseIngestor, IngestResult


def _read_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required for PDF ingestion. Run: pip install pypdf")
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext in {".docx", ".doc"}:
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx is required for Word document ingestion. Run: pip install python-docx")
        doc = Document(str(path))
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    return path.read_text(encoding="utf-8", errors="replace")


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


class TextIngestor(BaseIngestor):
    SUPPORTED = {".txt", ".md", ".pdf", ".docx", ".doc"}

    def __init__(self, embedder, db, config: Config):
        super().__init__(embedder, db)
        self._config = config

    async def ingest(self, path: str, metadata: dict | None = None) -> IngestResult:
        p = Path(path)
        if p.suffix.lower() not in self.SUPPORTED:
            raise ValueError(f"Unsupported text format '{p.suffix}'. Supported: {self.SUPPORTED}")

        text = _read_text(p)
        chunks = _chunk_text(text, self._config.TEXT_CHUNK_SIZE, self._config.TEXT_CHUNK_OVERLAP)
        total = len(chunks)
        ingested = 0
        errors = []

        for i, chunk in enumerate(chunks):
            source_key = f"{path}::chunk_{i}"
            try:
                embedding = await self.embedder.embed_text(chunk, task_type="RETRIEVAL_DOCUMENT")
                self.db.upsert(
                    content_type="text",
                    source=source_key,
                    content_text=chunk,
                    embedding=embedding,
                    metadata={**(metadata or {}), "chunk_index": i, "total_chunks": total, "file": str(p.name)},
                )
                ingested += 1
            except Exception as e:
                errors.append(f"chunk {i}: {e}")

        return IngestResult(source=str(path), content_type="text", chunks_ingested=ingested, errors=errors)
