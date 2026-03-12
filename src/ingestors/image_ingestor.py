import io
from pathlib import Path

from src.config import Config
from src.ingestors.base_ingestor import BaseIngestor, IngestResult

SUPPORTED = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


class ImageIngestor(BaseIngestor):
    def __init__(self, embedder, db, config: Config):
        super().__init__(embedder, db)
        self._config = config

    async def ingest(self, path: str, metadata: dict | None = None) -> IngestResult:
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for image ingestion. Run: pip install Pillow")

        p = Path(path)
        ext = p.suffix.lower()
        if ext not in SUPPORTED:
            raise ValueError(f"Unsupported image format '{ext}'. Supported: {list(SUPPORTED)}")

        img = Image.open(p)
        w, h = img.size
        mime = SUPPORTED[ext]

        # Resize if either dimension exceeds MAX_IMAGE_PX to reduce payload size
        max_px = self._config.MAX_IMAGE_PX
        if w > max_px or h > max_px:
            img.thumbnail((max_px, max_px), Image.LANCZOS)
            w, h = img.size
            # Re-encode resized image to bytes
            buf = io.BytesIO()
            fmt = "PNG" if ext == ".png" else "JPEG"
            img.save(buf, format=fmt)
            image_bytes = buf.getvalue()
        else:
            image_bytes = p.read_bytes()

        try:
            embedding = await self.embedder.embed_image_bytes(image_bytes, mime=mime)
            self.db.upsert(
                content_type="image",
                source=str(path),
                content_text=None,
                embedding=embedding,
                metadata={
                    **(metadata or {}),
                    "mime_type": mime,
                    "width": w,
                    "height": h,
                    "file": p.name,
                },
            )
            return IngestResult(source=str(path), content_type="image", chunks_ingested=1)
        except Exception as e:
            return IngestResult(source=str(path), content_type="image", chunks_ingested=0, errors=[str(e)])
