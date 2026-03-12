import base64
from pathlib import Path

from src.config import Config
from src.ingestors.base_ingestor import BaseIngestor, IngestResult

SUPPORTED = {".mp4", ".mov"}
KEYFRAME_INTERVAL_S = 5


class VideoIngestor(BaseIngestor):
    def __init__(self, embedder, db, config: Config):
        super().__init__(embedder, db)
        self._config = config

    async def ingest(
        self,
        path: str,
        metadata: dict | None = None,
        strategy: str = "whole",
    ) -> IngestResult:
        p = Path(path)
        if p.suffix.lower() not in SUPPORTED:
            raise ValueError(f"Unsupported video format '{p.suffix}'. Supported: {SUPPORTED}")
        if strategy not in ("whole", "keyframes"):
            raise ValueError("strategy must be 'whole' or 'keyframes'")

        if strategy == "whole":
            return await self._ingest_whole(p, metadata or {})
        return await self._ingest_keyframes(p, metadata or {})

    async def _ingest_whole(self, p: Path, metadata: dict) -> IngestResult:
        try:
            embedding = await self.embedder.embed_video(p)
            self.db.upsert(
                content_type="video",
                source=str(p),
                content_text=None,
                embedding=embedding,
                metadata={**metadata, "file": p.name, "strategy": "whole"},
            )
            return IngestResult(source=str(p), content_type="video", chunks_ingested=1)
        except Exception as e:
            return IngestResult(source=str(p), content_type="video", chunks_ingested=0, errors=[str(e)])

    async def _ingest_keyframes(self, p: Path, metadata: dict) -> IngestResult:
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python-headless is required for video ingestion. Run: pip install opencv-python-headless")

        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps

        ingested = 0
        errors = []
        t = 0.0

        while t < min(duration_s, self._config.MAX_VIDEO_DURATION_S):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret:
                _, buf = cv2.imencode(".jpg", frame)
                frame_bytes = buf.tobytes()
                source_key = f"{p}::frame_{int(t)}s"
                try:
                    embedding = await self.embedder.embed_image_bytes(frame_bytes, mime="image/jpeg")
                    self.db.upsert(
                        content_type="image",
                        source=source_key,
                        content_text=None,
                        embedding=embedding,
                        metadata={
                            **metadata,
                            "video_source": str(p),
                            "timestamp_s": t,
                            "strategy": "keyframes",
                            "file": p.name,
                        },
                    )
                    ingested += 1
                except Exception as e:
                    errors.append(f"frame at {t}s: {e}")
            t += KEYFRAME_INTERVAL_S

        cap.release()
        return IngestResult(source=str(p), content_type="video", chunks_ingested=ingested, errors=errors)
