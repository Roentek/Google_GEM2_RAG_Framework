import base64
import mimetypes
from pathlib import Path

import httpx

from src.config import Config


SUPPORTED_IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
SUPPORTED_VIDEO_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime"}
SUPPORTED_AUDIO_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".ogg": "audio/ogg"}


class GeminiEmbedError(Exception):
    pass


class GeminiEmbedder:
    def __init__(self, config: Config):
        self._config = config
        self._client = httpx.AsyncClient(
            headers={"Content-Type": "application/json"},
            params={"key": config.GEMINI_API_KEY},
            timeout=120.0,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

    async def close(self):
        await self._client.aclose()

    # -------------------------------------------------------------------------
    # Public embedding methods
    # -------------------------------------------------------------------------

    async def embed_text(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        if not text or not text.strip():
            raise ValueError("text must be non-empty")
        body = {
            "model": f"models/{self._config.GEMINI_EMBED_MODEL}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
            "outputDimensionality": self._config.EMBEDDING_DIMENSION,
        }
        return await self._post(body)

    async def embed_image(self, image_path: str | Path) -> list[float]:
        path = Path(image_path)
        ext = path.suffix.lower()
        if ext not in SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"Unsupported image format '{ext}'. Supported: {list(SUPPORTED_IMAGE_TYPES)}")
        mime = SUPPORTED_IMAGE_TYPES[ext]
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        body = {
            "model": f"models/{self._config.GEMINI_EMBED_MODEL}",
            "content": {"parts": [{"inlineData": {"mimeType": mime, "data": data}}]},
            "taskType": "RETRIEVAL_DOCUMENT",
            "outputDimensionality": self._config.EMBEDDING_DIMENSION,
        }
        return await self._post(body)

    async def embed_video(self, video_path: str | Path) -> list[float]:
        path = Path(video_path)
        ext = path.suffix.lower()
        if ext not in SUPPORTED_VIDEO_TYPES:
            raise ValueError(f"Unsupported video format '{ext}'. Supported: {list(SUPPORTED_VIDEO_TYPES)}")

        parts = self._extract_video_parts(path)
        body = {
            "model": f"models/{self._config.GEMINI_EMBED_MODEL}",
            "content": {"parts": parts},
            "taskType": "RETRIEVAL_DOCUMENT",
            "outputDimensionality": self._config.EMBEDDING_DIMENSION,
        }
        return await self._post(body)

    async def embed_image_bytes(self, image_bytes: bytes, mime: str = "image/jpeg") -> list[float]:
        data = base64.b64encode(image_bytes).decode("utf-8")
        body = {
            "model": f"models/{self._config.GEMINI_EMBED_MODEL}",
            "content": {"parts": [{"inlineData": {"mimeType": mime, "data": data}}]},
            "taskType": "RETRIEVAL_DOCUMENT",
            "outputDimensionality": self._config.EMBEDDING_DIMENSION,
        }
        return await self._post(body)

    async def embed_audio(self, audio_path: str | Path) -> list[float]:
        path = Path(audio_path)
        ext = path.suffix.lower()
        if ext not in SUPPORTED_AUDIO_TYPES:
            raise ValueError(f"Unsupported audio format '{ext}'. Supported: {list(SUPPORTED_AUDIO_TYPES)}")
        mime = SUPPORTED_AUDIO_TYPES[ext]
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        body = {
            "model": f"models/{self._config.GEMINI_EMBED_MODEL}",
            "content": {"parts": [{"inlineData": {"mimeType": mime, "data": data}}]},
            "taskType": "RETRIEVAL_DOCUMENT",
            "outputDimensionality": self._config.EMBEDDING_DIMENSION,
        }
        return await self._post(body)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _extract_video_parts(self, path: Path) -> list[dict]:
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python-headless is required for video embedding. Run: pip install opencv-python-headless")

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps

        if duration_s > self._config.MAX_VIDEO_DURATION_S:
            raise ValueError(
                f"Video duration {duration_s:.1f}s exceeds maximum "
                f"{self._config.MAX_VIDEO_DURATION_S}s allowed by Gemini API"
            )

        # Sample one frame every 5 seconds
        sample_interval_s = 5
        parts = []
        sampled_times = []
        t = 0.0
        while t < min(duration_s, self._config.MAX_VIDEO_DURATION_S):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret:
                _, buf = cv2.imencode(".jpg", frame)
                data = base64.b64encode(buf.tobytes()).decode("utf-8")
                parts.append({"inlineData": {"mimeType": "image/jpeg", "data": data}})
                sampled_times.append(t)
            t += sample_interval_s

        cap.release()

        if not parts:
            raise ValueError(f"Could not extract any frames from {path}")

        return parts

    async def _post(self, body: dict) -> list[float]:
        response = await self._client.post(self._config.GEMINI_EMBED_URL, json=body)
        if response.status_code != 200:
            raise GeminiEmbedError(
                f"Gemini API error {response.status_code}: {response.text}"
            )
        data = response.json()
        return data["embedding"]["values"]
