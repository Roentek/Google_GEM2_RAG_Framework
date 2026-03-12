import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # -------------------------------------------------------------------------
    # Required API keys — raises EnvironmentError if missing or empty
    # -------------------------------------------------------------------------
    GEMINI_API_KEY: str
    SUPABASE_URL: str
    SUPABASE_KEY: str
    OPENROUTER_API_KEY: str

    # -------------------------------------------------------------------------
    # Gemini embedding settings
    # -------------------------------------------------------------------------
    GEMINI_EMBED_MODEL: str = "gemini-embedding-2-preview"
    GEMINI_EMBED_URL: str = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-embedding-2-preview:embedContent"
    )

    # Matryoshka dimension: 128–3072 (default: full 3072)
    # Must match the VECTOR(N) dimension in supabase_schema.sql
    EMBEDDING_DIMENSION: int = 3072

    # -------------------------------------------------------------------------
    # OpenRouter / LLM settings
    # -------------------------------------------------------------------------
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "anthropic/claude-sonnet-4"

    # -------------------------------------------------------------------------
    # Text chunking
    # -------------------------------------------------------------------------
    TEXT_CHUNK_SIZE: int = 2000     # characters per chunk (well under 8192 token limit)
    TEXT_CHUNK_OVERLAP: int = 200   # character overlap between consecutive chunks

    # -------------------------------------------------------------------------
    # Media ingestion limits (enforced by Gemini API)
    # -------------------------------------------------------------------------
    MAX_IMAGE_COUNT: int = 6
    MAX_VIDEO_DURATION_S: int = 128
    MAX_AUDIO_DURATION_S: int = 80
    MAX_IMAGE_PX: int = 2048        # resize threshold to reduce base64 payload

    def __init__(self):
        self.GEMINI_API_KEY = self._require("GEMINI_API_KEY")
        self.SUPABASE_URL = self._require("SUPABASE_PROJECT_URL")
        self.SUPABASE_KEY = self._require("SUPABASE_PROJECT_KEY")
        self.OPENROUTER_API_KEY = self._require("OPENROUTER_API_KEY")

        self.OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", self.OPENROUTER_MODEL)

        dim = int(os.getenv("EMBEDDING_DIMENSION", str(self.EMBEDDING_DIMENSION)))
        if not (128 <= dim <= 3072):
            raise ValueError(
                f"EMBEDDING_DIMENSION must be between 128 and 3072, got {dim}"
            )
        self.EMBEDDING_DIMENSION = dim

    @staticmethod
    def _require(key: str) -> str:
        value = os.getenv(key, "").strip()
        if not value:
            raise EnvironmentError(
                f"Required environment variable '{key}' is missing or empty. "
                "Copy .env.example to .env and fill in your values."
            )
        return value
