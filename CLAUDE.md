# GEM2 RAG Framework

Multimodal RAG pipeline using Google's `gemini-embedding-2-preview` for embeddings, Supabase pgvector for storage, and OpenRouter for answer generation.

## Stack

- **Embeddings**: `gemini-embedding-2-preview` via raw REST (not SDK — preview model)
- **Vector DB**: Supabase pgvector, HNSW index, cosine similarity
- **LLM**: OpenRouter (`OPENROUTER_MODEL` env var, default `anthropic/claude-sonnet-4`)
- **Server**: FastAPI + uvicorn on port 8000
- **CLI**: `python -m src.main` via click

## How Embeddings Work

All modalities go through `src/gemini_embedder.py` → `GeminiEmbedder` (async context manager):

```text
embed_text(text, task_type)      # task_type = RETRIEVAL_DOCUMENT (ingest) or RETRIEVAL_QUERY (query)
embed_image(path)                # base64 PNG/JPEG → inlineData
embed_image_bytes(bytes, mime)   # used by video keyframe extraction
embed_video(path)                # cv2 samples frames every 5s → multi-part inlineData
embed_audio(path)                # base64 MP3/WAV/M4A/OGG → inlineData
```

- Output: `list[float]` of length `EMBEDDING_DIMENSION` (default 3072, Matryoshka 128–3072)
- **Critical**: use `taskType: RETRIEVAL_QUERY` for queries, `RETRIEVAL_DOCUMENT` for ingestion — measurably improves recall
- API endpoint: `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:embedContent`

## Vector Table Schema

Single `media_embeddings` table in Supabase (see `schema/supabase_schema.sql`):

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | PK |
| `content_type` | enum | `text`, `image`, `video`, `audio` |
| `source` | TEXT UNIQUE | upsert conflict key — format: `path::chunk_N` (text), `path` (image/audio), `path::frame_Ns` (keyframes) |
| `content_text` | TEXT | null for image/video/audio |
| `metadata` | JSONB | chunk_index, timestamp_s, width/height, mime_type, etc. |
| `embedding` | VECTOR(3072) | dimension must match `EMBEDDING_DIMENSION` in config |

Similarity search via RPC: `match_media_embeddings(query_embedding, match_threshold, match_count, filter_type)`

## RAG Flow

**Ingest**: file → ingestor → `embed_*(…, RETRIEVAL_DOCUMENT)` → `supabase_client.upsert(on_conflict="source")`

**Query**: question → `embed_text(…, RETRIEVAL_QUERY)` → `similarity_search()` via RPC → context string → OpenRouter chat completion → answer

## Key Files

```text
src/config.py              # env loading + validation; EMBEDDING_DIMENSION must match SQL schema
src/gemini_embedder.py     # all Gemini API calls — only file that touches the embedding API
src/supabase_client.py     # upsert + similarity_search via .rpc("match_media_embeddings")
src/ingestors/
  text_ingestor.py         # .txt .md .pdf .docx .doc — sliding-window chunking
  image_ingestor.py        # PNG/JPEG — resizes if >2048px before embedding
  video_ingestor.py        # strategy="whole" (1 vector) or "keyframes" (1 vector per 5s frame)
  audio_ingestor.py        # .mp3 .wav .m4a .ogg
src/retrieval/query_engine.py  # QueryEngine — ties embedder + DB + OpenRouter together
src/main.py                # CLI entry point
server.py                  # FastAPI server — GET / serves chat UI, POST /chat runs QueryEngine
templates/index.html       # single-file dark chat UI, zero dependencies
```

## Local Dev

```bash
# Run schema SQL in Supabase dashboard first (schema/supabase_schema.sql)
pip install -r requirements.txt

# Ingest data
python -m src.main ingest text  ./data/text  --recursive
python -m src.main ingest image ./data/images --recursive
python -m src.main ingest video ./data/video  --recursive --strategy keyframes
python -m src.main ingest audio ./data/audio  --recursive

# Start chat server (or use F5 in VSCode → "GEM2 RAG: Start Server")
uvicorn server:app --reload --port 8000
```

## Env Vars (`.env`)

| Var | Required | Notes |
| --- | --- | --- |
| `GEMINI_API_KEY` | Yes | Google AI Studio |
| `SUPABASE_PROJECT_URL` | Yes | `https://<ref>.supabase.co` |
| `SUPABASE_PROJECT_KEY` | Yes | service role key |
| `OPENROUTER_API_KEY` | Yes | OpenRouter |
| `OPENROUTER_MODEL` | No | default `anthropic/claude-sonnet-4` |
| `EMBEDDING_DIMENSION` | No | default `3072` — must match SQL `VECTOR(N)` |

## Constraints

- `EMBEDDING_DIMENSION` change requires re-running `schema/supabase_schema.sql` and re-ingesting all data — `gemini-embedding-2-preview` embeddings are incompatible across dimension settings
- `on_conflict="source"` makes re-ingestion idempotent — safe to re-run ingestors
- `opencv-python-headless` requires `ffmpeg` on system PATH for video decoding

## Applied Learning

When something fails repeatedly, when I have to re-explain, or when a workaround is found for a platform/tool limitation, add a one line bullet here. Keep each bullet under 15 words. No explanations. Only add things taht will save time in the future sessions.
