# Google Gemini Multimodal RAG Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready RAG (Retrieval-Augmented Generation) pipeline using Google's **Gemini Embedding 2** model — the first fully multimodal embedding model — to store and query embeddings across text, images, video, and audio. Vectors are stored in **Supabase pgvector** and answers are generated via **OpenRouter** (defaults to Claude Sonnet 4).

Includes a dark-theme chat web UI, a CLI ingestor, automatic data-folder sync, and real-time ingestion notifications.

---

## Stack

| Layer | Technology |
| --- | --- |
| Embeddings | `gemini-embedding-2-preview` (REST API, 3072-dim) |
| Vector DB | Supabase pgvector, HNSW index, cosine similarity |
| LLM | OpenRouter (default: `anthropic/claude-sonnet-4-5`) |
| Server | FastAPI + uvicorn |
| File watcher | watchfiles (startup sync + live watch) |

---

## Prerequisites

- Python 3.11+
- [Supabase](https://supabase.com) project (free tier works)
- [Google AI Studio](https://aistudio.google.com) API key (Gemini)
- [OpenRouter](https://openrouter.ai) API key
- `ffmpeg` on system PATH *(only required for video keyframe strategy)*

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd Google_GEM2_RAG_Framework
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
SUPABASE_PROJECT_URL=https://your-project.supabase.co
SUPABASE_PROJECT_KEY=your-service-role-key
GEMINI_API_KEY=your-google-ai-studio-key
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=anthropic/claude-sonnet-4-5
```

> **Never commit `.env`** — it is gitignored.

### 3. Apply the Supabase schema

In your [Supabase SQL Editor](https://supabase.com/dashboard), open and run:

```text
schema/supabase_schema.sql
```

This creates the `media_embeddings` table, HNSW vector index, and the `match_media_embeddings` RPC function used for similarity search.

---

## Running the server

```bash
python server.py
```

Then open **<http://localhost:8000>** in your browser.

The server starts a background task that:

1. **Startup sync** — diffs `data/` against the DB and ingests new files, re-ingests modified files, and removes embeddings for deleted files
2. **Live watch** — continues watching `data/` for any changes while the server is running

The chat UI shows a real-time status dot and toast notifications for all sync activity.

> **VS Code**: use the included launch configurations (`Run and Debug → GEM2 RAG Start Server`).

---

## Adding content

Drop files into the appropriate subfolder under `data/` — the watcher picks them up automatically:

```text
data/
├── text/      → .txt  .md  .pdf  .docx  .doc
├── images/    → .png  .jpg  .jpeg
├── video/     → .mp4  .mov  .avi  .mkv  .webm
└── audio/     → .mp3  .wav  .m4a  .ogg
```

Files are processed within ~2 seconds of being added, modified, or deleted.

---

## CLI ingestor

For bulk ingestion or scripted workflows, use the CLI directly:

```bash
# Ingest a single file
python -m src.main ingest text   data/text/report.pdf
python -m src.main ingest image  data/images/photo.jpg
python -m src.main ingest video  data/video/clip.mp4 --strategy whole
python -m src.main ingest audio  data/audio/meeting.mp3

# Recursive directory ingestion
python -m src.main ingest text data/text/ --recursive

# Query without the web UI
python -m src.main query "What does the report say about Q3?" --top-k 5
python -m src.main query "Describe the image" --filter-type image
```

**Video strategies:**

- `whole` *(default)* — embeds the entire video as one unit (≤128 s)
- `keyframes` — extracts one frame every 5 seconds and embeds each independently

---

## Architecture

```text
User query
    │
    ▼
GeminiEmbedder.embed_text(taskType=RETRIEVAL_QUERY)
    │
    ▼
SupabaseVectorClient.similarity_search()   ← cosine similarity via HNSW
    │
    ▼  (top-K chunks / images / frames)
QueryEngine._call_openrouter()             ← context-stuffed prompt
    │
    ▼
Answer + sources + token usage
```

**Ingestion flow:**

```text
File on disk
    │
    ├─ text  → chunk (2000 chars, 200 overlap) → embed each chunk
    ├─ image → resize if >2048px → embed whole image
    ├─ video → embed whole (≤128 s) OR extract keyframes → embed each
    └─ audio → embed whole file (≤80 s)
    │
    ▼
Supabase upsert (on_conflict="source") — idempotent re-ingestion
```

All content types share one `media_embeddings` table. Source keys use `path::chunk_N` / `path::frame_Ns` for multi-row content, making prefix-based deletion clean.

---

## Environment variable reference

| Variable | Required | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | ✓ | Google AI Studio key for embeddings |
| `SUPABASE_PROJECT_URL` | ✓ | e.g. `https://xyz.supabase.co` |
| `SUPABASE_PROJECT_KEY` | ✓ | Service role key from Supabase dashboard |
| `OPENROUTER_API_KEY` | ✓ | OpenRouter key for LLM answer generation |
| `OPENROUTER_MODEL` | — | Default `anthropic/claude-sonnet-4-5` |
| `EMBEDDING_DIMENSION` | — | 128–3072, default `3072` (Matryoshka) |

> `EMBEDDING_DIMENSION` must match the `VECTOR(N)` size in `supabase_schema.sql`. Change both together if you want a smaller index.

---

## Project structure

```text
├── server.py                   # FastAPI server + SSE broadcast
├── schema/
│   └── supabase_schema.sql     # Run once in Supabase SQL editor
├── src/
│   ├── config.py               # Env loading + validation
│   ├── gemini_embedder.py      # Async Gemini REST client (text/image/video/audio)
│   ├── supabase_client.py      # Upsert, similarity search, delete helpers
│   ├── watcher.py              # Startup sync + live watchfiles loop
│   ├── main.py                 # Click CLI
│   ├── ingestors/
│   │   ├── text_ingestor.py    # .txt .md .pdf .docx .doc
│   │   ├── image_ingestor.py   # .png .jpg .jpeg
│   │   ├── video_ingestor.py   # .mp4 .mov (whole or keyframes)
│   │   └── audio_ingestor.py   # .mp3 .wav .m4a .ogg
│   └── retrieval/
│       └── query_engine.py     # Embed query → search → LLM → answer
├── templates/
│   └── index.html              # Dark-theme chat UI
├── static/                     # Favicon + static assets
├── data/                       # Drop your files here
│   ├── text/
│   ├── images/
│   ├── video/
│   └── audio/
├── .env.example
└── requirements.txt
```

---

## Troubleshooting

| Error | Fix |
| --- | --- |
| `match_media_embeddings` not found | Run `schema/supabase_schema.sql` in Supabase SQL editor |
| `GEMINI_API_KEY` rejected | Verify key in Google AI Studio; ensure Gemini API is enabled |
| OpenRouter 401/403 | Check `OPENROUTER_API_KEY` has credits |
| No results returned | Ingest at least one file; check `data/` folder and server logs |
| Video ingestion fails | Ensure `ffmpeg` is on system PATH for keyframe strategy |

---

## License

MIT License - Copyright (c) 2026 Roentek Designs

See [LICENSE](LICENSE) for details.
