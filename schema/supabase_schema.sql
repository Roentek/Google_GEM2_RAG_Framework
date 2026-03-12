-- =============================================================================
-- Gemini Embedding 2 Multimodal RAG - Supabase Schema
-- Run this entire file in the Supabase SQL editor before using the Python app.
-- =============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Content type enum
CREATE TYPE media_content_type AS ENUM ('text', 'image', 'video', 'audio');

-- =============================================================================
-- Main embeddings table
-- =============================================================================
CREATE TABLE IF NOT EXISTS media_embeddings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Modality of the embedded content
    content_type    media_content_type NOT NULL,

    -- Human-readable source reference used as the upsert conflict key.
    -- For text chunks: "{file_path}::chunk_{index}"
    -- For images:      "{file_path}"
    -- For video whole: "{file_path}"
    -- For keyframes:   "{file_path}::frame_{timestamp_s}s"
    source          TEXT NOT NULL UNIQUE,

    -- The raw text payload that was embedded.
    -- NULL for image/video rows where no text exists.
    content_text    TEXT,

    -- Flexible metadata: chunk_index, total_chunks, timestamp_s, mime_type,
    -- original_filename, width, height, video_source, custom tags, etc.
    metadata        JSONB DEFAULT '{}'::jsonb,

    -- The embedding vector. Dimension must match EMBEDDING_DIMENSION in config.py.
    -- Default: 3072 (gemini-embedding-2-preview default via Matryoshka).
    embedding       VECTOR(3072) NOT NULL,

    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- HNSW index for approximate nearest-neighbor cosine similarity search.
-- m=16, ef_construction=64 are solid defaults.
-- Increase ef_construction (e.g. 128) for better recall at cost of build time.
CREATE INDEX IF NOT EXISTS media_embeddings_embedding_idx
    ON media_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Index for fast content_type filtering before ANN
CREATE INDEX IF NOT EXISTS media_embeddings_content_type_idx
    ON media_embeddings (content_type);

-- =============================================================================
-- Auto-update trigger for updated_at
-- =============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_updated_at ON media_embeddings;

CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON media_embeddings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Similarity search RPC function
-- Called from Python via: supabase.rpc("match_media_embeddings", {...})
--
-- Parameters:
--   query_embedding  : the vectorized query (same dimension as stored embeddings)
--   match_threshold  : minimum cosine similarity score to include (0.0 – 1.0)
--   match_count      : maximum number of rows to return
--   filter_type      : optional content_type filter; NULL returns all types
-- =============================================================================
CREATE OR REPLACE FUNCTION match_media_embeddings(
    query_embedding  VECTOR(3072),
    match_threshold  FLOAT    DEFAULT 0.5,
    match_count      INT      DEFAULT 10,
    filter_type      TEXT     DEFAULT NULL
)
RETURNS TABLE (
    id              UUID,
    content_type    media_content_type,
    source          TEXT,
    content_text    TEXT,
    metadata        JSONB,
    similarity      FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        me.id,
        me.content_type,
        me.source,
        me.content_text,
        me.metadata,
        1 - (me.embedding <=> query_embedding) AS similarity
    FROM media_embeddings me
    WHERE
        (filter_type IS NULL OR me.content_type::TEXT = filter_type)
        AND 1 - (me.embedding <=> query_embedding) >= match_threshold
    ORDER BY me.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
