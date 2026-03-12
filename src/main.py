import asyncio
import json
from pathlib import Path

import click

from src.config import Config
from src.gemini_embedder import GeminiEmbedder
from src.supabase_client import SupabaseVectorClient


def _get_files(path: str, extensions: set[str], recursive: bool) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p] if p.suffix.lower() in extensions else []
    if not p.is_dir():
        raise click.BadParameter(f"Path does not exist: {path}")
    pattern = "**/*" if recursive else "*"
    return [f for f in p.glob(pattern) if f.is_file() and f.suffix.lower() in extensions]


# =============================================================================
# CLI root
# =============================================================================

@click.group()
def cli():
    """Gemini Embedding 2 multimodal RAG framework."""


# =============================================================================
# ingest group
# =============================================================================

@cli.group()
def ingest():
    """Ingest content into the vector database."""


@ingest.command("text")
@click.argument("path")
@click.option("--recursive", is_flag=True, default=False, help="Recurse into directories.")
@click.option("--metadata-json", default="{}", help="Extra metadata as a JSON string.")
def ingest_text(path: str, recursive: bool, metadata_json: str):
    """Ingest text files (.txt, .md, .pdf, .docx, .doc)."""
    from src.ingestors.text_ingestor import TextIngestor

    metadata = json.loads(metadata_json)
    config = Config()
    db = SupabaseVectorClient(config)
    files = _get_files(path, {".txt", ".md", ".pdf", ".docx", ".doc"}, recursive)

    if not files:
        click.echo(f"No supported text files found at: {path}")
        return

    async def _run():
        async with GeminiEmbedder(config) as embedder:
            ingestor = TextIngestor(embedder, db, config)
            for f in files:
                click.echo(f"Ingesting {f} ...")
                result = await ingestor.ingest(str(f), metadata)
                click.echo(f"  -> {result.chunks_ingested} chunks ingested")
                for err in result.errors:
                    click.echo(f"  [ERROR] {err}", err=True)

    asyncio.run(_run())


@ingest.command("image")
@click.argument("path")
@click.option("--recursive", is_flag=True, default=False, help="Recurse into directories.")
@click.option("--metadata-json", default="{}", help="Extra metadata as a JSON string.")
def ingest_image(path: str, recursive: bool, metadata_json: str):
    """Ingest image files (.png, .jpg, .jpeg)."""
    from src.ingestors.image_ingestor import ImageIngestor

    metadata = json.loads(metadata_json)
    config = Config()
    db = SupabaseVectorClient(config)
    files = _get_files(path, {".png", ".jpg", ".jpeg"}, recursive)

    if not files:
        click.echo(f"No supported image files found at: {path}")
        return

    async def _run():
        async with GeminiEmbedder(config) as embedder:
            ingestor = ImageIngestor(embedder, db, config)
            for f in files:
                click.echo(f"Ingesting {f} ...")
                result = await ingestor.ingest(str(f), metadata)
                click.echo(f"  -> {result.chunks_ingested} item(s) ingested")
                for err in result.errors:
                    click.echo(f"  [ERROR] {err}", err=True)

    asyncio.run(_run())


@ingest.command("audio")
@click.argument("path")
@click.option("--recursive", is_flag=True, default=False, help="Recurse into directories.")
@click.option("--metadata-json", default="{}", help="Extra metadata as a JSON string.")
def ingest_audio(path: str, recursive: bool, metadata_json: str):
    """Ingest audio files (.mp3, .wav, .m4a, .ogg)."""
    from src.ingestors.audio_ingestor import AudioIngestor

    metadata = json.loads(metadata_json)
    config = Config()
    db = SupabaseVectorClient(config)
    files = _get_files(path, {".mp3", ".wav", ".m4a", ".ogg"}, recursive)

    if not files:
        click.echo(f"No supported audio files found at: {path}")
        return

    async def _run():
        async with GeminiEmbedder(config) as embedder:
            ingestor = AudioIngestor(embedder, db, config)
            for f in files:
                click.echo(f"Ingesting {f} ...")
                result = await ingestor.ingest(str(f), metadata)
                click.echo(f"  -> {result.chunks_ingested} item(s) ingested")
                for err in result.errors:
                    click.echo(f"  [ERROR] {err}", err=True)

    asyncio.run(_run())


@ingest.command("video")
@click.argument("path")
@click.option("--recursive", is_flag=True, default=False, help="Recurse into directories.")
@click.option(
    "--strategy",
    type=click.Choice(["whole", "keyframes"]),
    default="whole",
    show_default=True,
    help="'whole' embeds the full video as one vector. 'keyframes' embeds one frame per 5s independently.",
)
@click.option("--metadata-json", default="{}", help="Extra metadata as a JSON string.")
def ingest_video(path: str, recursive: bool, strategy: str, metadata_json: str):
    """Ingest video files (.mp4, .mov)."""
    from src.ingestors.video_ingestor import VideoIngestor

    metadata = json.loads(metadata_json)
    config = Config()
    db = SupabaseVectorClient(config)
    files = _get_files(path, {".mp4", ".mov"}, recursive)

    if not files:
        click.echo(f"No supported video files found at: {path}")
        return

    async def _run():
        async with GeminiEmbedder(config) as embedder:
            ingestor = VideoIngestor(embedder, db, config)
            for f in files:
                click.echo(f"Ingesting {f} (strategy={strategy}) ...")
                result = await ingestor.ingest(str(f), metadata, strategy=strategy)
                click.echo(f"  -> {result.chunks_ingested} item(s) ingested")
                for err in result.errors:
                    click.echo(f"  [ERROR] {err}", err=True)

    asyncio.run(_run())


# =============================================================================
# query command
# =============================================================================

@cli.command("query")
@click.argument("question")
@click.option(
    "--filter-type",
    type=click.Choice(["text", "image", "video", "audio"]),
    default=None,
    help="Restrict search to a specific content type.",
)
@click.option("--top-k", default=5, show_default=True, help="Number of results to retrieve.")
@click.option("--threshold", default=0.5, show_default=True, help="Minimum similarity score (0.0–1.0).")
def query(question: str, filter_type: str | None, top_k: int, threshold: float):
    """Ask a question against the vector database."""
    from src.retrieval.query_engine import QueryEngine

    config = Config()
    db = SupabaseVectorClient(config)

    async def _run():
        async with GeminiEmbedder(config) as embedder:
            engine = QueryEngine(embedder, db, config)
            result = await engine.query(
                question,
                match_threshold=threshold,
                match_count=top_k,
                filter_type=filter_type,
            )
            click.echo("\n--- Answer ---")
            click.echo(result.answer)
            click.echo("\n--- Sources ---")
            for src, score in zip(result.sources, result.similarity_scores):
                click.echo(f"  [{score:.3f}] {src}")

    asyncio.run(_run())


# =============================================================================
# schema command
# =============================================================================

@cli.command("schema")
def schema():
    """Print the Supabase SQL schema to stdout."""
    schema_path = Path(__file__).parent.parent / "schema" / "supabase_schema.sql"
    click.echo(schema_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    cli()
