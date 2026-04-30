from __future__ import annotations

from pathlib import Path

from .chunking import chunk_turns
from .config import Settings
from .embeddings import embed_texts
from .parse_transcripts import parse_transcript
from .store import append_embedding_cache, ensure_dirs, load_embedding_cache, write_catalog, write_lancedb


def ingest_folder(transcripts_dir: Path, settings: Settings, batch_size: int = 32) -> dict:
    ensure_dirs(settings)
    files = sorted(transcripts_dir.glob("*.txt"))
    chunks = []
    for path in files:
        turns = parse_transcript(path)
        if turns:
            chunks.extend(chunk_turns(turns))

    cache = load_embedding_cache(settings.embedding_cache)
    rows: list[dict] = []
    missing = [chunk for chunk in chunks if chunk.id not in cache]
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        vectors = embed_texts([chunk.text for chunk in batch], settings, input_type="document")
        cache_rows = [{"id": chunk.id, "vector": vector} for chunk, vector in zip(batch, vectors)]
        append_embedding_cache(settings.embedding_cache, cache_rows)
        for row in cache_rows:
            cache[row["id"]] = row["vector"]

    for chunk in chunks:
        row = chunk.to_dict()
        row["vector"] = cache[chunk.id]
        row["speakers_csv"] = ",".join(chunk.speakers)
        row["speaker_labels_csv"] = ",".join(chunk.speaker_labels)
        rows.append(row)

    write_catalog(settings, rows)
    write_lancedb(settings, rows)
    return {
        "transcript_files": len(files),
        "chunks": len(rows),
        "new_embeddings": len(missing),
        "embedding_provider": settings.embedding_provider,
    }

