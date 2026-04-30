from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import lancedb

from .config import Settings


def ensure_dirs(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.lancedb_dir.mkdir(parents=True, exist_ok=True)


def load_embedding_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    cache: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["id"]] = row["vector"]
    return cache


def append_embedding_cache(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_catalog(settings: Settings, rows: list[dict]) -> None:
    settings.chunk_catalog.parent.mkdir(parents=True, exist_ok=True)
    with settings.chunk_catalog.open("w", encoding="utf-8") as fh:
        for row in rows:
            item = {k: v for k, v in row.items() if k != "vector"}
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_catalog(settings: Settings) -> list[dict]:
    if not settings.chunk_catalog.exists():
        return []
    return [json.loads(line) for line in settings.chunk_catalog.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lancedb(settings: Settings, rows: list[dict]) -> None:
    ensure_dirs(settings)
    db = lancedb.connect(str(settings.lancedb_dir))
    if not rows:
        return
    db.create_table(settings.table_name, data=rows, mode="overwrite")


def open_table(settings: Settings):
    db = lancedb.connect(str(settings.lancedb_dir))
    return db.open_table(settings.table_name)

