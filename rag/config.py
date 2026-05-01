from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _load_podcast_config() -> dict:
    env_path = os.getenv("PODCAST_CONFIG")
    if env_path:
        config_path = Path(env_path).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"PODCAST_CONFIG={env_path!r} not found")
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        if not cfg.get("hosts"):
            raise ValueError(f"PODCAST_CONFIG={env_path!r} defines no hosts")
        return cfg
    config_path = project_root() / "podcast.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    return {"name": "Podcast", "hosts": {}}


_podcast_config = _load_podcast_config()
PODCAST_NAME: str = _podcast_config.get("name", "Podcast")
HOSTS: dict = {
    slug: {
        "display": meta["display"],
        "aliases": set(meta["aliases"]),
    }
    for slug, meta in _podcast_config.get("hosts", {}).items()
}


def canonical_speaker(label: str) -> str:
    normalized = re.sub(r"\s+", " ", label.strip().lower().rstrip(":"))
    for slug, meta in HOSTS.items():
        for alias in meta["aliases"]:
            if alias in normalized:
                return slug
    return "other"


def host_slug(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    for slug, meta in HOSTS.items():
        if normalized == slug or normalized in meta["aliases"]:
            return slug
    candidate = slugify(value)
    if candidate in HOSTS:
        return candidate
    raise ValueError(f"Unknown host: {value}. Known hosts: {', '.join(h['display'] for h in HOSTS.values())}")


def host_display(slug: str) -> str:
    return HOSTS.get(slug, {}).get("display", slug)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    transcripts_dir: Path
    lancedb_dir: Path
    table_name: str
    chunk_catalog: Path
    embedding_cache: Path
    personas_dir: Path
    podcast_name: str
    embedding_provider: str
    embedding_model: str
    rerank_provider: str
    rerank_model: str
    chat_provider: str
    anthropic_model: str
    aws_region: str | None
    codex_bin: str
    codex_model: str | None
    codex_oss: bool
    codex_local_provider: str | None
    tts_model: str
    tts_voice_andrew: str
    tts_voice_stephen: str


def settings() -> Settings:
    root = project_root()
    data_dir = Path(os.getenv("RAG_DATA_DIR", root / "data")).expanduser()
    return Settings(
        data_dir=data_dir,
        transcripts_dir=Path(os.getenv("TRANSCRIPTS_DIR", root / "transcripts" / "lordofspirits")).expanduser(),
        lancedb_dir=Path(os.getenv("LANCEDB_DIR", data_dir / "lancedb")).expanduser(),
        table_name=os.getenv("LANCEDB_TABLE", "transcript_chunks"),
        chunk_catalog=Path(os.getenv("CHUNK_CATALOG", data_dir / "chunks.jsonl")).expanduser(),
        embedding_cache=Path(os.getenv("EMBEDDING_CACHE", data_dir / "embedding_cache.jsonl")).expanduser(),
        personas_dir=Path(os.getenv("PERSONAS_DIR", root / "personas")).expanduser(),
        podcast_name=PODCAST_NAME,
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "voyage").lower(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "voyage-3"),
        rerank_provider=os.getenv("RERANK_PROVIDER", "voyage").lower(),
        rerank_model=os.getenv("RERANK_MODEL", "rerank-2"),
        chat_provider=os.getenv("CHAT_PROVIDER", "anthropic").lower(),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7"),
        aws_region=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
        codex_bin=os.getenv("CODEX_BIN", "codex"),
        codex_model=os.getenv("CODEX_MODEL") or None,
        codex_oss=os.getenv("CODEX_OSS", "").lower() in {"1", "true", "yes"},
        codex_local_provider=os.getenv("CODEX_LOCAL_PROVIDER") or None,
        tts_model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
        tts_voice_andrew=os.getenv("TTS_VOICE_ANDREW", "ash"),
        tts_voice_stephen=os.getenv("TTS_VOICE_STEPHEN", "echo"),
    )
