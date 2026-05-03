from __future__ import annotations

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .chunking import chunk_turns
from .config import Settings, host_slug
from .embeddings import STOPWORDS, content_terms
from .parse_transcripts import parse_transcript
from .retrieval import add_neighbors


def retrieve_text(query: str, settings: Settings, host: str | None = None, final_k: int = 8) -> list[dict]:
    catalog = load_text_catalog(settings.transcripts_dir)
    host_filter = host_slug(host) if host else None
    searchable = [row for row in catalog if row["primary_speaker"] == host_filter] if host_filter else catalog
    if not searchable:
        return []

    ranked = bm25_text_search(query, searchable, limit=max(final_k * 3, 24))
    if host_filter:
        ranked = add_neighbors(ranked, catalog, final_k)
    return ranked[:final_k]


def load_text_catalog(transcripts_dir: Path) -> list[dict]:
    files = sorted(transcripts_dir.glob("*.txt"))
    chunks = []
    for path in files:
        turns = parse_transcript(path)
        if turns:
            chunks.extend(chunk.to_dict() for chunk in chunk_turns(turns))
    return chunks


def bm25_text_search(query: str, rows: list[dict], limit: int) -> list[dict]:
    query_tokens = tokenize(query)
    tokenized = [tokenize(f'{row.get("episode_title", "")} {row["text"]}') for row in rows]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query_tokens)
    query_terms = content_terms(query)
    ranked = []
    for score, row in zip(scores, rows):
        text = f'{row.get("episode_title", "")} {row["text"]}'.lower()
        title_terms = content_terms(row.get("episode_title", ""))
        overlap = len(query_terms & content_terms(text)) / max(1, len(query_terms))
        title_bonus = 0.5 * len(query_terms & title_terms) / max(1, len(query_terms))
        item = dict(row)
        item["text_search_score"] = float(score) + overlap + title_bonus
        ranked.append(item)
    return sorted(ranked, key=lambda row: row["text_search_score"], reverse=True)[:limit]


def tokenize(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9']+", text.lower()) if term not in STOPWORDS and len(term) > 2]
