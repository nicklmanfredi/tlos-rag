from __future__ import annotations

import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from .config import Settings, host_slug
from .embeddings import STOPWORDS, embed_texts, rerank
from .store import load_catalog, open_table


def tokenize(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9']+", text.lower()) if term not in STOPWORDS and len(term) > 2]


def retrieve(query: str, settings: Settings, host: str | None = None, final_k: int = 8) -> list[dict]:
    host_filter = host_slug(host) if host else None
    catalog = load_catalog(settings)
    if host_filter:
        searchable = [row for row in catalog if row["primary_speaker"] == host_filter]
    else:
        searchable = catalog
    if not searchable:
        return []

    semantic = [] if settings.embedding_provider == "local" else semantic_search(query, settings, host_filter, limit=20)
    keyword = bm25_search(query, searchable, limit=20)
    fused = reciprocal_rank_fusion([semantic, keyword])
    candidates = [row for row in fused[:24]]
    ranked = rerank(query, candidates, settings, top_k=final_k)
    if host_filter:
        ranked = add_neighbors(ranked, catalog, final_k)
    return ranked[:final_k]


def semantic_search(query: str, settings: Settings, host_filter: str | None, limit: int) -> list[dict]:
    vector = embed_texts([query], settings, input_type="query")[0]
    table = open_table(settings)
    search = table.search(vector)
    if host_filter:
        search = search.where(f"primary_speaker = '{host_filter}'")
    search = search.limit(limit)
    return [_normalize_lance_row(row) for row in search.to_list()]


def bm25_search(query: str, rows: list[dict], limit: int) -> list[dict]:
    tokenized = [tokenize(f'{row.get("episode_title", "")} {row["text"]}') for row in rows]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(scores, rows), key=lambda item: item[0], reverse=True)
    results = []
    for score, row in ranked[:limit]:
        item = dict(row)
        item["bm25_score"] = float(score)
        results.append(item)
    return results


def reciprocal_rank_fusion(rankings: list[list[dict]], k: int = 60) -> list[dict]:
    scores = defaultdict(float)
    docs: dict[str, dict] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            doc_id = doc["id"]
            docs[doc_id] = doc
            scores[doc_id] += 1.0 / (k + rank)
    return [dict(docs[doc_id], rrf_score=score) for doc_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def add_neighbors(ranked: list[dict], catalog: list[dict], final_k: int) -> list[dict]:
    by_key = {(row["episode_id"], row["chunk_index"]): row for row in catalog}
    expanded: list[dict] = []
    seen: set[str] = set()
    for row in ranked:
        for idx in (row["chunk_index"] - 1, row["chunk_index"], row["chunk_index"] + 1):
            neighbor = by_key.get((row["episode_id"], idx))
            if neighbor and neighbor["id"] not in seen:
                expanded.append(neighbor if neighbor["id"] != row["id"] else row)
                seen.add(neighbor["id"])
            if len(expanded) >= final_k:
                return expanded
    return expanded


def _normalize_lance_row(row: dict) -> dict:
    row = dict(row)
    row.pop("vector", None)
    row.pop("_distance", None)
    if isinstance(row.get("speakers"), str):
        row["speakers"] = row["speakers"].split(",")
    elif not isinstance(row.get("speakers"), list):
        row["speakers"] = list(row.get("speakers", []))
    return row
