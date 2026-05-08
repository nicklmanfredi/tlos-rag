from __future__ import annotations

import re
from collections import Counter, defaultdict

from rank_bm25 import BM25Okapi

from .config import Settings, host_slug
from .embeddings import STOPWORDS, content_terms, embed_texts, rerank
from .store import load_catalog, open_table


def tokenize(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9']+", text.lower()) if term not in STOPWORDS and len(term) > 2]


def retrieve(
    query: str,
    settings: Settings,
    host: str | None = None,
    final_k: int = 8,
    search_backend: str = "rag",
) -> list[dict]:
    if search_backend == "agentic":
        return retrieve_agentic(query, settings, host=host, final_k=final_k)
    if search_backend == "text":
        from .text_search import retrieve_text

        return retrieve_text(query, settings, host=host, final_k=final_k)
    if search_backend != "rag":
        raise ValueError(f"Unsupported search_backend={search_backend}")

    host_filter, catalog, searchable = searchable_catalog(settings, host)
    if not searchable:
        return []

    ranked = retrieve_rag(query, settings, catalog, searchable, host_filter, final_k)
    return ranked[:final_k]


def searchable_catalog(settings: Settings, host: str | None = None) -> tuple[str | None, list[dict], list[dict]]:
    host_filter = host_slug(host) if host else None
    catalog = load_catalog(settings)
    searchable = [row for row in catalog if row["primary_speaker"] == host_filter] if host_filter else catalog
    return host_filter, catalog, searchable


def retrieve_rag(
    query: str,
    settings: Settings,
    catalog: list[dict],
    searchable: list[dict],
    host_filter: str | None,
    final_k: int,
) -> list[dict]:
    if not searchable:
        return []

    semantic = semantic_search(query, settings, host_filter, limit=20)
    keyword = bm25_search(query, searchable, limit=20)
    fused = reciprocal_rank_fusion([semantic, keyword])
    candidates = fused[:32]
    ranked = rerank(query, candidates, settings, top_k=max(final_k * 2, final_k + 4))
    expanded = add_neighbors(ranked, catalog, max(final_k * 2, final_k + 4))
    return rerank(query, expanded, settings, top_k=final_k)


def retrieve_agentic(query: str, settings: Settings, host: str | None = None, final_k: int = 8) -> list[dict]:
    from .text_search import plan_agentic_queries, rank_agentic_evidence, retrieve_agentic as retrieve_text_agentic

    host_filter, catalog, searchable = searchable_catalog(settings, host)
    if not searchable:
        return retrieve_text_agentic(query, settings, host=host, final_k=final_k)

    planned_queries = plan_agentic_queries(query)
    rankings = [
        retrieve_rag(
            planned_query,
            settings,
            catalog,
            searchable,
            host_filter,
            final_k=max(final_k * 2, final_k + 4),
        )
        for planned_query in planned_queries
    ]
    fused = reciprocal_rank_fusion(rankings)
    expansion_queries = corpus_expansion_queries(query, fused[:12], planned_queries)
    if expansion_queries:
        planned_queries.extend(expansion_queries)
        rankings.extend(
            retrieve_rag(
                planned_query,
                settings,
                catalog,
                searchable,
                host_filter,
                final_k=max(final_k * 2, final_k + 4),
            )
            for planned_query in expansion_queries
        )
        fused = reciprocal_rank_fusion(rankings)
    expanded = add_neighbors(fused, catalog, max(final_k * 3, final_k + 8))
    ranked = rank_agentic_evidence(query, planned_queries, expanded)
    ranked = rerank(" ".join([query, *planned_queries]), ranked[: max(final_k * 3, 24)], settings, top_k=final_k)
    for row in ranked:
        row["agentic_queries"] = planned_queries
    return ranked


def corpus_expansion_queries(
    original_query: str,
    rows: list[dict],
    existing_queries: list[str],
    max_queries: int = 4,
) -> list[str]:
    original_terms = ordered_unique_terms(original_query)
    if not original_terms or not rows:
        return []

    original_set = set(original_terms)
    counts: Counter[str] = Counter()
    title_counts: Counter[str] = Counter()
    for row in rows:
        title_terms = content_terms(row.get("episode_title", ""))
        text_terms = content_terms(row.get("text", ""))
        for term in text_terms - original_set:
            if len(term) > 4:
                counts[term] += 1
        for term in title_terms - original_set:
            if len(term) > 4:
                title_counts[term] += 1

    for term, count in title_counts.items():
        counts[term] += count * 2

    anchors = original_terms[:4]
    expansions = [term for term, _ in counts.most_common(16)]
    candidates = []
    for start in range(0, len(expansions), 4):
        group = expansions[start : start + 4]
        if len(group) >= 2:
            candidates.append(" ".join([*anchors, *group]))
        if len(candidates) >= max_queries:
            break

    seen = {normalize_query_text(query) for query in existing_queries}
    planned = []
    for candidate in candidates:
        normalized = normalize_query_text(candidate)
        if normalized and normalized not in seen:
            planned.append(normalized)
            seen.add(normalized)
    return planned


def ordered_unique_terms(text: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for term in tokenize(text):
        if term not in seen:
            terms.append(term)
            seen.add(term)
    return terms


def normalize_query_text(text: str) -> str:
    return " ".join(tokenize(text))


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
