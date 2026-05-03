from __future__ import annotations

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .chunking import chunk_turns
from .config import Settings, host_slug
from .embeddings import STOPWORDS, content_terms
from .parse_transcripts import parse_transcript
from .retrieval import add_neighbors, reciprocal_rank_fusion


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


def retrieve_agentic(query: str, settings: Settings, host: str | None = None, final_k: int = 8) -> list[dict]:
    catalog = load_text_catalog(settings.transcripts_dir)
    host_filter = host_slug(host) if host else None
    searchable = [row for row in catalog if row["primary_speaker"] == host_filter] if host_filter else catalog
    if not searchable:
        return []

    planned_queries = plan_agentic_queries(query)
    rankings = [bm25_text_search(planned_query, searchable, limit=max(final_k * 3, 16)) for planned_query in planned_queries]
    fused = reciprocal_rank_fusion(rankings)
    with_neighbors = add_neighbors(fused, catalog, max(final_k * 2, final_k + 4))
    ranked = rank_agentic_evidence(query, planned_queries, with_neighbors)
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


def plan_agentic_queries(query: str, max_queries: int = 8) -> list[str]:
    candidates: list[str] = [query]
    candidates.extend(extract_question_lines(query))

    phrases = extract_quoted_phrases(query)
    candidates.extend(phrases)

    terms = ordered_content_terms(query)
    if terms:
        candidates.append(" ".join(terms[:10]))

    theme_terms = [term for term in terms if len(term) > 4]
    for start in range(0, min(len(theme_terms), 18), 3):
        group = theme_terms[start : start + 6]
        if len(group) >= 2:
            candidates.append(" ".join(group))

    candidates.extend(term_windows(terms, width=3))

    planned: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = normalize_query(candidate)
        if normalized and normalized not in seen:
            planned.append(normalized)
            seen.add(normalized)
        if len(planned) >= max_queries:
            break
    return planned or [query]


def extract_question_lines(query: str) -> list[str]:
    questions = []
    for line in query.splitlines():
        line = re.sub(r"^\s*\d+\.\s*", "", line).strip()
        if line.endswith("?"):
            questions.append(line)
    if questions:
        return questions
    return [part.strip() for part in re.split(r"(?<=[?])\s+", query) if part.strip().endswith("?")]


def extract_quoted_phrases(query: str) -> list[str]:
    phrases: list[str] = []
    for double_quoted, single_quoted in re.findall(r'"([^"]+)"|\'([^\']+)\'', query):
        phrase = (double_quoted or single_quoted).strip()
        if phrase:
            phrases.append(phrase)
    return phrases


def ordered_content_terms(text: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for term in tokenize(text):
        if term not in seen:
            terms.append(term)
            seen.add(term)
    return terms


def term_windows(terms: list[str], width: int) -> list[str]:
    if width <= 1 or len(terms) < width:
        return []
    return [" ".join(terms[start : start + width]) for start in range(0, len(terms) - width + 1)]


def normalize_query(query: str) -> str:
    tokens = tokenize(query)
    return " ".join(tokens)


def rank_agentic_evidence(original_query: str, planned_queries: list[str], rows: list[dict]) -> list[dict]:
    original_terms = content_terms(original_query)
    planned_terms = content_terms(" ".join(planned_queries))
    scored = []
    for row in rows:
        text = f'{row.get("episode_title", "")} {row["text"]}'.lower()
        text_terms = content_terms(text)
        original_overlap = len(original_terms & text_terms) / max(1, len(original_terms))
        planned_overlap = len(planned_terms & text_terms) / max(1, len(planned_terms))
        score = float(row.get("rrf_score", 0.0)) + original_overlap + (0.5 * planned_overlap)
        item = dict(row)
        item["agentic_score"] = score
        item["agentic_queries"] = planned_queries
        scored.append((score, item))
    return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)]


def tokenize(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9']+", text.lower()) if term not in STOPWORDS and len(term) > 2]
