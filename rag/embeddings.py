from __future__ import annotations

import hashlib
import math
import os
from typing import Sequence

from .config import Settings


LOCAL_DIM = 384


def embed_texts(texts: Sequence[str], settings: Settings, input_type: str) -> list[list[float]]:
    provider = settings.embedding_provider
    if provider == "voyage":
        import voyageai

        client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        result = client.embed(list(texts), model=settings.embedding_model, input_type=input_type)
        return [list(vec) for vec in result.embeddings]
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        result = client.embeddings.create(model=settings.embedding_model, input=list(texts))
        return [list(item.embedding) for item in result.data]
    if provider == "local":
        return [local_embedding(text) for text in texts]
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER={provider}")


def local_embedding(text: str) -> list[float]:
    vector = [0.0] * LOCAL_DIM
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % LOCAL_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def rerank(query: str, docs: list[dict], settings: Settings, top_k: int) -> list[dict]:
    if not docs:
        return []
    if settings.rerank_provider == "voyage":
        import voyageai

        client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        result = client.rerank(
            query=query,
            documents=[doc["text"] for doc in docs],
            model=settings.rerank_model,
            top_k=min(top_k, len(docs)),
        )
        ranked = []
        for item in result.results:
            idx = getattr(item, "index", None)
            score = getattr(item, "relevance_score", None)
            doc = dict(docs[idx])
            doc["rerank_score"] = score
            ranked.append(doc)
        return ranked
    if settings.rerank_provider in {"local", "none"}:
        query_terms = set(query.lower().split())
        scored = []
        for doc in docs:
            terms = set(doc["text"].lower().split())
            score = len(query_terms & terms) / max(1, len(query_terms))
            scored.append((score, doc))
        return [dict(doc, rerank_score=score) for score, doc in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]]
    raise ValueError(f"Unsupported RERANK_PROVIDER={settings.rerank_provider}")
