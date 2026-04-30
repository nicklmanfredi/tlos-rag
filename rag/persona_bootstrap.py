from __future__ import annotations

from pathlib import Path

from .chat import anthropic_client
from .config import Settings, host_display, host_slug
from .store import load_catalog


BOOTSTRAP_PROMPT = """Write a detailed persona guide for {host}.
Cover: voice and cadence, verbal tics and catchphrases, recurring opinions and worldview, what they find funny,
what they push back on, how they interact with their co-host, topics they return to obsessively,
and areas where they defer vs take the lead.

Base the guide only on these primary-speaker transcript samples. Do not quote long passages.
"""


def bootstrap_persona(host: str, settings: Settings, max_chars: int = 120_000) -> Path:
    slug = host_slug(host)
    chunks = [row for row in load_catalog(settings) if row["primary_speaker"] == slug]
    if not chunks:
        raise RuntimeError(f"No indexed chunks found for {host_display(slug)}. Run ingest first.")

    samples = diverse_samples(chunks, max_chars=max_chars)
    sample_text = "\n\n".join(
        f'[{row["episode_title"]} {int(row["start_seconds"] // 60)}:{int(row["start_seconds"] % 60):02d}]\n{row["text"]}'
        for row in samples
    )
    client = anthropic_client(settings)
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=4000,
        system=[{"type": "text", "text": "You are an expert editor building a practical persona style guide."}],
        messages=[
            {
                "role": "user",
                "content": BOOTSTRAP_PROMPT.format(host=host_display(slug)) + "\n\n<samples>\n" + sample_text + "\n</samples>",
            }
        ],
    )
    text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    settings.personas_dir.mkdir(parents=True, exist_ok=True)
    path = settings.personas_dir / f"persona_{slug}.md"
    path.write_text(text, encoding="utf-8")
    return path


def diverse_samples(chunks: list[dict], max_chars: int) -> list[dict]:
    by_episode: dict[str, list[dict]] = {}
    for row in chunks:
        by_episode.setdefault(row["episode_id"], []).append(row)
    selected = []
    total = 0
    for episode_id in sorted(by_episode):
        rows = by_episode[episode_id]
        row = rows[len(rows) // 2]
        if total + len(row["text"]) > max_chars:
            break
        selected.append(row)
        total += len(row["text"])
    return selected

