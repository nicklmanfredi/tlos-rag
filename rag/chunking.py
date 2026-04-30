from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass

from .parse_transcripts import Turn


TARGET_TOKENS = 650
MIN_TOKENS = 400
MAX_TOKENS = 850
OVERLAP_SECONDS = 30.0


@dataclass
class Chunk:
    id: str
    episode_id: str
    episode_title: str
    source_file: str
    chunk_index: int
    text: str
    start_seconds: float
    end_seconds: float
    timestamp_source: str
    primary_speaker: str
    speakers: list[str]
    speaker_labels: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def token_count(text: str) -> int:
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def chunk_turns(turns: list[Turn]) -> list[Chunk]:
    chunks: list[Chunk] = []
    i = 0
    chunk_index = 0
    while i < len(turns):
        window: list[Turn] = []
        total = 0
        j = i
        while j < len(turns):
            next_tokens = token_count(turns[j].text)
            if window and total + next_tokens > MAX_TOKENS:
                break
            window.append(turns[j])
            total += next_tokens
            j += 1
            if total >= TARGET_TOKENS:
                break
        if total < MIN_TOKENS and j < len(turns):
            while j < len(turns) and total < MIN_TOKENS:
                window.append(turns[j])
                total += token_count(turns[j].text)
                j += 1
        chunks.append(_build_chunk(window, chunk_index))
        chunk_index += 1
        if j >= len(turns):
            break

        end = window[-1].end_seconds
        next_i = j
        for candidate in range(i + 1, j):
            if turns[candidate].start_seconds >= end - OVERLAP_SECONDS:
                next_i = candidate
                break
        if next_i <= i:
            next_i = j
        i = next_i
    return chunks


def _build_chunk(window: list[Turn], chunk_index: int) -> Chunk:
    speaker_counts: Counter[str] = Counter()
    labels: list[str] = []
    rendered: list[str] = []
    for turn in window:
        speaker_counts[turn.speaker] += token_count(turn.text)
        if turn.speaker_label not in labels:
            labels.append(turn.speaker_label)
        rendered.append(f"{turn.speaker_label}: {turn.text}")
    speakers = sorted(speaker_counts)
    primary = speaker_counts.most_common(1)[0][0] if speaker_counts else "other"
    text = "\n".join(rendered)
    first = window[0]
    last = window[-1]
    digest = hashlib.sha256(
        f"{first.episode_id}:{chunk_index}:{first.start_seconds}:{last.end_seconds}:{text}".encode("utf-8")
    ).hexdigest()[:24]
    return Chunk(
        id=digest,
        episode_id=first.episode_id,
        episode_title=first.episode_title,
        source_file=first.source_file,
        chunk_index=chunk_index,
        text=text,
        start_seconds=first.start_seconds,
        end_seconds=last.end_seconds,
        timestamp_source="source"
        if all(turn.timestamp_source == "source" for turn in window)
        else "estimated",
        primary_speaker=primary,
        speakers=speakers,
        speaker_labels=labels,
    )


def format_time(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
