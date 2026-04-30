from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import canonical_speaker


SPEAKER_RE = re.compile(
    r"^\s*(?:(?:\[|\()?(?P<time>\d{1,2}:\d{2}(?::\d{2})?)(?:\]|\))?\s+)?"
    r"(?P<speaker>[A-Z][A-Za-z0-9 .'\-]+?):\s*(?P<text>.*)$"
)


@dataclass
class Turn:
    episode_id: str
    episode_title: str
    source_file: str
    speaker_label: str
    speaker: str
    text: str
    start_seconds: float
    end_seconds: float
    timestamp_source: str

    def to_dict(self) -> dict:
        return asdict(self)


def title_from_path(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").title()


def parse_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    parts = [int(p) for p in value.split(":")]
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    return None


def estimate_duration_seconds(text: str, words_per_minute: int = 155) -> float:
    words = max(1, len(re.findall(r"\w+", text)))
    return max(2.0, words / words_per_minute * 60.0)


def parse_transcript(path: Path) -> list[Turn]:
    episode_id = path.stem
    episode_title = title_from_path(path)
    turns: list[Turn] = []
    current: dict | None = None
    running_seconds = 0.0

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = SPEAKER_RE.match(line)
        if match:
            if current:
                running_seconds = _finish_turn(current, turns, running_seconds)
            start = parse_timestamp(match.group("time"))
            current = {
                "speaker_label": match.group("speaker").strip(),
                "speaker": canonical_speaker(match.group("speaker")),
                "text": match.group("text").strip(),
                "explicit_start": start,
            }
        elif current:
            current["text"] = f'{current["text"]} {line}'.strip()
        else:
            current = {
                "speaker_label": "Unknown",
                "speaker": "other",
                "text": line,
                "explicit_start": None,
            }

    if current:
        _finish_turn(current, turns, running_seconds)

    for turn in turns:
        turn.episode_id = episode_id
        turn.episode_title = episode_title
        turn.source_file = str(path)
    return turns


def _finish_turn(current: dict, turns: list[Turn], running_seconds: float) -> float:
    text = current["text"].strip()
    if not text:
        return running_seconds
    explicit_start = current["explicit_start"]
    start = explicit_start if explicit_start is not None else running_seconds
    duration = estimate_duration_seconds(text)
    end = max(start + duration, running_seconds + duration)
    turns.append(
        Turn(
            episode_id="",
            episode_title="",
            source_file="",
            speaker_label=current["speaker_label"],
            speaker=current["speaker"],
            text=text,
            start_seconds=start,
            end_seconds=end,
            timestamp_source="source" if explicit_start is not None else "estimated",
        )
    )
    return end

