from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import HOSTS, PODCAST_NAME


BATCH_SIZE = 60
OVERLAP = 10

_HOST_LIST = "\n".join(f"- {meta['display']}" for meta in HOSTS.values())
_HOST_KEYS = list(HOSTS.keys())

SYSTEM_PROMPT = f"""You are labeling speaker turns in a podcast transcript for "{PODCAST_NAME}".

The hosts are:
{_HOST_LIST}

Each segment below is a short excerpt from the episode. Segments sometimes contain brief
back-and-forth, so label with the dominant speaker — the one who speaks the most in that segment.

Return ONLY a JSON array. Each element must have:
  "id": the integer id shown before the segment
  "speaker": one of {json.dumps(_HOST_KEYS + ["unknown"])}

Use "unknown" only if the segment truly cannot be attributed (e.g. both hosts speak equally,
or it is a non-speech sound). Do not explain anything. Output only the JSON array.
"""


def parse_segments(text: str) -> list[tuple[str, str]]:
    segments = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\[(\d{1,2}:\d{2}:\d{2})\] \w[^:]+: (.+)$", line)
        if m:
            segments.append((m.group(1), m.group(2)))
    return segments


def label_batch(
    client,
    segments: list[tuple[str, str]],
    context: list[tuple[str, str]],
    id_offset: int,
) -> dict[int, str]:
    lines = []
    if context:
        lines.append("=== Prior context (do not label these) ===")
        for ts, text in context:
            lines.append(f"  [{ts}] {text}")
        lines.append("=== Segments to label ===")
    for i, (ts, text) in enumerate(segments):
        lines.append(f"{id_offset + i}: [{ts}] {text}")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": "\n".join(lines)}],
    )

    raw = response.content[0].text.strip()
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return {id_offset + i: "unknown" for i in range(len(segments))}
    try:
        items = json.loads(m.group())
        return {int(item["id"]): item["speaker"] for item in items}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {id_offset + i: "unknown" for i in range(len(segments))}


def diarize_file(path: Path, client, output_path: Path) -> None:
    segments = parse_segments(path.read_text(encoding="utf-8"))
    if not segments:
        print(f"  skipping {path.name}: no parseable segments")
        return

    labels: dict[int, str] = {}
    context: list[tuple[str, str]] = []

    for batch_start in range(0, len(segments), BATCH_SIZE):
        batch = segments[batch_start : batch_start + BATCH_SIZE]
        batch_labels = label_batch(client, batch, context, batch_start)
        labels.update(batch_labels)
        context = batch[-OVERLAP:]

    speaker_display = {slug: meta["display"] for slug, meta in HOSTS.items()}
    speaker_display["unknown"] = "Unknown"

    lines = []
    for i, (ts, text) in enumerate(segments):
        slug = labels.get(i, "unknown")
        display = speaker_display.get(slug, "Unknown")
        lines.append(f"[{ts}] {display}: {text}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    counts = {slug: sum(1 for v in labels.values() if v == slug) for slug in list(HOSTS.keys()) + ["unknown"]}
    summary = " ".join(f"{slug}={n}" for slug, n in counts.items())
    print(f"  {len(segments)} segments — {summary}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Label speakers in transcripts using Claude. Reads the current PODCAST_CONFIG for host names."
    )
    ap.add_argument("input_dir", type=Path, help="Folder of transcripts with placeholder speaker labels.")
    ap.add_argument("output_dir", type=Path, help="Folder to write relabeled transcripts.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    ap.add_argument("--delay", type=float, default=0.5, help="Delay between episodes in seconds.")
    args = ap.parse_args()

    if not HOSTS:
        sys.exit("No hosts found in podcast config. Set PODCAST_CONFIG or add a podcast.json.")

    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.txt"))
    print(f'Diarizing {len(files)} transcripts for "{PODCAST_NAME}" ({", ".join(m["display"] for m in HOSTS.values())})')

    for idx, path in enumerate(files, 1):
        out = args.output_dir / path.name
        if out.exists() and not args.overwrite:
            print(f"[{idx}/{len(files)}] skip {path.name}")
            continue
        print(f"[{idx}/{len(files)}] {path.name}")
        diarize_file(path, client, out)
        if idx < len(files):
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
