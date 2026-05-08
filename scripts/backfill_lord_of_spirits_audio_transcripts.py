from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASE_URL = "https://www.ancientfaith.com"
PODCAST_URL = f"{BASE_URL}/podcasts/lordofspirits/"
USER_AGENT = "tlos-rag audio transcript backfill (+https://github.com/nicklmanfredi/tlos-rag)"
DEFAULT_TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-transcribe-diarize")
DEFAULT_CHUNK_SECONDS = 5 * 60


@dataclass(frozen=True)
class Episode:
    slug: str
    url: str
    title: str = ""
    published: str = ""
    duration: str = ""
    description: str = ""
    audio_url: str = ""
    has_official_transcript: bool = False

    @property
    def transcript_filename(self) -> str:
        return f"{self.slug.replace('-', '_')}.txt"

    @property
    def audio_filename(self) -> str:
        suffix = Path(self.audio_url.split("?", 1)[0]).suffix or ".mp3"
        return f"{self.slug.replace('-', '_')}{suffix}"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


class EpisodeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_transcript = False
        self.transcript_depth = 0
        self.transcript_parts: list[str] = []
        self.in_anchor = False
        self.anchor_href = ""
        self.anchor_text: list[str] = []
        self.anchors: list[tuple[str, str]] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "a":
            self.in_anchor = True
            self.anchor_href = attr.get("href", "")
            self.anchor_text = []

        if not self.in_transcript and tag == "div" and attr.get("id") == "transcript-reader":
            self.in_transcript = True
            self.transcript_depth = 1
            return

        if self.in_transcript:
            self.transcript_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.in_anchor:
            text = " ".join("".join(self.anchor_text).split())
            self.anchors.append((self.anchor_href, text))
            self.in_anchor = False
            self.anchor_href = ""
            self.anchor_text = []

        if self.in_transcript:
            self.transcript_depth -= 1
            if self.transcript_depth <= 0:
                self.in_transcript = False

    def handle_data(self, data: str) -> None:
        if self.in_anchor:
            self.anchor_text.append(data)
        if self.in_transcript:
            self.transcript_parts.append(data)
        if data.strip():
            self.text_parts.append(data.strip())

    def transcript_text(self) -> str:
        return " ".join("".join(self.transcript_parts).split())

    def page_text(self) -> str:
        return "\n".join(part.strip() for part in self.text_parts if part.strip())


def fetch(url: str, retries: int = 3, delay: float = 1.0) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=60) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(delay * attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def fetch_text(url: str) -> str:
    return fetch(url).decode("utf-8", errors="replace")


def episode_urls(pages: int, delay: float) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for page in range(1, pages + 1):
        html = fetch_text(f"{PODCAST_URL}?page={page}")
        parser = LinkParser()
        parser.feed(html)
        for href in parser.links:
            absolute = urljoin(BASE_URL, href)
            match = re.fullmatch(r"https://www\.ancientfaith\.com/podcasts/lordofspirits/([^/?#]+)/", absolute)
            if match and absolute not in seen:
                seen.add(absolute)
                urls.append(absolute)
        print(f"page {page}/{pages}: {len(urls)} episode URLs so far", flush=True)
        time.sleep(delay)
    return urls


def parse_episode(url: str) -> Episode:
    html = fetch_text(url)
    parser = EpisodeParser()
    parser.feed(html)
    text = parser.page_text()
    slug = url.rstrip("/").split("/")[-1]
    audio_url = ""
    for href, label in parser.anchors:
        absolute = urljoin(BASE_URL, href)
        if label.lower() == "download audio" or "media.ancientfaith.com" in absolute:
            audio_url = absolute
            break

    title = ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if line == "The Lord of Spirits" and index + 1 < len(lines):
            title = lines[index + 1]
            break

    published = ""
    duration = ""
    match = re.search(r"([A-Z][a-z]+day,\s+[A-Z][a-z]+ \d{1,2}, \d{4})\s+(\d+\s+mins)", text)
    if match:
        published = re.sub(r"\s+", " ", match.group(1))
        duration = re.sub(r"\s+", " ", match.group(2))

    description = ""
    if title and title in lines:
        title_index = lines.index(title)
        for line in lines[title_index + 1 :]:
            if re.match(r"[A-Z][a-z]+day, [A-Z][a-z]+ \d{1,2}, \d{4}", line):
                break
            if line not in {"Listen now", "Download audio"}:
                description = line
                break

    transcript_text = parser.transcript_text()
    has_official_transcript = bool(transcript_text) and "Transcripts can be commissioned" not in transcript_text

    return Episode(
        slug=slug,
        url=url,
        title=title,
        published=published,
        duration=duration,
        description=description,
        audio_url=audio_url,
        has_official_transcript=has_official_transcript,
    )


def discover_missing(args: argparse.Namespace) -> list[Episode]:
    existing = {path.name for path in args.out.glob("*.txt")}
    episodes: list[Episode] = []
    for index, url in enumerate(episode_urls(args.pages, args.delay), start=1):
        episode = parse_episode(url)
        transcript_exists = episode.transcript_filename in existing
        if not transcript_exists and not episode.has_official_transcript:
            episodes.append(episode)
            print(f"[{index}] missing transcript: {episode.transcript_filename} ({episode.duration or 'unknown duration'})", flush=True)
            if args.limit and len(episodes) >= args.limit:
                break
        time.sleep(args.delay)
    return episodes


def write_manifest(path: Path, episodes: list[Episode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for episode in episodes:
            fh.write(json.dumps(asdict(episode), ensure_ascii=False) + "\n")


def download_audio(episode: Episode, audio_dir: Path, overwrite: bool) -> Path:
    if not episode.audio_url:
        raise RuntimeError(f"no audio URL found for {episode.url}")
    audio_dir.mkdir(parents=True, exist_ok=True)
    path = audio_dir / episode.audio_filename
    if path.exists() and not overwrite:
        return path
    data = fetch(episode.audio_url, retries=4, delay=2.0)
    path.write_bytes(data)
    return path


def split_audio(audio_path: Path, chunk_dir: Path, chunk_seconds: int) -> list[Path]:
    if not shutil.which("ffmpeg"):
        if audio_path.stat().st_size <= 24_000_000:
            return [audio_path]
        raise RuntimeError(
            "ffmpeg is required to split long episodes before transcription; "
            "install ffmpeg or provide an audio file under 24 MB"
        )
    chunk_pattern = chunk_dir / "chunk_%04d.mp3"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "48k",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        str(chunk_pattern),
    ]
    subprocess.run(command, check=True)
    chunks = sorted(chunk_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise RuntimeError(f"ffmpeg did not produce chunks for {audio_path}")
    too_large = [path for path in chunks if path.stat().st_size > 24_000_000]
    if too_large:
        raise RuntimeError(f"audio chunks exceed OpenAI upload limit; lower --chunk-seconds: {too_large[0]}")
    return chunks


def transcribe_chunk(client, chunk_path: Path, model: str, offset: float):
    with chunk_path.open("rb") as audio_file:
        if model == "gpt-4o-transcribe-diarize":
            return client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="diarized_json",
                chunking_strategy="auto",
            )
        return client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="json",
            prompt=transcription_prompt(offset),
        )


def transcription_prompt(offset: float) -> str:
    return (
        "The audio is from The Lord of Spirits, an Orthodox Christian podcast hosted by "
        "Fr. Andrew Stephen Damick and Fr. Stephen De Young. Preserve Scripture references, "
        "patristic names, ancient place names, and theological terms. The chunk starts at "
        f"{format_timestamp(offset)} in the full episode."
    )


def transcribe_audio(audio_path: Path, episode: Episode, args: argparse.Namespace) -> str:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI()
    with tempfile.TemporaryDirectory(prefix=f"{episode.slug}-chunks-") as tmp:
        chunk_dir = Path(tmp)
        chunks = split_audio(audio_path, chunk_dir, args.chunk_seconds)
        rendered: list[str] = []
        for index, chunk_path in enumerate(chunks):
            offset = 0.0 if chunk_path == audio_path else float(index * args.chunk_seconds)
            print(f"transcribing {episode.transcript_filename} chunk {index + 1}/{len(chunks)}", flush=True)
            result = transcribe_chunk(client, chunk_path, args.model, offset)
            rendered.append(render_transcription_result(result, offset, args.model))
        return "\n".join(part.rstrip() for part in rendered if part.strip()).strip() + "\n"


def render_transcription_result(result, offset: float, model: str) -> str:
    segments = getattr(result, "segments", None)
    if segments:
        lines = []
        for segment in segments:
            speaker = getattr(segment, "speaker", None) or "Unknown"
            start = float(getattr(segment, "start", 0.0) or 0.0) + offset
            text = str(getattr(segment, "text", "")).strip()
            if text:
                lines.append(f"[{format_timestamp(start)}] {normalize_speaker_label(speaker)}: {text}")
        return "\n".join(lines)

    text = getattr(result, "text", None)
    if text is None:
        text = str(result)
    return f"[{format_timestamp(offset)}] Unknown: {str(text).strip()}"


def normalize_speaker_label(value: str) -> str:
    label = str(value).strip().replace("_", " ")
    label = re.sub(r"\s+", " ", label)
    if not label:
        return "Unknown"
    if label.lower().startswith("speaker"):
        return label.title()
    return label


def format_timestamp(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def write_transcript(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise RuntimeError(f"transcript already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ingest(out_dir: Path) -> None:
    from rag.config import settings
    from rag.ingest import ingest_folder

    result = ingest_folder(out_dir, settings())
    print(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover Lord of Spirits episodes without official transcripts, download audio, and make machine transcripts."
    )
    parser.add_argument("--out", type=Path, default=ROOT / "transcripts" / "lordofspirits")
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "data" / "audio" / "lordofspirits")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "missing_lordofspirits_episodes.jsonl")
    parser.add_argument("--pages", type=int, default=28)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--limit", type=int, help="Only process the first N missing episodes.")
    parser.add_argument("--download", action="store_true", help="Download missing episode audio.")
    parser.add_argument("--transcribe", action="store_true", help="Transcribe downloaded audio and write transcript .txt files.")
    parser.add_argument("--ingest", action="store_true", help="Refresh the RAG database after transcripts are written.")
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--overwrite-transcript", action="store_true")
    parser.add_argument("--model", default=DEFAULT_TRANSCRIPTION_MODEL)
    parser.add_argument("--chunk-seconds", type=int, default=DEFAULT_CHUNK_SECONDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out = args.out.expanduser()
    args.audio_dir = args.audio_dir.expanduser()
    args.manifest = args.manifest.expanduser()

    episodes = discover_missing(args)
    write_manifest(args.manifest, episodes)
    print(f"wrote manifest: {args.manifest}", flush=True)
    print(f"missing episodes: {len(episodes)}", flush=True)

    if not args.download and not args.transcribe and not args.ingest:
        print("dry run only; add --download, --transcribe, and/or --ingest to take action", flush=True)
        return

    written = 0
    for episode in episodes:
        audio_path = download_audio(episode, args.audio_dir, args.overwrite_audio) if args.download or args.transcribe else None
        if args.transcribe:
            assert audio_path is not None
            transcript_path = args.out / episode.transcript_filename
            text = transcribe_audio(audio_path, episode, args)
            write_transcript(transcript_path, text, args.overwrite_transcript)
            written += 1
            print(f"wrote transcript: {transcript_path}", flush=True)

    if args.ingest:
        ingest(args.out)
    print(f"done: downloaded={args.download} transcripts_written={written}", flush=True)


if __name__ == "__main__":
    main()
