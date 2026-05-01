from __future__ import annotations

import argparse
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


BASE_URL = "https://podscripts.co"
USER_AGENT = "podcast-rag transcript fetcher (+https://github.com/tomx4096/tlos-rag)"


def fetch(url: str, retries: int = 5, delay: float = 1.0) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt < retries:
                wait = 30 * attempt
                print(f"  rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif attempt < retries:
                time.sleep(delay * attempt)
        except (URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(delay * attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


class EpisodeListParser(HTMLParser):
    """Extracts episode slugs and titles from a podcast listing page."""

    def __init__(self, podcast_slug: str) -> None:
        super().__init__(convert_charrefs=True)
        self.podcast_slug = podcast_slug
        self.episodes: list[tuple[str, str]] = []
        self._in_h3 = False
        self._episode_pattern = re.compile(
            rf"^/podcasts/{re.escape(podcast_slug)}/([^/?#]+)$"
        )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "h3":
            self._in_h3 = True
        if tag == "a" and self._in_h3:
            href = dict(attrs).get("href", "")
            match = self._episode_pattern.match(href)
            if match:
                self._pending_slug = match.group(1)

    def handle_endtag(self, tag: str) -> None:
        if tag == "h3":
            self._in_h3 = False
            self._pending_slug = None

    def handle_data(self, data: str) -> None:
        slug = getattr(self, "_pending_slug", None)
        if self._in_h3 and slug:
            title = data.strip()
            if title:
                self.episodes.append((slug, title))
                self._pending_slug = None


class TranscriptParser(HTMLParser):
    """Extracts timestamped segments from a podscripts.co episode page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.segments: list[tuple[str, str]] = []
        self._in_sentence = False
        self._in_timestamp = False
        self._in_text = False
        self._current_time: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = set(dict(attrs).get("class", "").split())
        if tag == "div" and "single-sentence" in classes:
            self._in_sentence = True
            self._current_time = None
            self._parts = []
        elif self._in_sentence and tag == "span" and "pod_timestamp_indicator" in classes:
            self._in_timestamp = True
        elif self._in_sentence and tag == "span" and "transcript-text" in classes:
            self._in_text = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "span":
            if self._in_timestamp:
                self._in_timestamp = False
            elif self._in_text:
                self._in_text = False
        elif tag == "div" and self._in_sentence:
            text = " ".join(self._parts).strip()
            text = re.sub(r"\s+", " ", text)
            if self._current_time and text:
                self.segments.append((self._current_time, text))
            self._in_sentence = False

    def handle_data(self, data: str) -> None:
        if self._in_timestamp:
            match = re.search(r"(\d{1,2}:\d{2}:\d{2})", data)
            if match:
                self._current_time = match.group(1)
        elif self._in_text:
            chunk = data.strip()
            if chunk:
                self._parts.append(chunk)


def episode_urls(podcast_slug: str, pages: int, delay: float) -> list[tuple[str, str]]:
    seen: set[str] = set()
    episodes: list[tuple[str, str]] = []
    for page in range(1, pages + 1):
        url = f"{BASE_URL}/podcasts/{podcast_slug}?page={page}"
        html = fetch(url)
        parser = EpisodeListParser(podcast_slug)
        parser.feed(html)
        new = 0
        for slug, title in parser.episodes:
            if slug not in seen:
                seen.add(slug)
                episodes.append((slug, title))
                new += 1
        print(f"page {page}/{pages}: {new} new, {len(episodes)} total")
        if new == 0:
            break
        time.sleep(delay)
    return episodes


def extract_transcript(html: str) -> list[tuple[str, str]]:
    parser = TranscriptParser()
    parser.feed(html)
    return parser.segments


def segments_to_text(segments: list[tuple[str, str]]) -> str:
    lines = []
    for timestamp, text in segments:
        lines.append(f"[{timestamp}] Speaker: {text}")
    return "\n".join(lines) + "\n"


def main() -> None:
    argp = argparse.ArgumentParser(
        description="Fetch transcripts for any podcast on podscripts.co."
    )
    argp.add_argument("podcast_slug", help="Podcast slug from the podscripts.co URL, e.g. my-podcast")
    argp.add_argument("--out", type=Path, default=Path("transcripts"), help="Output folder for <episode-slug>.txt files.")
    argp.add_argument("--pages", type=int, default=20, help="Max number of listing pages to scan.")
    argp.add_argument("--delay", type=float, default=1.5, help="Delay between requests in seconds.")
    argp.add_argument("--overwrite", action="store_true", help="Overwrite existing transcript files.")
    args = argp.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    episodes = episode_urls(args.podcast_slug, args.pages, args.delay)
    written = 0
    skipped = 0
    no_transcript = 0

    for index, (slug, title) in enumerate(episodes, start=1):
        path = args.out / f"{slug}.txt"
        if path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(episodes)}] skip existing {path.name}")
            continue
        url = f"{BASE_URL}/podcasts/{args.podcast_slug}/{slug}"
        html = fetch(url)
        segments = extract_transcript(html)
        if not segments:
            no_transcript += 1
            print(f"[{index}/{len(episodes)}] no transcript: {slug}")
            time.sleep(args.delay)
            continue
        path.write_text(segments_to_text(segments), encoding="utf-8")
        written += 1
        print(f"[{index}/{len(episodes)}] wrote {path.name} ({len(segments)} segments) — {title}")
        time.sleep(args.delay)

    print(
        f"done: episodes={len(episodes)} written={written} "
        f"skipped={skipped} no_transcript={no_transcript} out={args.out}"
    )


if __name__ == "__main__":
    main()
