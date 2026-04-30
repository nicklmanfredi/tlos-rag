from __future__ import annotations

import argparse
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


BASE_URL = "https://www.ancientfaith.com"
PODCAST_URL = f"{BASE_URL}/podcasts/lordofspirits/"
USER_AGENT = "tlos-rag transcript fetcher (+https://github.com/nicklmanfredi/tlos-rag)"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr = dict(attrs)
        href = attr.get("href")
        if href:
            self.links.append(href)


class TranscriptParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_transcript = False
        self.depth = 0
        self.parts: list[str] = []

    def _newline(self) -> None:
        if self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if not self.in_transcript:
            if tag == "div" and attr.get("id") == "transcript-reader":
                self.in_transcript = True
                self.depth = 1
            return
        self.depth += 1
        if tag in {"p", "div", "li", "center", "blockquote", "h1", "h2", "h3"}:
            self._newline()
        elif tag == "br":
            self._newline()

    def handle_endtag(self, tag: str) -> None:
        if not self.in_transcript:
            return
        if tag in {"p", "div", "li", "center", "blockquote", "h1", "h2", "h3"}:
            self._newline()
        self.depth -= 1
        if self.depth <= 0:
            self.in_transcript = False

    def handle_data(self, data: str) -> None:
        if self.in_transcript:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts).replace("\xa0", " ")
        lines: list[str] = []
        previous_blank = True
        for line in raw.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                if not previous_blank:
                    lines.append("")
                previous_blank = True
            else:
                lines.append(line)
                previous_blank = False
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines) + ("\n" if lines else "")


def fetch(url: str, retries: int = 3, delay: float = 1.0) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(delay * attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def episode_urls(pages: int, delay: float) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for page in range(1, pages + 1):
        page_url = f"{PODCAST_URL}?page={page}"
        html = fetch(page_url)
        parser = LinkParser()
        parser.feed(html)
        for href in parser.links:
            absolute = urljoin(BASE_URL, href)
            match = re.fullmatch(r"https://www\.ancientfaith\.com/podcasts/lordofspirits/([^/?#]+)/", absolute)
            if match and absolute not in seen:
                seen.add(absolute)
                urls.append(absolute)
        print(f"page {page}/{pages}: {len(urls)} episode URLs so far")
        time.sleep(delay)
    return urls


def extract_transcript(html: str) -> str:
    parser = TranscriptParser()
    parser.feed(html)
    return parser.text()


def main() -> None:
    argp = argparse.ArgumentParser(description="Fetch available Lord of Spirits transcripts from Ancient Faith.")
    argp.add_argument("--out", type=Path, default=Path("transcripts"), help="Output folder for <episode-slug>.txt files.")
    argp.add_argument("--pages", type=int, default=28, help="Number of podcast index pages to scan.")
    argp.add_argument("--delay", type=float, default=0.5, help="Delay between requests in seconds.")
    argp.add_argument("--overwrite", action="store_true", help="Overwrite existing transcript files.")
    args = argp.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    urls = episode_urls(args.pages, args.delay)
    written = 0
    skipped = 0
    no_transcript = 0

    for index, url in enumerate(urls, start=1):
        slug = url.rstrip("/").split("/")[-1]
        path = args.out / f"{slug}.txt"
        if path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(urls)}] skip existing {path.name}")
            continue
        html = fetch(url)
        transcript = extract_transcript(html)
        if not transcript.strip():
            no_transcript += 1
            print(f"[{index}/{len(urls)}] no transcript {slug}")
            time.sleep(args.delay)
            continue
        path.write_text(transcript, encoding="utf-8")
        written += 1
        print(f"[{index}/{len(urls)}] wrote {path.name}")
        time.sleep(args.delay)

    print(
        f"done: episode_urls={len(urls)} transcripts_written={written} "
        f"existing_skipped={skipped} no_transcript={no_transcript} out={args.out}"
    )


if __name__ == "__main__":
    main()

