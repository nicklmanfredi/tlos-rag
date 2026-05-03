from __future__ import annotations

import argparse
import sys
import termios
import tty
from pathlib import Path

from .chat import answer_once
from .chunking import format_time
from .config import settings
from .ingest import ingest_folder
from .persona_bootstrap import bootstrap_persona
from .retrieval import retrieve


def main() -> None:
    parser = argparse.ArgumentParser(prog="podcast-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Build or refresh the local transcript index.")
    ingest.add_argument("path", type=Path)

    boot = sub.add_parser("bootstrap-persona", help="Generate a persona guide for one host.")
    boot.add_argument("--host", required=True)

    search = sub.add_parser("search", help="Debug retrieval without calling Claude.")
    search.add_argument("query")
    search.add_argument("--host")
    search.add_argument("--limit", type=int, default=8)
    search.add_argument("--search-backend", choices=["rag", "agentic", "text", "both"], default="rag")

    chat = sub.add_parser("chat", help="Interactive chat REPL.")
    chat.add_argument("--host")
    chat.add_argument("--both", action="store_true", default=False)
    chat.add_argument("--show", action="store_true", default=False)
    chat.add_argument("--message")
    chat.add_argument("--turns", type=int, default=4, help="Number of alternating host turns when using --both.")
    chat.add_argument("--search-backend", choices=["rag", "agentic", "text", "both"], default="rag")

    args = parser.parse_args()
    cfg = settings()

    if args.command == "ingest":
        result = ingest_folder(args.path.expanduser(), cfg)
        print(result)
    elif args.command == "bootstrap-persona":
        print(bootstrap_persona(args.host, cfg))
    elif args.command == "search":
        for backend in selected_backends(args.search_backend):
            if args.search_backend == "both":
                print(f"\n=== {backend.upper()} SEARCH ===")
            print_search_results(args.query, cfg, args.host, args.limit, backend)
    elif args.command == "chat":
        mode = "show" if args.show else "host" if args.host else "both"
        if args.message:
            for backend in selected_backends(args.search_backend):
                if args.search_backend == "both":
                    print(f"\n=== {backend.upper()} CHAT ===")
                answer_once(
                    args.message,
                    cfg,
                    mode=mode,
                    host=args.host,
                    stream=True,
                    turns=args.turns,
                    search_backend=backend,
                )
            return
        if args.search_backend == "both":
            raise SystemExit("--search-backend both is only supported with --message; use rag or text for the REPL.")
        print("Press Esc to exit.")
        while True:
            message = read_repl_line("\nYou: ")
            if message is None:
                print()
                break
            message = message.strip()
            if not message:
                continue
            print()
            answer_once(message, cfg, mode=mode, host=args.host, stream=True, turns=args.turns, search_backend=args.search_backend)


def selected_backends(value: str) -> list[str]:
    return ["rag", "agentic"] if value == "both" else [value]


def print_search_results(query: str, cfg, host: str | None, limit: int, backend: str) -> None:
    for i, row in enumerate(retrieve(query, cfg, host=host, final_k=limit, search_backend=backend), start=1):
        approx = " estimated" if row.get("timestamp_source") == "estimated" else ""
        print(f"[{i}] {row['episode_title']} {format_time(row['start_seconds'])}-{format_time(row['end_seconds'])}{approx}")
        print(f"    primary={row['primary_speaker']} speakers={','.join(row.get('speakers', []))}")
        print(f"    {row['text'][:700].replace(chr(10), ' ')}")


def read_repl_line(prompt: str) -> str | None:
    """Read one terminal line, returning None if Escape is pressed."""
    if not sys.stdin.isatty():
        try:
            return input(prompt)
        except EOFError:
            return None

    sys.stdout.write(prompt)
    sys.stdout.flush()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    chars: list[str] = []
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                return None
            if ch in {"\r", "\n"}:
                sys.stdout.write("\n")
                sys.stdout.flush()
                return "".join(chars)
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch == "\x04":
                return None
            if ch in {"\x7f", "\b"}:
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if ch.isprintable():
                chars.append(ch)
                sys.stdout.write(ch)
                sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
