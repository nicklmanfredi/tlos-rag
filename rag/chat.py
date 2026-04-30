from __future__ import annotations

import os
from pathlib import Path

from .chunking import format_time
from .config import HOSTS, Settings, host_display, host_slug
from .retrieval import retrieve


BASE_INSTRUCTIONS = """You are powering a transcript-grounded persona chat for The Lord of Spirits podcast.
Use the retrieved transcript excerpts for factual recall. When making factual claims from the show, cite the episode and timestamp inline.
If the excerpts do not support a factual answer, say so in the selected persona's voice instead of inventing details.
Stay in character for voice, cadence, humor, and emphasis, but do not claim to be the real person.
"""


def load_persona(settings: Settings, slug: str) -> str:
    path = settings.personas_dir / f"persona_{slug}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return f"No hand-edited persona exists yet for {host_display(slug)}. Use transcript excerpts conservatively."


def build_static_prompt(settings: Settings, mode: str, host: str | None) -> str:
    if mode == "host":
        slug = host_slug(host or "")
        return (
            BASE_INSTRUCTIONS
            + f"\nRespond only as {host_display(slug)}.\n\n"
            + f"<persona name=\"{host_display(slug)}\">\n{load_persona(settings, slug)}\n</persona>"
        )
    if mode == "show":
        return BASE_INSTRUCTIONS + "\nRespond as a merged show voice, prioritizing factual clarity over host-specific fidelity."
    personas = []
    for slug in HOSTS:
        personas.append(f"<persona name=\"{host_display(slug)}\">\n{load_persona(settings, slug)}\n</persona>")
    return (
        BASE_INSTRUCTIONS
        + "\nYou have both host personas. Usually pick the host whose voice best fits the question. "
        "Use a brief two-host exchange only when the question genuinely benefits from both perspectives. "
        "This is one Claude call; do not simulate independent agents.\n\n"
        + "\n\n".join(personas)
    )


def format_excerpts(chunks: list[dict]) -> str:
    lines = ["<transcript_excerpts>"]
    for i, chunk in enumerate(chunks, start=1):
        approx = " estimated" if chunk.get("timestamp_source") == "estimated" else ""
        citation = (
            f'{chunk["episode_title"]} '
            f'{format_time(chunk["start_seconds"])}-{format_time(chunk["end_seconds"])}{approx}; '
            f'primary={chunk["primary_speaker"]}; speakers={",".join(chunk.get("speakers", []))}'
        )
        lines.append(f'<excerpt id="{i}" citation="{citation}">\n{chunk["text"]}\n</excerpt>')
    lines.append("</transcript_excerpts>")
    return "\n".join(lines)


def answer_once(message: str, settings: Settings, mode: str = "both", host: str | None = None, stream: bool = True) -> str:
    chunks = retrieve(message, settings, host=host if mode == "host" else None)
    static_prompt = build_static_prompt(settings, mode, host)
    user_text = f"{format_excerpts(chunks)}\n\n<user_message>\n{message}\n</user_message>"

    if settings.chat_provider == "mock":
        response = "[mock chat provider]\n\nRetrieved excerpts:\n" + "\n".join(
            f'- {c["episode_title"]} {format_time(c["start_seconds"])}: {c["text"][:160]}...' for c in chunks[:3]
        )
        if stream:
            print(response)
        return response

    client = anthropic_client(settings)
    system = [{"type": "text", "text": static_prompt, "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "user", "content": user_text}]

    if stream:
        collected = []
        with client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=1800,
            system=system,
            messages=messages,
        ) as stream_obj:
            for text in stream_obj.text_stream:
                print(text, end="", flush=True)
                collected.append(text)
        print()
        return "".join(collected)

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1800,
        system=system,
        messages=messages,
    )
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


def anthropic_client(settings: Settings):
    if settings.chat_provider == "bedrock":
        from anthropic import AnthropicBedrock

        return AnthropicBedrock(aws_region=settings.aws_region)
    from anthropic import Anthropic

    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

