from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from .chunking import format_time
from .config import HOSTS, Settings, host_display, host_slug, project_root
from .retrieval import retrieve


BASE_INSTRUCTIONS = """You are powering a transcript-grounded persona chat for The Lord of Spirits podcast.
Use the transcript context for factual recall. Treat it as private research notes, not as something to discuss.
When making factual claims from the show, cite the episode and timestamp inline.
Do not say "the excerpts," "the retrieved context," "the provided transcript," "from these excerpts," or similar phrases.
If the transcript context does not support a factual answer, say "I can't verify that in the indexed transcripts" in the selected persona's voice instead of inventing details.
Stay in character for voice, cadence, humor, and emphasis, but do not claim to be the real person.
"""

TURN_LABELS = {
    "fr_andrew_stephen_damick": "Fr. Andrew",
    "fr_stephen_de_young": "Fr. Stephen",
}


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
        + "\nYou have both host personas. Write a brief transcript-style exchange when using this prompt directly.\n\n"
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


def answer_once(
    message: str,
    settings: Settings,
    mode: str = "both",
    host: str | None = None,
    stream: bool = True,
    turns: int = 4,
    search_backend: str = "rag",
    turn_words: int | None = None,
) -> str:
    chunks = retrieve(message, settings, host=host if mode == "host" else None, search_backend=search_backend)

    if mode == "both":
        return answer_both_turns(message, settings, chunks, turns=turns, stream=stream, turn_words=turn_words)

    static_prompt = build_static_prompt(settings, mode, host)
    user_text = f"{format_excerpts(chunks)}\n\n<user_message>\n{message}\n</user_message>"

    if settings.chat_provider == "mock":
        response = "[mock chat provider]\n\nRetrieved excerpts:\n" + "\n".join(
            f'- {c["episode_title"]} {format_time(c["start_seconds"])}: {c["text"][:160]}...' for c in chunks[:3]
        )
        if stream:
            print(response)
        return response

    return answer_with_provider(static_prompt, user_text, settings, stream=stream)


def answer_both_turns(
    message: str,
    settings: Settings,
    chunks: list[dict],
    turns: int,
    stream: bool = True,
    turn_words: int | None = None,
) -> str:
    turn_count = max(1, turns)
    host_order = tuple(HOSTS)
    transcript_turns: list[tuple[str, str]] = []
    context = format_excerpts(chunks)

    if settings.chat_provider == "mock":
        retrieved = "\n".join(
            f'- {c["episode_title"]} {format_time(c["start_seconds"])}: {c["text"][:120]}...' for c in chunks[:3]
        )
        for index in range(turn_count):
            slug = host_order[index % len(host_order)]
            text = f"[mock chat provider] Turn {index + 1}. Retrieved excerpts:\n{retrieved}"
            transcript_turns.append((slug, text))
        response = format_turn_transcript(transcript_turns)
        if stream:
            print(response)
        return response

    for index in range(turn_count):
        slug = host_order[index % len(host_order)]
        target_words = turn_word_target(slug, turn_words)
        static_prompt = build_turn_prompt(settings, slug)
        user_text = build_turn_user_text(
            context=context,
            original_message=message,
            transcript_turns=transcript_turns,
            speaker_slug=slug,
            is_final_turn=index == turn_count - 1,
            turn_words=target_words,
        )
        text = answer_with_provider(static_prompt, user_text, settings, stream=False)
        text = strip_speaker_label(text, slug).strip()
        transcript_turns.append((slug, text))
        if stream:
            print(f"{TURN_LABELS.get(slug, host_display(slug))}: {text}\n")

    return format_turn_transcript(transcript_turns)


def build_turn_prompt(settings: Settings, speaker_slug: str) -> str:
    display = host_display(speaker_slug)
    return (
        BASE_INSTRUCTIONS
        + "\nYou are generating exactly one turn in a multi-host transcript-style answer. "
        f"Respond only as {display}. Do not write dialogue for the other host. "
        "Do not include a speaker label; the CLI will add it. "
        "Keep the turn conversational and responsive to what has already been said.\n\n"
        + f"<persona name=\"{display}\">\n{load_persona(settings, speaker_slug)}\n</persona>"
    )


def build_turn_user_text(
    context: str,
    original_message: str,
    transcript_turns: list[tuple[str, str]],
    speaker_slug: str,
    is_final_turn: bool,
    turn_words: int | None = None,
) -> str:
    conversation = format_turn_transcript(transcript_turns) if transcript_turns else "(no host turns yet)"
    finish_instruction = (
        "This is the final planned turn, so bring the answer to a natural stopping point."
        if is_final_turn
        else "Leave room for the other host to continue the exchange."
    )
    length_instruction = (
        f"Keep this turn around {turn_words} words, with a natural podcast cadence and one focused point."
        if turn_words
        else "Keep this turn concise enough to feel like a real back-and-forth podcast exchange."
    )
    return (
        f"{context}\n\n"
        f"<user_message>\n{original_message}\n</user_message>\n\n"
        f"<conversation_so_far>\n{conversation}\n</conversation_so_far>\n\n"
        f"Write the next transcript turn as {host_display(speaker_slug)}. "
        "Answer the user's question through the conversation rather than meta-commenting on the format. "
        f"{length_instruction} "
        f"{finish_instruction}"
    )


def turn_word_target(speaker_slug: str, requested_words: int | None) -> int | None:
    if requested_words is None:
        return None
    if "andrew" in speaker_slug:
        return max(20, int(requested_words * 0.65))
    if "stephen" in speaker_slug:
        return max(30, int(requested_words * 1.25))
    return requested_words


def format_turn_transcript(turns: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"{TURN_LABELS.get(slug, host_display(slug))}: {text}" for slug, text in turns)


def strip_speaker_label(text: str, speaker_slug: str) -> str:
    label = TURN_LABELS.get(speaker_slug, host_display(speaker_slug))
    display = host_display(speaker_slug)
    pattern = rf"^\s*(?:{re.escape(label)}|{re.escape(display)})\s*:\s*"
    return re.sub(pattern, "", text, count=1)


def answer_with_provider(static_prompt: str, user_text: str, settings: Settings, stream: bool) -> str:
    if settings.chat_provider == "codex":
        return answer_with_codex(static_prompt, user_text, settings, stream=stream)

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


def answer_with_codex(static_prompt: str, user_text: str, settings: Settings, stream: bool) -> str:
    prompt = (
        static_prompt
        + "\n\nYou are being called non-interactively by a RAG CLI. "
        "Answer only the user's podcast question. Do not inspect files or run commands. "
        "Use only the transcript context below for factual claims, but never mention the context itself.\n\n"
        + user_text
    )
    cmd = [
        settings.codex_bin,
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--cd",
        str(project_root()),
        "-",
    ]
    if settings.codex_model:
        cmd[2:2] = ["--model", settings.codex_model]
    if settings.codex_oss:
        cmd.insert(2, "--oss")
    if settings.codex_local_provider:
        cmd[2:2] = ["--local-provider", settings.codex_local_provider]

    proc = subprocess.run(cmd, input=prompt, text=True, capture_output=True, check=False)
    output = proc.stdout.strip()
    if proc.returncode != 0:
        error = proc.stderr.strip() or output
        raise RuntimeError(f"codex exec failed with exit code {proc.returncode}: {error}")
    if stream:
        print(output)
    return output
