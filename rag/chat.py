from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from .chunking import format_time
from .config import HOSTS, Settings, host_display, host_slug, project_root
from .retrieval import retrieve


CHAT_RETRIEVAL_K = 12
MAX_EMPTY_TURN_RETRIES = 2

FORBIDDEN_PUBLIC_META_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:the|these|those)?\s*(?:transcript|transcripts|excerpt|excerpts|indexed transcripts)\b",
        r"\b(?:retrieved|provided|supplied)\s+context\b",
        r"\b(?:source material|source notes|retrieval|search results|indexed corpus)\b",
        r"\b(?:i|we)\s+(?:can't|cannot|couldn't|could not|don't|do not|didn't|did not)\s+(?:find|verify|see|locate)\b",
        r"\b(?:not|isn't|wasn't)\s+(?:in|supported by|found in)\s+(?:the\s+)?(?:transcripts|context|excerpts|index)\b",
    )
]

BASE_INSTRUCTIONS = """You are powering a source-grounded persona chat for The Lord of Spirits podcast.
Use the supplied source material for factual recall. Treat it as private research notes, not as something to discuss.
When making factual claims from the show, cite the episode and timestamp inline.
Never mention transcripts, excerpts, retrieved context, indexed material, search, or evidence-gathering in the public answer.
If the source material does not support a factual answer, do not call attention to the gap; answer only with the closest directly supported point, narrow the claim, or omit the unsupported detail.
Stay in character for voice, cadence, humor, and emphasis, but do not claim to be the real person.
Reason through the actual idea before making a joke. Humor should come from a concrete contrast, modern category mistake, or absurd implication already present in the answer.
When a joke or dry aside needs timing, use [beat] before the turn or [Laughter] after it. Use these markers sparingly.
"""

BRIEFING_INSTRUCTIONS = """You prepare private notes for a source-grounded podcast answer.
Use only the source items and the user's question.
Do not imitate a host voice.
Do not write the final answer.
Go deeper than a list of hits. Build the notes a writer would need before recording a serious but conversational answer.
Produce a structured briefing with:
- Central thesis: the main answer in one or two sentences.
- Concept map: key terms, distinctions, and background assumptions needed to explain the idea clearly.
- Grounded claims: 5 to 9 claims, each with episode/timestamp citations and the exact role it should play in the answer.
- Connections: how the claims fit together, including causal, biblical, liturgical, historical, or polemical relationships.
- Tensions and limits: real qualifications, ambiguities, or places to avoid overclaiming.
- Answer path: a concise sequence of moves the final answer should make.
- Natural voice moments: one or two comic or conversational angles rooted in the material, not generic jokes.
This briefing is private. It may mention transcript evidence internally, but the later public answer must not mention transcripts, excerpts, search, retrieval, or missing evidence.
"""

TURN_LABELS = {
    "fr_andrew_stephen_damick": "Fr. Andrew",
    "fr_stephen_de_young": "Fr. Stephen",
}


def load_persona(settings: Settings, slug: str) -> str:
    path = settings.personas_dir / f"persona_{slug}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return f"No hand-edited persona exists yet for {host_display(slug)}. Use source samples conservatively."


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
        + "\nYou have both host personas. Write a brief podcast-style exchange when using this prompt directly.\n\n"
        + "\n\n".join(personas)
    )


def format_excerpts(chunks: list[dict]) -> str:
    lines = ["<source_material>"]
    for i, chunk in enumerate(chunks, start=1):
        approx = " estimated" if chunk.get("timestamp_source") == "estimated" else ""
        citation = (
            f'{chunk["episode_title"]} '
            f'{format_time(chunk["start_seconds"])}-{format_time(chunk["end_seconds"])}{approx}; '
            f'primary={chunk["primary_speaker"]}; speakers={",".join(chunk.get("speakers", []))}'
        )
        lines.append(f'<source id="{i}" citation="{citation}">\n{chunk["text"]}\n</source>')
    lines.append("</source_material>")
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
    chunks = retrieve(
        message,
        settings,
        host=host if mode == "host" else None,
        final_k=CHAT_RETRIEVAL_K,
        search_backend=search_backend,
    )

    if mode == "both":
        return answer_both_turns(message, settings, chunks, turns=turns, stream=stream, turn_words=turn_words)

    static_prompt = build_static_prompt(settings, mode, host)
    context = format_excerpts(chunks)
    briefing = synthesize_evidence_brief(message, settings, context)
    user_text = build_answer_user_text(context, briefing, message)

    if settings.chat_provider == "mock":
        response = mock_public_answer(chunks)
        if stream:
            print(response)
        return response

    return answer_public_with_provider(static_prompt, user_text, settings, stream=stream)


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
        sample_answer = mock_public_answer(chunks)
        for index in range(turn_count):
            slug = host_order[index % len(host_order)]
            text = f"[mock chat provider] Turn {index + 1}. {sample_answer}"
            transcript_turns.append((slug, text))
        response = format_turn_transcript(transcript_turns)
        if stream:
            print(response)
        return response

    briefing = synthesize_evidence_brief(message, settings, context)

    index = 0
    attempts = 0
    while len(transcript_turns) < turn_count and attempts < turn_count + (turn_count * MAX_EMPTY_TURN_RETRIES):
        slug = host_order[index % len(host_order)]
        target_words = turn_word_target(slug, turn_words)
        static_prompt = build_turn_prompt(settings, slug)
        user_text = build_turn_user_text(
            context=context,
            briefing=briefing,
            original_message=message,
            transcript_turns=transcript_turns,
            speaker_slug=slug,
            is_final_turn=len(transcript_turns) == turn_count - 1,
            turn_words=target_words,
        )
        attempts += 1
        text = answer_public_with_provider(static_prompt, user_text, settings, stream=False)
        text = strip_speaker_label(text, slug).strip()
        if not text:
            continue
        transcript_turns.append((slug, text))
        index += 1
        if stream:
            print(f"{TURN_LABELS.get(slug, host_display(slug))}: {text}\n")

    if len(transcript_turns) < turn_count:
        raise RuntimeError(f"Generated only {len(transcript_turns)} non-empty turns out of requested {turn_count}.")

    return format_turn_transcript(transcript_turns)


def build_turn_prompt(settings: Settings, speaker_slug: str) -> str:
    display = host_display(speaker_slug)
    return (
        BASE_INSTRUCTIONS
        + "\nYou are generating exactly one turn in a multi-host podcast-style answer. "
        f"Respond only as {display}. Do not write dialogue for the other host. "
        "Do not include a speaker label; the CLI will add it. "
        "Keep the turn conversational and responsive to what has already been said. "
        "Do not summarize research notes. Make a point, react to the previous turn, and move the conversation. "
        "Use at most one joke or aside per turn; it must clarify the point rather than decorate it. "
        "For audio timing, place [beat] before a deadpan line or [Laughter] after a genuinely funny aside when needed.\n\n"
        + f"<persona name=\"{display}\">\n{load_persona(settings, speaker_slug)}\n</persona>"
    )


def build_turn_user_text(
    context: str,
    briefing: str,
    original_message: str,
    transcript_turns: list[tuple[str, str]],
    speaker_slug: str,
    is_final_turn: bool,
    turn_words: int | None = None,
) -> str:
    conversation = format_turn_transcript(transcript_turns) if transcript_turns else "(no host turns yet)"
    opening_instruction = (
        "This is the opening turn. Start cleanly for the listener: name the topic, frame the user's question, "
        "and set up the conversation. Do not begin with agreement words like 'Right,' 'Yeah,' or 'Exactly,' "
        "because nothing has been said yet."
        if not transcript_turns
        else "Respond naturally to the conversation so far."
    )
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
        f"<private_evidence_briefing>\n{briefing}\n</private_evidence_briefing>\n\n"
        f"<user_message>\n{original_message}\n</user_message>\n\n"
        f"<conversation_so_far>\n{conversation}\n</conversation_so_far>\n\n"
        f"Write the next podcast turn as {host_display(speaker_slug)}. "
        f"{opening_instruction} "
        "Answer the user's question through the conversation rather than meta-commenting on the format. "
        "Advance the argument from the private briefing in your own words. "
        "Do not mention transcripts, excerpts, search, retrieval, source material, or whether something could be found. "
        "If you use humor, make it sound like a quick live aside prompted by the substance, not a written punchline. "
        "Use [beat] only where a short pause improves the timing; use [Laughter] only after the line that earns it. "
        f"{length_instruction} "
        f"{finish_instruction}"
    )


def build_answer_user_text(context: str, briefing: str, message: str) -> str:
    return (
        f"{context}\n\n"
        f"<private_evidence_briefing>\n{briefing}\n</private_evidence_briefing>\n\n"
        f"<user_message>\n{message}\n</user_message>\n\n"
        "Answer the user's question naturally. Use the private briefing to keep the argument coherent, "
        "but do not mention the briefing, transcripts, excerpts, search, retrieval, source material, or whether something could be found."
    )


def synthesize_evidence_brief(message: str, settings: Settings, context: str) -> str:
    if settings.chat_provider == "mock":
        return "(mock evidence briefing)"
    user_text = f"{context}\n\n<user_message>\n{message}\n</user_message>"
    return answer_with_provider(
        BRIEFING_INSTRUCTIONS,
        user_text,
        settings,
        stream=False,
        codex_task="Write only the requested private evidence briefing for the later answer.",
        max_tokens=1600,
    ).strip()


def mock_public_answer(chunks: list[dict]) -> str:
    if not chunks:
        return "The closest supported answer is narrow, so this mock response stays general instead of filling in details."
    first = chunks[0]
    return (
        "[mock chat provider] "
        f"A grounded answer would build from {first['episode_title']} "
        f"{format_time(first['start_seconds'])}, keeping the claim tied to that part of the show."
    )


def answer_public_with_provider(static_prompt: str, user_text: str, settings: Settings, stream: bool) -> str:
    text = answer_with_provider(static_prompt, user_text, settings, stream=False)
    text = clean_public_answer(text).strip()
    if stream:
        print(text)
    return text


def clean_public_answer(text: str) -> str:
    """Remove public-facing retrieval chatter if the model leaks it."""
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        if is_public_meta_line(stripped):
            continue
        cleaned_lines.append(remove_inline_public_meta(line))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines))


def is_public_meta_line(text: str) -> bool:
    return any(pattern.search(text) for pattern in FORBIDDEN_PUBLIC_META_PATTERNS)


def remove_inline_public_meta(text: str) -> str:
    replacements = {
        "from the transcripts": "",
        "from these transcripts": "",
        "in the transcripts": "",
        "from the excerpts": "",
        "from these excerpts": "",
        "in the excerpts": "",
        "from the retrieved context": "",
        "in the retrieved context": "",
        "from the context": "",
        "in the context": "",
    }
    cleaned = text
    for phrase, replacement in replacements.items():
        cleaned = re.sub(re.escape(phrase), replacement, cleaned, flags=re.IGNORECASE)
    return re.sub(r" {2,}", " ", cleaned).strip()


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


def answer_with_provider(
    static_prompt: str,
    user_text: str,
    settings: Settings,
    stream: bool,
    codex_task: str | None = None,
    max_tokens: int = 1800,
) -> str:
    if settings.chat_provider == "codex":
        return answer_with_codex(static_prompt, user_text, settings, stream=stream, task=codex_task)

    client = anthropic_client(settings)
    system = [{"type": "text", "text": static_prompt, "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "user", "content": user_text}]

    if stream:
        collected = []
        with client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=max_tokens,
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
        max_tokens=max_tokens,
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


def answer_with_codex(static_prompt: str, user_text: str, settings: Settings, stream: bool, task: str | None = None) -> str:
    task = task or "Answer only the user's podcast question."
    prompt = (
        static_prompt
        + "\n\nYou are being called non-interactively by a RAG CLI. "
        f"{task} Do not inspect files or run commands. "
        "Use only the source material below for factual claims, but never mention the source material itself.\n\n"
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
