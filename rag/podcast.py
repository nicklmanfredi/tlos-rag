from __future__ import annotations

import re
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from .chat import TURN_LABELS, answer_once
from .config import Settings


MAX_TTS_CHARS = 3500
TURN_PAUSE_MS = 550
LAUGHTER_PAUSE_MS = 850
BEAT_PAUSE_MS = 500

SPEAKER_TO_SETTING = {
    TURN_LABELS["fr_andrew_stephen_damick"]: "andrew",
    TURN_LABELS["fr_stephen_de_young"]: "stephen",
}

VOICE_INSTRUCTIONS = {
    "andrew": (
        "Use a generic synthetic American male-presenting voice. Sound conversational, lightly nerdy, "
        "clear, and friendly, with a mild Midwestern public-radio cadence. React to jokes and qualifications "
        "as live speech, not as narration. Do not imitate any real person."
    ),
    "stephen": (
        "Use a different generic synthetic American male-presenting voice. Sound dry, bookish, precise, "
        "and lightly amused, with a mild Midwestern seminar-room cadence. Let deadpan lines land plainly, "
        "with small pauses where the syntax turns. Do not imitate any real person."
    ),
}


@dataclass(frozen=True)
class PodcastResult:
    out_path: Path
    script_path: Path
    segments: int


@dataclass(frozen=True)
class SpeechSegment:
    speaker: str
    text: str


@dataclass(frozen=True)
class AudioClip:
    path: Path | None = None
    silence_ms: int = 0


def generate_podcast(
    message: str,
    settings: Settings,
    out_path: Path,
    turns: int = 4,
    search_backend: str = "rag",
    script_path: Path | None = None,
    turn_words: int | None = None,
) -> PodcastResult:
    script = answer_once(
        message,
        settings,
        mode="both",
        stream=False,
        turns=turns,
        search_backend=search_backend,
        turn_words=turn_words,
    )
    segments = parse_speech_segments(script)
    if not segments:
        raise ValueError("No Fr. Andrew/Fr. Stephen transcript turns were found in the generated script.")

    out_path = out_path.expanduser()
    script_path = (script_path or out_path.with_suffix(".txt")).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script, encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        clips = synthesize_segments(segments, settings, Path(tmp))
        stitch_wav(clips, out_path)

    return PodcastResult(out_path=out_path, script_path=script_path, segments=len(segments))


def synthesize_podcast_from_script(script_path: Path, settings: Settings, out_path: Path) -> PodcastResult:
    script_path = script_path.expanduser()
    out_path = out_path.expanduser()
    script = script_path.read_text(encoding="utf-8")
    segments = parse_speech_segments(script)
    if not segments:
        raise ValueError("No Fr. Andrew/Fr. Stephen transcript turns were found in the script file.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        clips = synthesize_segments(segments, settings, Path(tmp))
        stitch_wav(clips, out_path)

    return PodcastResult(out_path=out_path, script_path=script_path, segments=len(segments))


def parse_speech_segments(script: str) -> list[SpeechSegment]:
    segments: list[SpeechSegment] = []
    current_speaker: str | None = None
    current_lines: list[str] = []
    label_re = re.compile(r"^\s*(Fr\. Andrew|Fr\. Stephen)\s*:\s*(.*)$")

    for line in script.splitlines():
        match = label_re.match(line)
        if match:
            if current_speaker and current_lines:
                segments.append(SpeechSegment(current_speaker, "\n".join(current_lines).strip()))
            current_speaker = match.group(1)
            current_lines = [match.group(2).strip()] if match.group(2).strip() else []
            continue
        if current_speaker:
            current_lines.append(line)

    if current_speaker and current_lines:
        segments.append(SpeechSegment(current_speaker, "\n".join(current_lines).strip()))
    return [segment for segment in segments if segment.text]


def synthesize_segments(segments: list[SpeechSegment], settings: Settings, tmp_dir: Path) -> list[AudioClip]:
    client = OpenAI()
    clips: list[AudioClip] = []
    audio_index = 0
    for segment_index, segment in enumerate(segments):
        if segment_index:
            clips.append(AudioClip(silence_ms=TURN_PAUSE_MS))
        voice_key = SPEAKER_TO_SETTING[segment.speaker]
        voice = settings.tts_voice_andrew if voice_key == "andrew" else settings.tts_voice_stephen
        for part in split_timing_parts(segment.text):
            if isinstance(part, int):
                clips.append(AudioClip(silence_ms=part))
                continue
            for text in split_for_tts(part):
                audio_index += 1
                path = tmp_dir / f"segment_{audio_index:03d}.wav"
                response = client.audio.speech.create(
                    model=settings.tts_model,
                    voice=voice,
                    input=text,
                    instructions=VOICE_INSTRUCTIONS[voice_key],
                    response_format="wav",
                )
                response.write_to_file(path)
                clips.append(AudioClip(path=path))
    return clips


def split_timing_parts(text: str) -> list[str | int]:
    parts: list[str | int] = []
    cursor = 0
    marker_re = re.compile(r"\[(?:laughter|laughs?|pause|beat)\]", re.IGNORECASE)
    for match in marker_re.finditer(text):
        before = normalize_tts_text(text[cursor : match.start()])
        if before:
            parts.append(before)
        marker = match.group(0).lower()
        parts.append(LAUGHTER_PAUSE_MS if "laugh" in marker else BEAT_PAUSE_MS)
        cursor = match.end()

    remainder = normalize_tts_text(text[cursor:])
    if remainder:
        parts.append(remainder)
    return compact_timing_parts(parts)


def normalize_tts_text(text: str) -> str:
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_timing_parts(parts: list[str | int]) -> list[str | int]:
    compact: list[str | int] = []
    for part in parts:
        if isinstance(part, int) and compact and isinstance(compact[-1], int):
            compact[-1] += part
        elif part:
            compact.append(part)
    return compact


def split_for_tts(text: str, max_chars: int = MAX_TTS_CHARS) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    for part in parts:
        if not part:
            continue
        if len(part) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(part[i : i + max_chars] for i in range(0, len(part), max_chars))
            continue
        candidate = f"{current} {part}".strip()
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            current = part
        else:
            current = candidate
    if current:
        chunks.append(current.strip())
    return chunks


def stitch_wav(clips: list[AudioClip], out_path: Path) -> None:
    audio_format: tuple[int, int, int, str, str] | None = None
    frame_items: list[bytes | int] = []

    for clip in clips:
        if clip.path is None:
            if clip.silence_ms > 0:
                frame_items.append(clip.silence_ms)
            continue
        path = clip.path
        with wave.open(str(path), "rb") as src:
            current_format = (
                src.getnchannels(),
                src.getsampwidth(),
                src.getframerate(),
                src.getcomptype(),
                src.getcompname(),
            )
            if audio_format is None:
                audio_format = current_format
            elif current_format != audio_format:
                raise ValueError("TTS returned WAV segments with incompatible audio parameters.")
            frame_items.append(src.readframes(src.getnframes()))

    if audio_format is None:
        raise ValueError("No WAV segments were generated.")

    channels, sample_width, frame_rate, comp_type, comp_name = audio_format
    with wave.open(str(out_path), "wb") as dest:
        dest.setnchannels(channels)
        dest.setsampwidth(sample_width)
        dest.setframerate(frame_rate)
        dest.setcomptype(comp_type, comp_name)
        for item in frame_items:
            if isinstance(item, int):
                silence_count = int(frame_rate * item / 1000)
                dest.writeframes(b"\x00" * silence_count * channels * sample_width)
            else:
                dest.writeframes(item)
