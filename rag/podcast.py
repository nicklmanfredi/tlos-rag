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

SPEAKER_TO_SETTING = {
    TURN_LABELS["fr_andrew_stephen_damick"]: "andrew",
    TURN_LABELS["fr_stephen_de_young"]: "stephen",
}

VOICE_INSTRUCTIONS = {
    "andrew": (
        "Use a generic synthetic American male-presenting voice. Sound conversational, lightly nerdy, "
        "clear, and friendly, with a mild Midwestern public-radio cadence. Do not imitate any real person."
    ),
    "stephen": (
        "Use a different generic synthetic American male-presenting voice. Sound dry, bookish, precise, "
        "and lightly amused, with a mild Midwestern seminar-room cadence. Do not imitate any real person."
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
        segment_paths = synthesize_segments(segments, settings, Path(tmp))
        stitch_wav(segment_paths, out_path)

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
        segment_paths = synthesize_segments(segments, settings, Path(tmp))
        stitch_wav(segment_paths, out_path)

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


def synthesize_segments(segments: list[SpeechSegment], settings: Settings, tmp_dir: Path) -> list[Path]:
    client = OpenAI()
    paths = []
    audio_index = 0
    for segment in segments:
        voice_key = SPEAKER_TO_SETTING[segment.speaker]
        voice = settings.tts_voice_andrew if voice_key == "andrew" else settings.tts_voice_stephen
        for text in split_for_tts(segment.text):
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
            paths.append(path)
    return paths


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


def stitch_wav(segment_paths: list[Path], out_path: Path, silence_ms: int = 450) -> None:
    audio_format: tuple[int, int, int, str, str] | None = None
    frames: list[bytes] = []
    silence_frames = b""

    for path in segment_paths:
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
                silence_count = int(src.getframerate() * silence_ms / 1000)
                silence_frames = b"\x00" * silence_count * src.getnchannels() * src.getsampwidth()
            elif current_format != audio_format:
                raise ValueError("TTS returned WAV segments with incompatible audio parameters.")
            frames.append(src.readframes(src.getnframes()))

    if audio_format is None:
        raise ValueError("No WAV segments were generated.")

    channels, sample_width, frame_rate, comp_type, comp_name = audio_format
    with wave.open(str(out_path), "wb") as dest:
        dest.setnchannels(channels)
        dest.setsampwidth(sample_width)
        dest.setframerate(frame_rate)
        dest.setcomptype(comp_type, comp_name)
        for index, frame_data in enumerate(frames):
            if index:
                dest.writeframes(silence_frames)
            dest.writeframes(frame_data)
