from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from .chat import TURN_LABELS, answer_once
from .config import Settings


MAX_TTS_CHARS = 3500
MAX_ELEVENLABS_CHARS = 2800
TURN_PAUSE_MS = 550
LAUGHTER_PAUSE_MS = 850
BEAT_PAUSE_MS = 500
CLIP_FADE_MS = 8
TARGET_PEAK = 0.89

SPEAKER_TO_SETTING = {
    TURN_LABELS["fr_andrew_stephen_damick"]: "andrew",
    TURN_LABELS["fr_stephen_de_young"]: "stephen",
}

VOICE_INSTRUCTIONS = {
    "andrew": (
        "Use a generic synthetic American male-presenting voice. Sound conversational, lightly nerdy, "
        "clear, and friendly, with a mild public-radio cadence. Vary the speed naturally: quicker for setup, "
        "slower for careful distinctions. React to jokes and qualifications as live speech, not narration. "
        "Do not imitate any real person."
    ),
    "stephen": (
        "Use a different generic synthetic American male-presenting voice. Sound dry, bookish, precise, "
        "and lightly amused, with a seminar-room cadence. Let deadpan lines land plainly, and slow slightly "
        "around technical terms, names, and ancient sources. Do not imitate any real person."
    ),
}


@dataclass(frozen=True)
class PodcastResult:
    out_path: Path
    script_path: Path
    segments: int
    performance_script_path: Path | None = None


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
    performance_script: bool = True,
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

    spoken_script_path: Path | None = None
    synthesis_script = script
    if performance_script:
        synthesis_script = prepare_performance_script(script)
        spoken_script_path = performance_script_path_for(script_path)
        spoken_script_path.write_text(synthesis_script, encoding="utf-8")

    synthesis_segments = parse_speech_segments(synthesis_script)
    if not synthesis_segments:
        raise ValueError("No Fr. Andrew/Fr. Stephen transcript turns were found in the spoken script.")

    with tempfile.TemporaryDirectory() as tmp:
        clips = synthesize_segments(synthesis_segments, settings, Path(tmp))
        stitch_wav(clips, out_path, master_audio=settings.tts_master_audio)

    return PodcastResult(
        out_path=out_path,
        script_path=script_path,
        segments=len(synthesis_segments),
        performance_script_path=spoken_script_path,
    )


def synthesize_podcast_from_script(
    script_path: Path,
    settings: Settings,
    out_path: Path,
    performance_script: bool = True,
) -> PodcastResult:
    script_path = script_path.expanduser()
    out_path = out_path.expanduser()
    script = script_path.read_text(encoding="utf-8")
    segments = parse_speech_segments(script)
    if not segments:
        raise ValueError("No Fr. Andrew/Fr. Stephen transcript turns were found in the script file.")

    spoken_script_path: Path | None = None
    synthesis_script = script
    if performance_script:
        synthesis_script = prepare_performance_script(script)
        spoken_script_path = performance_script_path_for(out_path.with_suffix(".txt"))
        spoken_script_path.parent.mkdir(parents=True, exist_ok=True)
        spoken_script_path.write_text(synthesis_script, encoding="utf-8")

    synthesis_segments = parse_speech_segments(synthesis_script)
    if not synthesis_segments:
        raise ValueError("No Fr. Andrew/Fr. Stephen transcript turns were found in the spoken script.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        clips = synthesize_segments(synthesis_segments, settings, Path(tmp))
        stitch_wav(clips, out_path, master_audio=settings.tts_master_audio)

    return PodcastResult(
        out_path=out_path,
        script_path=script_path,
        segments=len(synthesis_segments),
        performance_script_path=spoken_script_path,
    )


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


def prepare_performance_script(script: str) -> str:
    segments = parse_speech_segments(script)
    lines: list[str] = []
    for segment in segments:
        text = clean_spoken_text(segment.text)
        if text:
            lines.append(f"{segment.speaker}: {text}")
    return "\n\n".join(lines).strip() + "\n"


def performance_script_path_for(script_path: Path) -> Path:
    if script_path.suffix:
        return script_path.with_name(f"{script_path.stem}.spoken{script_path.suffix}")
    return script_path.with_name(f"{script_path.name}.spoken.txt")


def clean_spoken_text(text: str) -> str:
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = strip_audio_hostile_citations(text)
    text = text.replace("—", ", ")
    text = text.replace("–", "-")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([.!?])\s+(?=[A-Z])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_audio_hostile_citations(text: str) -> str:
    # Keep ordinary asides, but remove transcript citations that sound bad when read aloud.
    citation_re = re.compile(r"\s*\(([^()]*(?:\d{1,2}:\d{2}|estimated|transcript|episode)[^()]*)\)", re.IGNORECASE)
    previous = None
    while previous != text:
        previous = text
        text = citation_re.sub("", text)
    text = re.sub(r"\s+\[[0-9,\s]+\]", "", text)
    text = re.sub(r"\s+\(\s*\)", "", text)
    return text


def synthesize_segments(segments: list[SpeechSegment], settings: Settings, tmp_dir: Path) -> list[AudioClip]:
    provider = settings.tts_provider.lower()
    if provider not in {"openai", "elevenlabs"}:
        raise ValueError("TTS_PROVIDER must be 'openai' or 'elevenlabs'.")
    client = OpenAI() if provider == "openai" else None
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
            max_chars = MAX_ELEVENLABS_CHARS if provider == "elevenlabs" else MAX_TTS_CHARS
            for text in split_for_tts(part, max_chars=max_chars):
                audio_index += 1
                path = tmp_dir / f"segment_{audio_index:03d}.wav"
                if provider == "openai":
                    assert client is not None
                    response = client.audio.speech.create(
                        model=settings.tts_model,
                        voice=voice,
                        input=text,
                        instructions=VOICE_INSTRUCTIONS[voice_key],
                        response_format="wav",
                    )
                    response.write_to_file(path)
                else:
                    synthesize_elevenlabs_wav(text, voice_key, settings, path)
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


def synthesize_elevenlabs_wav(text: str, voice_key: str, settings: Settings, path: Path) -> None:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY is required when TTS_PROVIDER=elevenlabs.")

    voice_id = settings.elevenlabs_voice_andrew if voice_key == "andrew" else settings.elevenlabs_voice_stephen
    if not voice_id:
        env_name = "ELEVENLABS_VOICE_ANDREW" if voice_key == "andrew" else "ELEVENLABS_VOICE_STEPHEN"
        raise ValueError(f"{env_name} is required when TTS_PROVIDER=elevenlabs.")

    output_format = "pcm_24000"
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{urllib.parse.quote(voice_id)}"
        f"?output_format={output_format}"
    )
    payload = {
        "text": text,
        "model_id": settings.elevenlabs_tts_model,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/octet-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            pcm = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ElevenLabs TTS failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"ElevenLabs TTS request failed: {exc.reason}") from exc
    write_pcm16_wav(path, pcm, frame_rate=24000, channels=1)


def write_pcm16_wav(path: Path, pcm: bytes, frame_rate: int, channels: int) -> None:
    with wave.open(str(path), "wb") as dest:
        dest.setnchannels(channels)
        dest.setsampwidth(2)
        dest.setframerate(frame_rate)
        dest.writeframes(pcm)


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


def stitch_wav(clips: list[AudioClip], out_path: Path, master_audio: bool = True) -> None:
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
            frames = src.readframes(src.getnframes())
            if master_audio:
                frames = fade_clip_edges(frames, current_format, fade_ms=CLIP_FADE_MS)
            frame_items.append(frames)

    if audio_format is None:
        raise ValueError("No WAV segments were generated.")

    if master_audio:
        frame_items = master_frame_items(frame_items, audio_format)

    channels, sample_width, frame_rate, comp_type, comp_name = audio_format
    with wave.open(str(out_path), "wb") as dest:
        dest.setnchannels(channels)
        dest.setsampwidth(sample_width)
        dest.setframerate(frame_rate)
        dest.setcomptype(comp_type, comp_name)
        for item in frame_items:
            if isinstance(item, int):
                dest.writeframes(make_pause_frames(item, audio_format, room_tone=master_audio))
            else:
                dest.writeframes(item)


def fade_clip_edges(frames: bytes, audio_format: tuple[int, int, int, str, str], fade_ms: int) -> bytes:
    channels, sample_width, frame_rate, _comp_type, _comp_name = audio_format
    if sample_width != 2 or not frames:
        return frames

    frame_size = channels * sample_width
    frame_count = len(frames) // frame_size
    fade_frames = min(int(frame_rate * fade_ms / 1000), frame_count // 2)
    if fade_frames <= 1:
        return frames

    data = bytearray(frames)
    for frame_index in range(fade_frames):
        start_factor = (frame_index + 1) / fade_frames
        end_factor = (fade_frames - frame_index) / fade_frames
        scale_frame_pcm16(data, frame_index, channels, sample_width, start_factor)
        scale_frame_pcm16(data, frame_count - frame_index - 1, channels, sample_width, end_factor)
    return bytes(data)


def master_frame_items(frame_items: list[bytes | int], audio_format: tuple[int, int, int, str, str]) -> list[bytes | int]:
    channels, sample_width, _frame_rate, _comp_type, _comp_name = audio_format
    if sample_width != 2:
        return frame_items

    peak = 0
    for item in frame_items:
        if isinstance(item, bytes):
            peak = max(peak, pcm16_peak(item))
    if peak == 0:
        return frame_items

    target = int(32767 * TARGET_PEAK)
    factor = target / peak
    if 0.98 <= factor <= 1.02:
        return frame_items

    mastered: list[bytes | int] = []
    for item in frame_items:
        if isinstance(item, bytes):
            mastered.append(scale_pcm16(item, channels, sample_width, factor))
        else:
            mastered.append(item)
    return mastered


def make_pause_frames(silence_ms: int, audio_format: tuple[int, int, int, str, str], room_tone: bool) -> bytes:
    channels, sample_width, frame_rate, _comp_type, _comp_name = audio_format
    frame_count = int(frame_rate * silence_ms / 1000)
    if not room_tone or sample_width != 2:
        return b"\x00" * frame_count * channels * sample_width

    data = bytearray()
    seed = 17
    for _ in range(frame_count * channels):
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        sample = ((seed >> 16) % 5) - 2
        data.extend(int(sample).to_bytes(2, "little", signed=True))
    return bytes(data)


def pcm16_peak(frames: bytes) -> int:
    peak = 0
    usable = len(frames) - (len(frames) % 2)
    for offset in range(0, usable, 2):
        sample = int.from_bytes(frames[offset : offset + 2], "little", signed=True)
        peak = max(peak, abs(sample))
    return peak


def scale_pcm16(frames: bytes, channels: int, sample_width: int, factor: float) -> bytes:
    data = bytearray(frames)
    frame_size = channels * sample_width
    frame_count = len(data) // frame_size
    for frame_index in range(frame_count):
        scale_frame_pcm16(data, frame_index, channels, sample_width, factor)
    return bytes(data)


def scale_frame_pcm16(data: bytearray, frame_index: int, channels: int, sample_width: int, factor: float) -> None:
    frame_offset = frame_index * channels * sample_width
    for channel in range(channels):
        offset = frame_offset + channel * sample_width
        sample = int.from_bytes(data[offset : offset + 2], "little", signed=True)
        scaled = max(-32768, min(32767, int(sample * factor)))
        data[offset : offset + 2] = int(scaled).to_bytes(2, "little", signed=True)
