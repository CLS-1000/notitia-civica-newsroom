"""Turn the daily World State Brief into a five-minute audio briefing.

Input is the brief as Markdown or plain text. Output, in the chosen directory:

- <stem>.wav: 16-bit mono PCM, sentence by sentence with a short pause between;
- <stem>.vtt: a WebVTT transcript, one cue per sentence, timed from the audio;
- <stem>.json: a manifest naming the engine, duration, word counts, what was
  left out, and SHA-256 digests of the brief and of the audio.

Fitting five minutes: the script keeps whole sentences, in order, while their
reading time at 150 words a minute, plus the pauses between them, fits in five
minutes, and records how many sentences it left out. Piper may read faster or
slower than that estimate, so the manifest reports the measured duration and
whether it ran over. Each sentence is synthesized on its own, so every cue's
timing is measured from the audio rather than estimated.

Engines:

- Piper (local, open source), used when a piper binary is found (PIPER_BIN, else
  `piper` on PATH) and a voice model is given (PIPER_MODEL, an .onnx file).
- Deterministic fallback, used otherwise: each sentence becomes silence lasting
  as long as it would take to read at 150 words a minute. It exists so the
  pipeline, its timings, and its transcript can run and be tested anywhere. The
  manifest marks it "publishable": false; a silent file is never a briefing.

Kokoro is not wired in: it ships as a Python library, not a CLI, and adding it
means a model download this repository does not manage.

Usage:
    python -m newsroom.audio_pipeline.brief_tts brief.md --out-dir out/ --stem 2026-11-11
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

logger = logging.getLogger(__name__)

WORDS_PER_MINUTE: Final = 150
TARGET_SECONDS: Final = 300
PAUSE_SECONDS: Final = 0.25
SAMPLE_RATE: Final = 22050
SAMPLE_WIDTH: Final = 2
CHANNELS: Final = 1
MIN_SENTENCE_SECONDS: Final = 0.5
PIPER_TIMEOUT_SECONDS: Final = 120

FALLBACK_ENGINE: Final = "fallback_silent"
PIPER_ENGINE: Final = "piper"

_SENTENCE_END: Final = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


class TtsError(RuntimeError):
    """Synthesis failed or produced audio that cannot be joined."""


# --- Script -------------------------------------------------------------------


def markdown_to_text(markdown: str) -> str:
    """Strip Markdown to the words a listener should hear."""
    text = re.sub(r"```.*?```", " ", markdown, flags=re.DOTALL)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[(S-\d+)\]", " ", text)  # citation markers are not read aloud
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or re.fullmatch(r"[-*_=|: ]{3,}", line):
            lines.append("")
            continue
        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", line)
        line = re.sub(r"^>\s?", "", line)
        line = line.strip("|").replace("|", ", ")
        if raw.lstrip().startswith("#") and not re.search(r"[.!?:]$", line):
            line += "."
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"(\*\*|__|\*|_|`)", "", text)
    paragraphs = [" ".join(block.split()) for block in re.split(r"\n\s*\n", text)]
    return "\n\n".join(block for block in paragraphs if block)


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        sentences.extend(
            part.strip() for part in _SENTENCE_END.split(paragraph) if part.strip()
        )
    return sentences


def word_count(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True, slots=True)
class Script:
    sentences: list[str]
    words: int
    dropped_sentences: int
    dropped_words: int

    @property
    def truncated(self) -> bool:
        return self.dropped_sentences > 0


def reading_seconds(sentence: str) -> float:
    return max(MIN_SENTENCE_SECONDS, word_count(sentence) * 60 / WORDS_PER_MINUTE)


def fit_script(text: str, seconds: float = TARGET_SECONDS) -> Script:
    """Keep whole sentences in order while their estimated reading fits."""
    sentences = split_sentences(text)
    kept: list[str] = []
    elapsed = 0.0
    for index, sentence in enumerate(sentences):
        needed = reading_seconds(sentence) + (PAUSE_SECONDS if kept else 0.0)
        if elapsed + needed > seconds and kept:
            rest = sentences[index:]
            return Script(
                kept, sum(map(word_count, kept)), len(rest), sum(map(word_count, rest))
            )
        kept.append(sentence)
        elapsed += needed
    return Script(kept, sum(map(word_count, kept)), 0, 0)


# --- Engines ------------------------------------------------------------------


class Engine(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def voice(self) -> str | None: ...

    @property
    def publishable(self) -> bool: ...

    def synthesize(self, sentence: str) -> bytes:
        """Return one sentence as a WAV file's bytes."""
        ...


def _wav_bytes(frames: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return buffer.getvalue()


class FallbackEngine:
    """Silence as long as the sentence takes to read. Deterministic; not audio."""

    name = FALLBACK_ENGINE
    voice = None
    publishable = False

    def synthesize(self, sentence: str) -> bytes:
        frames = round(reading_seconds(sentence) * SAMPLE_RATE)
        return _wav_bytes(b"\x00" * frames * SAMPLE_WIDTH * CHANNELS)


class PiperEngine:
    """Piper reads text on stdin and writes a WAV to --output_file."""

    name = PIPER_ENGINE
    publishable = True

    def __init__(self, binary: str, model: Path) -> None:
        self._binary = binary
        self._model = model
        self.voice: str | None = model.stem

    def synthesize(self, sentence: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="nc-tts-") as workdir:
            out = Path(workdir) / "sentence.wav"
            try:
                subprocess.run(
                    [
                        self._binary,
                        "--model",
                        str(self._model),
                        "--output_file",
                        str(out),
                    ],
                    input=sentence.encode("utf-8"),
                    capture_output=True,
                    check=True,
                    timeout=PIPER_TIMEOUT_SECONDS,
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise TtsError(f"piper failed: {error}") from error
            try:
                return out.read_bytes()
            except OSError as error:
                raise TtsError("piper wrote no audio") from error


def select_engine(environ: Mapping[str, str] | None = None) -> Engine:
    """Piper when its binary and a voice model are present, else the fallback."""
    env = environ if environ is not None else os.environ
    binary = env.get("PIPER_BIN") or shutil.which("piper")
    model = env.get("PIPER_MODEL")
    if binary and model and Path(model).is_file():
        return PiperEngine(binary, Path(model))
    logger.warning(
        "no Piper binary and voice model found; using the silent fallback, "
        "which is not publishable"
    )
    return FallbackEngine()


# --- Assembly -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Cue:
    start: float
    end: float
    text: str


def vtt_timestamp(seconds: float) -> str:
    millis = round(seconds * 1000)
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _cue_text(text: str) -> str:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return escaped.replace("-->", "--&gt;")


def render_vtt(cues: Sequence[Cue]) -> str:
    blocks = ["WEBVTT", ""]
    for index, cue in enumerate(cues, start=1):
        blocks.extend(
            [
                str(index),
                f"{vtt_timestamp(cue.start)} --> {vtt_timestamp(cue.end)}",
                _cue_text(cue.text),
                "",
            ]
        )
    return "\n".join(blocks)


def assemble(engine: Engine, sentences: Sequence[str]) -> tuple[bytes, list[Cue], int]:
    """Synthesize each sentence, join them with pauses, and time each cue."""
    params: tuple[int, int, int] | None = None
    chunks: list[bytes] = []
    cues: list[Cue] = []
    position = 0
    for index, sentence in enumerate(sentences):
        try:
            with wave.open(io.BytesIO(engine.synthesize(sentence)), "rb") as handle:
                current = (
                    handle.getnchannels(),
                    handle.getsampwidth(),
                    handle.getframerate(),
                )
                frames = handle.readframes(handle.getnframes())
                count = handle.getnframes()
        except (wave.Error, EOFError) as error:
            raise TtsError(f"sentence {index + 1} is not a WAV: {error}") from error
        if params is None:
            params = current
        elif current != params:
            raise TtsError(f"sentence {index + 1} has different audio parameters")
        channels, width, rate = params
        if index > 0:
            pause = round(PAUSE_SECONDS * rate)
            chunks.append(b"\x00" * pause * width * channels)
            position += pause
        cues.append(Cue(position / rate, (position + count) / rate, sentence))
        chunks.append(frames)
        position += count
    if params is None:
        raise TtsError("the brief has no sentences to read")
    channels, width, rate = params
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(b"".join(chunks))
    return buffer.getvalue(), cues, rate


@dataclass(frozen=True, slots=True)
class Briefing:
    audio_path: Path
    transcript_path: Path
    manifest_path: Path
    manifest: dict[str, object]


def build_briefing(
    brief_text: str,
    out_dir: Path,
    stem: str,
    *,
    engine: Engine | None = None,
) -> Briefing:
    """Write <stem>.wav, <stem>.vtt, and <stem>.json for one brief."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", stem):
        raise ValueError(f"unsafe output name: {stem!r}")
    chosen = engine if engine is not None else select_engine()
    script = fit_script(markdown_to_text(brief_text))
    audio, cues, _rate = assemble(chosen, script.sentences)
    duration = cues[-1].end if cues else 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / f"{stem}.wav"
    transcript_path = out_dir / f"{stem}.vtt"
    manifest_path = out_dir / f"{stem}.json"
    audio_path.write_bytes(audio)
    transcript_path.write_text(render_vtt(cues), encoding="utf-8")
    manifest: dict[str, object] = {
        "engine": chosen.name,
        "voice": chosen.voice,
        "publishable": chosen.publishable,
        "duration_seconds": round(duration, 3),
        "target_seconds": TARGET_SECONDS,
        "over_target": duration > TARGET_SECONDS,
        "sentences": len(script.sentences),
        "words": script.words,
        "truncated": script.truncated,
        "dropped_sentences": script.dropped_sentences,
        "dropped_words": script.dropped_words,
        "audio": audio_path.name,
        "transcript": transcript_path.name,
        "audio_sha256": hashlib.sha256(audio).hexdigest(),
        "brief_sha256": hashlib.sha256(brief_text.encode("utf-8")).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return Briefing(audio_path, transcript_path, manifest_path, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("brief", type=Path, help="the brief, as Markdown or text")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stem", help="output file name (default: the brief's)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        text = args.brief.read_text(encoding="utf-8")
        briefing = build_briefing(text, args.out_dir, args.stem or args.brief.stem)
    except (OSError, TtsError, ValueError) as error:
        logger.error("audio briefing failed: %s", error)
        return 1
    print(json.dumps(briefing.manifest, indent=2, sort_keys=True))
    return 0 if briefing.manifest["publishable"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
