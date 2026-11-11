"""Audio briefing pipeline and the commercial Pro checkout component, offline."""

from __future__ import annotations

import io
import json
import re
import stat
import sys
import wave
from pathlib import Path

import pytest
from conftest import REPO_ROOT

from newsroom.audio_pipeline.brief_tts import (
    FALLBACK_ENGINE,
    PAUSE_SECONDS,
    PIPER_ENGINE,
    SAMPLE_RATE,
    TARGET_SECONDS,
    WORDS_PER_MINUTE,
    Cue,
    FallbackEngine,
    PiperEngine,
    TtsError,
    assemble,
    build_briefing,
    fit_script,
    markdown_to_text,
    render_vtt,
    select_engine,
    split_sentences,
    vtt_timestamp,
)

BRIEF = """# World State Brief: 2026-11-11

## Pacific Northwest

- The Willamette at Salem rose **sharply** overnight. [S-12]
- See the [USGS gauge](https://waterdata.usgs.gov/) for readings.

Freight waits at Blaine reached 70 minutes. Passenger waits stayed near 10!

```
internal note: not for broadcast
```
"""


def wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def long_brief(sentences: int = 120) -> str:
    sentence = "The county board approved twelve new road projects this week."
    return " ".join(f"{sentence[:-1]} number {index}." for index in range(sentences))


# --- Script -------------------------------------------------------------------


def test_markdown_is_reduced_to_what_a_listener_hears() -> None:
    text = markdown_to_text(BRIEF)
    assert "#" not in text and "**" not in text and "```" not in text
    assert "internal note" not in text
    assert "[S-12]" not in text and "https://" not in text
    assert "See the USGS gauge for readings." in text
    assert "World State Brief: 2026-11-11." in text


def test_sentences_split_on_their_ends() -> None:
    assert split_sentences(markdown_to_text(BRIEF)) == [
        "World State Brief: 2026-11-11.",
        "Pacific Northwest.",
        "The Willamette at Salem rose sharply overnight.",
        "See the USGS gauge for readings.",
        "Freight waits at Blaine reached 70 minutes.",
        "Passenger waits stayed near 10!",
    ]


def test_long_brief_is_cut_at_a_sentence_to_fit_five_minutes() -> None:
    script = fit_script(long_brief())
    estimated = sum(
        max(0.5, len(item.split()) * 60 / WORDS_PER_MINUTE) for item in script.sentences
    ) + PAUSE_SECONDS * (len(script.sentences) - 1)
    assert estimated <= TARGET_SECONDS
    assert script.truncated and script.dropped_sentences > 0
    assert script.sentences[-1].endswith(".")
    assert script.words + script.dropped_words == len(long_brief().split())


def test_short_brief_is_kept_whole() -> None:
    script = fit_script(markdown_to_text(BRIEF))
    assert not script.truncated and len(script.sentences) == 6


# --- Fallback engine ----------------------------------------------------------


def test_fallback_briefing_is_deterministic_and_never_publishable(
    tmp_path: Path,
) -> None:
    first = build_briefing(BRIEF, tmp_path / "a", "2026-11-11", engine=FallbackEngine())
    second = build_briefing(
        BRIEF, tmp_path / "b", "2026-11-11", engine=FallbackEngine()
    )
    assert first.audio_path.read_bytes() == second.audio_path.read_bytes()
    assert first.transcript_path.read_text() == second.transcript_path.read_text()
    manifest = json.loads(first.manifest_path.read_text())
    assert manifest["engine"] == FALLBACK_ENGINE
    assert manifest["publishable"] is False
    assert manifest["sentences"] == 6 and manifest["truncated"] is False
    assert manifest["audio"] == "2026-11-11.wav"
    assert manifest["transcript"] == "2026-11-11.vtt"
    assert manifest["duration_seconds"] == pytest.approx(
        wav_seconds(first.audio_path.read_bytes()), abs=1e-3
    )


def test_full_length_fallback_briefing_fits_five_minutes(tmp_path: Path) -> None:
    briefing = build_briefing(long_brief(), tmp_path, "long", engine=FallbackEngine())
    manifest = briefing.manifest
    assert manifest["truncated"] is True and manifest["over_target"] is False
    assert (
        TARGET_SECONDS - 10 < float(str(manifest["duration_seconds"])) <= TARGET_SECONDS
    )


def test_transcript_cues_match_the_audio(tmp_path: Path) -> None:
    briefing = build_briefing(BRIEF, tmp_path, "brief", engine=FallbackEngine())
    vtt = briefing.transcript_path.read_text()
    assert vtt.startswith("WEBVTT\n\n1\n00:00:00.000 --> ")
    spans = re.findall(r"(\d\d:\d\d:\d\d\.\d{3}) --> (\d\d:\d\d:\d\d\.\d{3})", vtt)
    assert len(spans) == 6
    assert "Freight waits at Blaine reached 70 minutes." in vtt
    ends = [span[1] for span in spans]
    starts = [span[0] for span in spans]
    assert all(
        later > earlier for earlier, later in zip(ends, starts[1:], strict=False)
    )
    assert ends[-1] == vtt_timestamp(wav_seconds(briefing.audio_path.read_bytes()))


def test_vtt_escapes_markup_and_cue_arrows() -> None:
    vtt = render_vtt([Cue(0.0, 1.5, "A <b>bold</b> & risky --> claim")])
    assert "A &lt;b&gt;bold&lt;/b&gt; &amp; risky --&gt; claim" in vtt
    assert "00:00:00.000 --> 00:00:01.500" in vtt


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0, "00:00:00.000"), (61.25, "00:01:01.250"), (3725.5, "01:02:05.500")],
)
def test_vtt_timestamps(seconds: float, text: str) -> None:
    assert vtt_timestamp(seconds) == text


@pytest.mark.parametrize("stem", ["../escape", "", "a/b", ".hidden"])
def test_unsafe_output_names_are_refused(tmp_path: Path, stem: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        build_briefing(BRIEF, tmp_path, stem, engine=FallbackEngine())


def test_an_empty_brief_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(TtsError, match="no sentences"):
        build_briefing(
            "```\nonly code\n```", tmp_path, "empty", engine=FallbackEngine()
        )


# --- Piper engine -------------------------------------------------------------

FAKE_PIPER = """#!{python}
import sys, wave
args = sys.argv[1:]
out = args[args.index("--output_file") + 1]
text = sys.stdin.read()
if "FAIL" in text:
    sys.exit(1)
rate = 16000 if "ODD" in text else 22050
with wave.open(out, "wb") as handle:
    handle.setnchannels(1)
    handle.setsampwidth(2)
    handle.setframerate(rate)
    handle.writeframes(b"\\x01\\x00" * (len(text) * 441))
"""


@pytest.fixture
def piper(tmp_path: Path) -> tuple[Path, Path]:
    binary = tmp_path / "piper"
    binary.write_text(FAKE_PIPER.format(python=sys.executable))
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    model = tmp_path / "en_US-test-medium.onnx"
    model.write_bytes(b"model")
    return binary, model


def test_piper_is_chosen_when_binary_and_voice_exist(piper: tuple[Path, Path]) -> None:
    binary, model = piper
    engine = select_engine({"PIPER_BIN": str(binary), "PIPER_MODEL": str(model)})
    assert engine.name == PIPER_ENGINE and engine.voice == "en_US-test-medium"


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"PIPER_BIN": "/nonexistent/piper"},
        {"PIPER_BIN": "piper", "PIPER_MODEL": "/nonexistent/voice.onnx"},
    ],
)
def test_without_a_voice_model_the_fallback_is_used(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    assert select_engine(env).name == FALLBACK_ENGINE


def test_piper_cues_are_timed_from_its_own_audio(
    piper: tuple[Path, Path], tmp_path: Path
) -> None:
    engine = PiperEngine(str(piper[0]), piper[1])
    briefing = build_briefing(BRIEF, tmp_path / "out", "piper", engine=engine)
    assert briefing.manifest["engine"] == PIPER_ENGINE
    assert briefing.manifest["publishable"] is True
    assert briefing.manifest["voice"] == "en_US-test-medium"
    # The fake voice reads 0.02 s per character.
    first = "World State Brief: 2026-11-11."
    vtt = briefing.transcript_path.read_text()
    assert f"00:00:00.000 --> {vtt_timestamp(len(first) * 441 / SAMPLE_RATE)}" in vtt


def test_piper_failure_is_an_error(piper: tuple[Path, Path]) -> None:
    engine = PiperEngine(str(piper[0]), piper[1])
    with pytest.raises(TtsError, match="piper failed"):
        assemble(engine, ["This will FAIL."])


def test_mismatched_audio_cannot_be_joined(piper: tuple[Path, Path]) -> None:
    engine = PiperEngine(str(piper[0]), piper[1])
    with pytest.raises(TtsError, match="different audio parameters"):
        assemble(engine, ["First sentence.", "An ODD one."])


def test_non_wav_output_is_an_error() -> None:
    class Broken:
        name = "broken"
        voice = None
        publishable = True

        def synthesize(self, sentence: str) -> bytes:
            return b"not audio"

    with pytest.raises(TtsError, match="not a WAV"):
        assemble(Broken(), ["Hello."])


# --- Pro checkout component ---------------------------------------------------

COMPONENT = (
    REPO_ROOT / "web_public" / "src" / "components" / "SubscriptionCheckout.astro"
)
NOTICE = (
    "Subscriptions to Notitia Civica are commercial transactions supporting "
    "independent reporting by EVASTARARCANA LLC and are NOT tax-deductible."
)


def test_checkout_shows_the_mandatory_notice_verbatim() -> None:
    source = COMPONENT.read_text()
    assert f'"{NOTICE}"' in source
    markup = source.split("---", 2)[2]
    notice = markup.index("{TAX_NOTICE}")
    # Rendered after the open/closed branch, outside it, so it always shows.
    assert notice > markup.index("Pro subscriptions are not open yet.")
    assert markup.count("{TAX_NOTICE}") == 1


def test_checkout_offers_pro_briefs_and_raw_data() -> None:
    source = COMPONENT.read_text()
    assert "Pro Briefs" in source and "Raw data access" in source


def test_checkout_never_collects_payment_details() -> None:
    markup = COMPONENT.read_text().split("---", 2)[2].lower()
    for tag in ("<form", "<input", "<iframe", "<script"):
        assert tag not in markup


def test_checkout_links_only_to_https() -> None:
    source = COMPONENT.read_text()
    assert 'protocol === "https:"' in source
    assert "PUBLIC_PRO_CHECKOUT_URL" in source
