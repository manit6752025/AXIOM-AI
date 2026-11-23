"""Real local-model integration tests.

These tests deliberately use already-installed local runtimes and locally
downloaded model files. They never call a hosted inference provider. They are
skipped by the normal unit suite unless the dedicated workflow opts in.
"""
from __future__ import annotations

import base64
import os
import shutil
import struct
import zlib
from pathlib import Path

import pytest

from axiom.core.inference import InferenceRuntimeError, run_local_vision
from axiom.core.local_speech import (
    synthesize_local_speech,
    transcribe_local_audio,
    write_speech_output_create_only,
)

RUN_REAL_MODELS = os.environ.get("AXIOM_RUN_REAL_LOCAL_MODEL_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_REAL_MODELS,
    reason="Real local runtime tests run only in the dedicated local-model integration workflow.",
)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _blue_png(width: int = 16, height: int = 16) -> bytes:
    """Create a valid RGB PNG fixture without adding an image-library dependency."""
    row = b"\x00" + (b"\x20\x60\xf0" * width)
    pixels = row * height
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(pixels, level=9))
        + _png_chunk(b"IEND", b"")
    )




@pytest.fixture
def repository_workspace(tmp_path: Path):
    """Create temporary workspace data under the repository root required by MCP path guards."""
    root = Path.cwd() / ".axiom-local-model-integration" / tmp_path.name
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=False)


def test_real_local_ollama_vision_inference_and_nonvision_refusal(repository_workspace: Path):

    model = os.environ.get("AXIOM_INTEGRATION_VISION_MODEL", "moondream:1.8b")
    text_model = os.environ.get("AXIOM_INTEGRATION_TEXT_MODEL", "qwen2.5:0.5b")
    workspace = repository_workspace
    (workspace / "data").mkdir(parents=True)
    image_path = workspace / "data" / "blue-square.png"
    image_path.write_bytes(_blue_png())

    report = run_local_vision(
        model,
        "data/blue-square.png",
        "Describe the main visible color in one short sentence.",
        workspace_root=workspace,
        max_new_tokens=48,
        timeout_seconds=180,
    )
    assert report["model"] == model
    assert isinstance(report["response"], str) and report["response"].strip()
    assert report["image"]["format"] == "png"
    assert report["image"]["width"] == 16
    assert report["privacy"]["local_loopback_only"] is True
    assert report["privacy"]["network_used"] is False
    assert report["privacy"]["image_persisted_by_axiom"] is False

    with pytest.raises(InferenceRuntimeError, match="did not explicitly report the vision capability"):
        run_local_vision(
            text_model,
            "data/blue-square.png",
            "Describe the image.",
            workspace_root=workspace,
            max_new_tokens=16,
            timeout_seconds=30,
        )


def test_real_local_piper_synthesis_and_whisper_transcription(repository_workspace: Path, monkeypatch):
    models = Path(os.environ["AXIOM_SPEECH_MODEL_ROOTS"]).resolve(strict=True)
    piper_voice = models / "en_US-lessac-medium.onnx"
    whisper_model = models / "ggml-tiny.en.bin"
    assert piper_voice.is_file()
    assert Path(str(piper_voice) + ".json").is_file()
    assert whisper_model.is_file()

    workspace = repository_workspace
    (workspace / "data").mkdir(parents=True)
    monkeypatch.setenv("AXIOM_SPEECH_MODEL_ROOTS", str(models))

    source_text = "AXIOM runs local speech synthesis and recognition."
    synthesized = synthesize_local_speech(
        str(piper_voice),
        source_text,
        timeout_seconds=120,
        speaker_id=0,
    )
    assert synthesized["engine"] == "piper"
    assert synthesized["audio"]["format"] == "wav"
    assert synthesized["audio"]["size_bytes"] >= 44
    assert synthesized["privacy"]["local_process_only"] is True
    assert synthesized["privacy"]["network_used"] is False
    audio = base64.b64decode(synthesized["audio_base64"], validate=True)

    write_speech_output_create_only("data/piper-output.wav", workspace, audio)
    assert (workspace / "data" / "piper-output.wav").read_bytes() == audio

    transcribed = transcribe_local_audio(
        str(whisper_model),
        "data/piper-output.wav",
        workspace_root=workspace,
        language="en",
        timeout_seconds=180,
        threads=2,
        use_accelerator=False,
    )
    assert transcribed["engine"] == "whisper.cpp"
    assert isinstance(transcribed["transcript"], str) and transcribed["transcript"].strip()
    assert transcribed["privacy"]["network_used"] is False
    assert transcribed["privacy"]["audio_persisted_by_axiom"] is False
    assert transcribed["privacy"]["transcript_persisted_by_axiom"] is False
