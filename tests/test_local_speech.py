from __future__ import annotations

import base64
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom.core import local_speech


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "workspace"
    (root / "data").mkdir(parents=True)
    return root


def _wav_bytes() -> bytes:
    from io import BytesIO

    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 160)
    return output.getvalue()


def test_resolve_speech_models_only_from_explicit_trusted_root(tmp_path, monkeypatch):
    root = tmp_path / "speech-models"
    root.mkdir()
    whisper = root / "ggml-base.bin"
    whisper.write_bytes(b"local-model")
    piper = root / "en_US-local.onnx"
    piper.write_bytes(b"local-voice")
    (root / "en_US-local.onnx.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("AXIOM_SPEECH_MODEL_ROOTS", str(root))

    assert local_speech._resolve_trusted_model(str(whisper), "whisper")[0] == whisper.resolve()
    assert local_speech._resolve_trusted_model(str(piper), "piper") == (
        piper.resolve(),
        Path(str(piper.resolve()) + ".json"),
    )


def test_speech_model_resolver_rejects_files_outside_roots_and_missing_piper_config(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"model")
    piper = root / "voice.onnx"
    piper.write_bytes(b"voice")
    monkeypatch.setenv("AXIOM_SPEECH_MODEL_ROOTS", str(root))

    with pytest.raises(ValueError, match="trusted speech model root"):
        local_speech._resolve_trusted_model(str(outside), "whisper")
    with pytest.raises(ValueError, match="configuration"):
        local_speech._resolve_trusted_model(str(piper), "piper")


def test_speech_model_resolver_rejects_symlinked_model_file(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"model")
    link = root / "linked.bin"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"File symlinks unavailable: {exc}")
    monkeypatch.setenv("AXIOM_SPEECH_MODEL_ROOTS", str(root))
    with pytest.raises(ValueError, match="trusted speech model root"):
        local_speech._resolve_trusted_model(str(link), "whisper")


def test_transcribe_uses_only_local_whisper_cli_and_bounded_workspace_audio(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    audio = _wav_bytes()
    (root / "data" / "question.wav").write_bytes(audio)
    model = tmp_path / "trusted" / "ggml-base.bin"
    model.parent.mkdir()
    model.write_bytes(b"model")
    executable = tmp_path / "bin" / "whisper-cli"
    executable.parent.mkdir()
    executable.write_text("test executable stub", encoding="utf-8")
    monkeypatch.setattr(local_speech, "_resolve_trusted_model", lambda *_args: (model, None))
    monkeypatch.setattr(local_speech.shutil, "which", lambda name: str(executable) if name == "whisper-cli" else None)
    monkeypatch.setenv("AXIOM_SECRET_TOKEN", "must-not-be-inherited")
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        input_path = Path(command[command.index("-f") + 1])
        observed["audio"] = input_path.read_bytes()
        prefix = command[command.index("-of") + 1]
        Path(prefix + ".txt").write_text("  Hello from local speech.\\n", encoding="utf-8")
        assert kwargs["env"].get("AXIOM_SECRET_TOKEN") is None
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(local_speech.subprocess, "run", fake_run)
    report = local_speech.transcribe_local_audio(
        str(model), "data/question.wav", workspace_root=root,
        language="en", timeout_seconds=90, threads=2,
    )

    assert report["engine"] == "whisper.cpp"
    assert report["transcript"] == "Hello from local speech.\\n"
    assert report["audio"] == {"format": "wav", "size_bytes": len(audio)}
    assert report["privacy"]["network_used"] is False
    assert report["privacy"]["audio_persisted_by_axiom"] is False
    assert observed["audio"] == audio
    assert observed["command"][0] == str(executable)
    assert "-ng" in observed["command"]
    assert observed["kwargs"]["shell"] is False
    assert observed["kwargs"]["timeout"] == 90
    assert observed["kwargs"]["stdout"] == local_speech.subprocess.DEVNULL
    assert not Path(observed["command"][observed["command"].index("-f") + 1]).exists()


def test_transcribe_rejects_invalid_audio_paths_and_signatures_before_starting_binary(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    (root / "data" / "bad.wav").write_bytes(b"not wav")
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    monkeypatch.setattr(local_speech, "_resolve_trusted_model", lambda *_args: (model, None))
    monkeypatch.setattr(local_speech.shutil, "which", lambda _name: pytest.fail("binary lookup should not occur"))
    with pytest.raises(ValueError, match="signature"):
        local_speech.transcribe_local_audio(str(model), "data/bad.wav", workspace_root=root)
    with pytest.raises(ValueError):
        local_speech.transcribe_local_audio(str(model), "../secret.wav", workspace_root=root)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"language": "english"},
        {"language": "en;--help"},
        {"timeout_seconds": 601},
        {"threads": 17},
        {"use_accelerator": "yes"},
    ],
)
def test_transcribe_rejects_unbounded_options_before_runtime(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(local_speech, "_resolve_trusted_model", lambda *_args: pytest.fail("model must not be resolved"))
    with pytest.raises(ValueError):
        local_speech.transcribe_local_audio("model.bin", "data/audio.wav", workspace_root=tmp_path, **kwargs)


def test_transcribe_fails_closed_when_local_whisper_binary_is_missing(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    (root / "data" / "input.wav").write_bytes(_wav_bytes())
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    monkeypatch.setattr(local_speech, "_resolve_trusted_model", lambda *_args: (model, None))
    monkeypatch.setattr(local_speech.shutil, "which", lambda _name: None)
    with pytest.raises(local_speech.LocalSpeechError, match="whisper-cli was not found"):
        local_speech.transcribe_local_audio(str(model), "data/input.wav", workspace_root=root)


def test_synthesize_uses_local_piper_and_returns_bounded_wav(tmp_path, monkeypatch):
    model = tmp_path / "voice.onnx"
    config = Path(str(model) + ".json")
    model.write_bytes(b"model")
    config.write_text("{}", encoding="utf-8")
    executable = tmp_path / "bin" / "piper"
    executable.parent.mkdir()
    executable.write_text("test executable stub", encoding="utf-8")
    monkeypatch.setattr(local_speech, "_resolve_trusted_model", lambda *_args: (model, config))
    monkeypatch.setattr(local_speech.shutil, "which", lambda name: str(executable) if name == "piper" else None)
    wav_data = _wav_bytes()
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        output = Path(command[command.index("--output_file") + 1])
        output.write_bytes(wav_data)
        assert kwargs["input"] == "Hello AXIOM.".encode()
        assert kwargs["env"]["HOME"] != str(Path.home())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(local_speech.subprocess, "run", fake_run)
    report = local_speech.synthesize_local_speech(str(model), "Hello AXIOM.", speaker_id=1)
    assert report["engine"] == "piper"
    assert base64.b64decode(report["audio_base64"]) == wav_data
    assert report["audio"]["format"] == "wav"
    assert report["audio"]["sample_rate_hz"] == 16_000
    assert report["audio"]["duration_seconds"] == 0.01
    assert report["privacy"]["network_used"] is False
    assert report["privacy"]["text_persisted_by_axiom"] is False
    assert "--cuda" not in observed["command"]
    assert observed["kwargs"]["shell"] is False
    assert observed["kwargs"]["timeout"] == 180


def test_synthesize_rejects_invalid_text_options_and_missing_runtime(tmp_path, monkeypatch):
    model = tmp_path / "voice.onnx"
    config = Path(str(model) + ".json")
    model.write_bytes(b"model")
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(local_speech, "_resolve_trusted_model", lambda *_args: (model, config))
    monkeypatch.setattr(local_speech.shutil, "which", lambda _name: None)
    with pytest.raises(ValueError):
        local_speech.synthesize_local_speech(str(model), " ")
    with pytest.raises(ValueError):
        local_speech.synthesize_local_speech(str(model), "x" * 4097)
    with pytest.raises(ValueError):
        local_speech.synthesize_local_speech(str(model), "hello", speaker_id=True)
    with pytest.raises(local_speech.LocalSpeechError, match="piper was not found"):
        local_speech.synthesize_local_speech(str(model), "hello")


def test_speech_output_is_create_only_and_workspace_scoped(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    audio = _wav_bytes()
    report = local_speech.write_speech_output_create_only("data/speech.wav", root, audio)
    assert report == {"path": "data/speech.wav", "size_bytes": len(audio), "created": True}
    assert (root / "data" / "speech.wav").read_bytes() == audio
    with pytest.raises(FileExistsError, match="already exists"):
        local_speech.write_speech_output_create_only("data/speech.wav", root, audio)
    with pytest.raises(ValueError):
        local_speech.write_speech_output_create_only("../outside.wav", root, audio)
