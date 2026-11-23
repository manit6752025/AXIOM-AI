"""Bounded local speech adapters for whisper.cpp and Piper.

The adapters invoke already-installed executables with explicit, locally trusted
model files. They never download models, contact cloud services, or use a shell.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any
import base64

from axiom.core.mcp_paths import open_mcp_workspace_file, open_mcp_workspace_output_parent

MAX_SPEECH_INPUT_BYTES = 64 * 1024 * 1024
MAX_TRANSCRIPT_CHARS = 65_536
MAX_SYNTHESIS_TEXT_CHARS = 4_096
MAX_SYNTHESIS_WAV_BYTES = 12 * 1024 * 1024
MAX_SPEECH_TIMEOUT_SECONDS = 600
MAX_SPEECH_STDERR_CAPTURE_BYTES = 4096
_LANGUAGE = re.compile(r"^(?:auto|[a-z]{2,3})$")
_SPEAKER_ID_MAX = 255
_SPEECH_PROCESS_LOCK = threading.Lock()


class LocalSpeechError(RuntimeError):
    """Raised when a local speech runtime cannot complete a bounded request."""


def _allowed_speech_roots() -> tuple[Path, ...]:
    """Resolve trusted speech model roots without accepting symlink roots."""
    configured = os.environ.get("AXIOM_SPEECH_MODEL_ROOTS", "").strip()
    if not configured:
        configured = os.environ.get("AXIOM_MODEL_ROOTS", "").strip()
    defaults = (Path.cwd() / "models", Path.home() / "models")
    custom = tuple(Path(item).expanduser() for item in configured.split(os.pathsep) if item.strip())
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in (*defaults, *custom):
        try:
            is_junction = getattr(candidate, "is_junction", lambda: False)
            if candidate.is_symlink() or is_junction():
                if candidate in custom:
                    raise ValueError("Speech model roots must not be symbolic links or junctions.")
                continue
            canonical = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            if candidate in custom:
                raise ValueError("AXIOM_SPEECH_MODEL_ROOTS entries must be existing local directories.") from None
            continue
        if not canonical.is_dir():
            if candidate in custom:
                raise ValueError("AXIOM_SPEECH_MODEL_ROOTS entries must be existing local directories.")
            continue
        key = os.path.normcase(str(canonical))
        if key not in seen:
            seen.add(key)
            roots.append(canonical)
    return tuple(roots)


def _resolve_trusted_model(raw: str, kind: str) -> tuple[Path, Path | None]:
    """Resolve an exact model file by matching a bounded walk of trusted roots."""
    if (
        not isinstance(raw, str) or not raw or raw != raw.strip() or len(raw) > 4096
        or any(ord(char) < 32 or ord(char) == 127 for char in raw)
    ):
        raise ValueError("Speech model path must be a valid absolute path under a trusted speech model root.")
    suffix = ".bin" if kind == "whisper" else ".onnx"
    candidate_key = os.path.normcase(os.path.abspath(os.path.expanduser(raw)))
    roots = _allowed_speech_roots()
    if not roots:
        raise ValueError("No trusted speech model roots are configured. Set AXIOM_SPEECH_MODEL_ROOTS or AXIOM_MODEL_ROOTS.")
    for root in roots:
        root_text = str(root)
        visited = 0
        for directory, subdirs, filenames in os.walk(root_text, topdown=True, followlinks=False):
            visited += 1
            if visited > 20_000:
                break
            try:
                relative = os.path.relpath(directory, root_text)
                depth = 0 if relative == os.curdir else len(Path(relative).parts)
            except (OSError, ValueError):
                subdirs[:] = []
                continue
            safe_subdirs = []
            for name in subdirs:
                child = os.path.join(directory, name)
                if not os.path.islink(child):
                    safe_subdirs.append(name)
            subdirs[:] = safe_subdirs if depth < 10 else []
            for name in filenames:
                if not name.lower().endswith(suffix):
                    continue
                discovered = os.path.join(directory, name)
                key = os.path.normcase(os.path.abspath(discovered))
                if key != candidate_key or os.path.islink(discovered):
                    continue
                try:
                    resolved = Path(discovered).resolve(strict=True)
                    metadata = resolved.stat()
                    if (
                        not stat.S_ISREG(metadata.st_mode)
                        or os.path.commonpath((str(resolved), root_text)) != root_text
                        or metadata.st_size <= 0
                        or metadata.st_size > 16 * 1024 * 1024 * 1024
                    ):
                        raise ValueError("Speech model file is not a bounded regular file under its trusted root.")
                except (OSError, RuntimeError, ValueError):
                    raise ValueError("Speech model file could not be verified under a trusted root.") from None
                config: Path | None = None
                if kind == "piper":
                    config_path = Path(str(resolved) + ".json")
                    try:
                        config_metadata = config_path.lstat()
                        config_resolved = config_path.resolve(strict=True)
                    except (OSError, RuntimeError):
                        raise ValueError("Piper voice configuration (.onnx.json) is missing.") from None
                    if (
                        stat.S_ISLNK(config_metadata.st_mode)
                        or not stat.S_ISREG(config_metadata.st_mode)
                        or os.path.commonpath((str(config_resolved), root_text)) != root_text
                        or not config_resolved.is_file()
                    ):
                        raise ValueError("Piper voice configuration is not a regular trusted file.")
                    config = config_resolved
                return resolved, config
    raise ValueError("Speech model path must match an existing model file discovered under a trusted speech model root.")


def _safe_environment(executable: str, temporary_root: Path) -> dict[str, str]:
    """Avoid passing application secrets or proxy settings to local audio binaries."""
    binary_dir = str(Path(executable).resolve(strict=True).parent)
    environment = {"PATH": binary_dir + os.pathsep + os.defpath}
    for key in ("SYSTEMROOT", "WINDIR"):
        value = os.environ.get(key)
        if value:
            environment[key] = value
    environment["HOME"] = str(temporary_root)
    environment["TMPDIR"] = str(temporary_root)
    environment["TEMP"] = str(temporary_root)
    return environment


def _drain_capped_stderr(stream: Any, captured: bytearray) -> None:
    """Drain the entire child pipe while retaining only a small diagnostic prefix."""
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            remaining = MAX_SPEECH_STDERR_CAPTURE_BYTES - len(captured)
            if remaining > 0:
                captured.extend(chunk[:remaining])
    except (OSError, ValueError):
        return


def _write_child_stdin(stream: Any, payload: bytes) -> None:
    """Send the small, bounded TTS input without blocking process supervision."""
    try:
        view = memoryview(payload)
        while view:
            written = stream.write(view)
            if written is None or written <= 0:
                break
            view = view[written:]
        stream.flush()
    except (BrokenPipeError, OSError, ValueError):
        # A runtime may exit before consuming all text; its exit status is handled
        # by the supervising thread below.
        pass
    finally:
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def _run_local_process(
    executable: str,
    arguments: list[str],
    *,
    temporary_root: Path,
    timeout_seconds: int,
    input_text: str | None = None,
) -> None:
    """Run a local binary without a shell, inherited secrets, or unbounded logs."""
    if not _SPEECH_PROCESS_LOCK.acquire(blocking=False):
        raise LocalSpeechError("Another local speech task is active; retry after it finishes.")
    try:
        try:
            process = subprocess.Popen(
                [executable, *arguments],
                cwd=temporary_root,
                env=_safe_environment(executable, temporary_root),
                stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                shell=False,
            )
        except OSError:
            raise LocalSpeechError("The local speech executable could not be started.") from None

        if process.stderr is None:
            if process.stdin is not None:
                process.stdin.close()
            process.kill()
            process.wait()
            raise LocalSpeechError("The local speech process did not expose a diagnostic pipe.")

        captured_stderr = bytearray()
        reader = threading.Thread(
            target=_drain_capped_stderr,
            args=(process.stderr, captured_stderr),
            name="axiom-speech-stderr",
            daemon=True,
        )
        reader.start()

        writer: threading.Thread | None = None
        if input_text is not None and process.stdin is not None:
            writer = threading.Thread(
                target=_write_child_stdin,
                args=(process.stdin, input_text.encode("utf-8")),
                name="axiom-speech-stdin",
                daemon=True,
            )
            writer.start()
        elif process.stdin is not None:
            process.stdin.close()

        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait()
            except OSError:
                pass
            raise LocalSpeechError("The local speech process exceeded its configured time limit.") from None
        except OSError:
            process.kill()
            try:
                process.wait()
            except OSError:
                pass
            raise LocalSpeechError("The local speech process could not be supervised safely.") from None
        finally:
            if writer is not None:
                writer.join(timeout=5)
                if writer.is_alive() and process.stdin is not None:
                    try:
                        process.stdin.close()
                    except (OSError, ValueError):
                        pass
                    writer.join(timeout=1)
            elif process.stdin is not None:
                try:
                    process.stdin.close()
                except (OSError, ValueError):
                    pass
            reader.join(timeout=5)
            if reader.is_alive():
                try:
                    process.stderr.close()
                except (OSError, ValueError):
                    pass
                reader.join(timeout=1)
            try:
                process.stderr.close()
            except (OSError, ValueError):
                pass

        detail = bytes(captured_stderr).decode("utf-8", errors="replace").replace("\x00", "")[:1000]
        if return_code != 0:
            safe_detail = " ".join(detail.split())[:400]
            raise LocalSpeechError(
                "The local speech process failed." + (f" {safe_detail}" if safe_detail else "")
            )
    finally:
        _SPEECH_PROCESS_LOCK.release()


def _workspace_audio(image_path: str, workspace_root: Path) -> tuple[Path, bytes]:
    if (
        not isinstance(image_path, str) or not image_path.startswith("data/")
        or image_path != image_path.strip()
    ):
        raise ValueError("Audio path must be workspace-relative under data/.")
    suffix = Path(image_path).suffix.lower()
    if suffix not in {".wav", ".mp3", ".flac", ".ogg"}:
        raise ValueError("Audio input must be a WAV, MP3, FLAC, or OGG file.")
    path, stream = open_mcp_workspace_file(image_path, workspace_root)
    try:
        before = os.fstat(stream.fileno())
        if before.st_size <= 0 or before.st_size > MAX_SPEECH_INPUT_BYTES:
            raise ValueError("Audio file must be between 1 byte and 64 MiB.")
        with stream:
            data = stream.read(MAX_SPEECH_INPUT_BYTES + 1)
            after = os.fstat(stream.fileno())
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or len(data) != after.st_size
        ):
            raise ValueError("Audio file changed while AXIOM was reading it; retry with a stable file.")
    except OSError:
        stream.close()
        raise ValueError("Audio file could not be read safely.") from None
    if suffix == ".wav" and not (len(data) >= 12 and data[:4] in {b"RIFF", b"RF64"} and data[8:12] == b"WAVE"):
        raise ValueError("WAV extension and RIFF/WAVE file signature do not match.")
    if suffix == ".flac" and not data.startswith(b"fLaC"):
        raise ValueError("FLAC extension and file signature do not match.")
    if suffix == ".ogg" and not data.startswith(b"OggS"):
        raise ValueError("OGG extension and file signature do not match.")
    if suffix == ".mp3" and not (data.startswith(b"ID3") or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0)):
        raise ValueError("MP3 extension and file signature do not match.")
    return path, data


def transcribe_local_audio(
    model_path: str,
    audio_path: str,
    *,
    workspace_root: Path,
    language: str = "auto",
    timeout_seconds: int = 300,
    threads: int = 4,
    use_accelerator: bool = False,
) -> dict[str, Any]:
    """Transcribe one workspace audio file with a preinstalled whisper.cpp runtime."""
    if not isinstance(language, str) or not _LANGUAGE.fullmatch(language):
        raise ValueError("Language must be 'auto' or a two/three-letter Whisper language code.")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 10 <= timeout_seconds <= MAX_SPEECH_TIMEOUT_SECONDS:
        raise ValueError("Speech timeout must be an integer from 10 through 600 seconds.")
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 16:
        raise ValueError("Whisper thread count must be an integer from 1 through 16.")
    if not isinstance(use_accelerator, bool):
        raise ValueError("Accelerator selection must be a boolean.")
    model, _ = _resolve_trusted_model(model_path, "whisper")
    _audio_file, audio_data = _workspace_audio(audio_path, workspace_root)
    executable = shutil.which("whisper-cli")
    if not executable:
        raise LocalSpeechError("whisper-cli was not found. Install whisper.cpp locally; AXIOM will not download it.")
    with tempfile.TemporaryDirectory(prefix="axiom-whisper-") as temporary:
        temporary_root = Path(temporary)
        source = temporary_root / ("input" + Path(audio_path).suffix.lower())
        source.write_bytes(audio_data)
        prefix = temporary_root / "transcript"
        args = [
            "-m", str(model), "-f", str(source), "-otxt", "-of", str(prefix),
            "-np", "-nt", "-t", str(threads), "-l", language,
        ]
        if not use_accelerator:
            args.append("-ng")
        _run_local_process(
            executable, args, temporary_root=temporary_root, timeout_seconds=timeout_seconds,
        )
        transcript_path = Path(str(prefix) + ".txt")
        try:
            metadata = transcript_path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_TRANSCRIPT_CHARS * 4:
                raise LocalSpeechError("whisper.cpp produced an invalid or oversized transcript file.")
            transcript = transcript_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise LocalSpeechError("whisper.cpp did not produce a readable UTF-8 transcript.") from None
    transcript = transcript.strip()
    if not transcript:
        raise LocalSpeechError("whisper.cpp completed without producing speech text.")
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        raise LocalSpeechError("Transcript exceeds the 65,536-character output limit.")
    return {
        "schema_version": 1,
        "engine": "whisper.cpp",
        "model": model.name,
        "transcript": transcript,
        "audio": {"format": Path(audio_path).suffix.lower().lstrip("."), "size_bytes": len(audio_data)},
        "execution": {"device": "auto" if use_accelerator else "cpu", "threads": threads, "language": language},
        "privacy": {
            "local_process_only": True,
            "network_used": False,
            "audio_returned": False,
            "audio_persisted_by_axiom": False,
            "transcript_persisted_by_axiom": False,
            "model_downloaded": False,
        },
        "scope_note": "AXIOM sent the audio only to the installed local whisper.cpp executable. No cloud transcription or model download was used.",
    }


def synthesize_local_speech(
    model_path: str,
    text: str,
    *,
    max_audio_bytes: int = MAX_SYNTHESIS_WAV_BYTES,
    timeout_seconds: int = 180,
    speaker_id: int = 0,
    use_accelerator: bool = False,
) -> dict[str, Any]:
    """Synthesize bounded text to an in-memory WAV using an installed Piper voice."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Speech text must contain non-whitespace characters.")
    if len(text) > MAX_SYNTHESIS_TEXT_CHARS:
        raise ValueError("Speech text exceeds the 4,096-character limit.")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 10 <= timeout_seconds <= 300:
        raise ValueError("Speech synthesis timeout must be an integer from 10 through 300 seconds.")
    if isinstance(speaker_id, bool) or not isinstance(speaker_id, int) or not 0 <= speaker_id <= _SPEAKER_ID_MAX:
        raise ValueError("Piper speaker ID must be an integer from 0 through 255.")
    if isinstance(max_audio_bytes, bool) or not isinstance(max_audio_bytes, int) or not 1024 <= max_audio_bytes <= MAX_SYNTHESIS_WAV_BYTES:
        raise ValueError("Maximum generated audio size must be between 1 KiB and 12 MiB.")
    if not isinstance(use_accelerator, bool):
        raise ValueError("Accelerator selection must be a boolean.")
    model, config = _resolve_trusted_model(model_path, "piper")
    executable = shutil.which("piper")
    if not executable:
        raise LocalSpeechError("piper was not found. Install Piper locally; AXIOM will not download it or a voice.")
    with tempfile.TemporaryDirectory(prefix="axiom-piper-") as temporary:
        temporary_root = Path(temporary)
        output = temporary_root / "speech.wav"
        args = ["--model", str(model), "--config", str(config), "--output_file", str(output), "--speaker", str(speaker_id)]
        if use_accelerator:
            args.append("--cuda")
        _run_local_process(
            executable, args, temporary_root=temporary_root, timeout_seconds=timeout_seconds,
            input_text=text.strip(),
        )
        try:
            metadata = output.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise LocalSpeechError("Piper did not produce a regular WAV file.")
            if not 44 <= metadata.st_size <= max_audio_bytes:
                raise LocalSpeechError("Generated speech exceeds the configured audio output limit.")
            with wave.open(str(output), "rb") as wav:
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                sample_rate = wav.getframerate()
                frames = wav.getnframes()
                duration = frames / sample_rate if sample_rate else 0
                if (
                    channels not in {1, 2} or sample_width not in {2, 4}
                    or not 8_000 <= sample_rate <= 96_000
                    or frames <= 0 or duration > 600
                ):
                    raise LocalSpeechError("Piper WAV output has unsupported audio parameters.")
            audio = output.read_bytes()
        except (OSError, EOFError, wave.Error):
            raise LocalSpeechError("Piper did not produce a valid, bounded WAV file.") from None
    return {
        "schema_version": 1,
        "engine": "piper",
        "model": model.name,
        "audio_base64": base64.b64encode(audio).decode("ascii"),
        "audio": {
            "format": "wav",
            "size_bytes": len(audio),
            "channels": channels,
            "sample_rate_hz": sample_rate,
            "sample_width_bytes": sample_width,
            "duration_seconds": round(duration, 3),
        },
        "execution": {"device": "cuda" if use_accelerator else "cpu", "speaker_id": speaker_id},
        "privacy": {
            "local_process_only": True,
            "network_used": False,
            "text_persisted_by_axiom": False,
            "audio_persisted_by_axiom": False,
            "model_downloaded": False,
        },
        "scope_note": "AXIOM sent the text only to the installed local Piper executable. WAV bytes remain in the response and temporary directory; no cloud synthesis or model download was used.",
    }



def write_speech_output_create_only(output_path: str, workspace_root: Path, audio: bytes) -> dict[str, Any]:
    """Create a new workspace WAV without following links or overwriting an existing file."""
    if (
        not isinstance(output_path, str) or not output_path.startswith("data/")
        or output_path != output_path.strip() or not output_path.lower().endswith(".wav")
    ):
        raise ValueError("Speech output must be a new workspace-relative data/NAME.wav file.")
    if not isinstance(audio, bytes) or not 44 <= len(audio) <= MAX_SYNTHESIS_WAV_BYTES:
        raise ValueError("Speech output bytes are outside the supported WAV size bounds.")
    path, parent_fd, name, available = open_mcp_workspace_output_parent(output_path, workspace_root)
    if not available:
        if parent_fd is not None:
            os.close(parent_fd)
        raise FileExistsError("Speech output already exists; choose a new data/NAME.wav path.")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    created = False
    succeeded = False
    try:
        if parent_fd is not None:
            descriptor = os.open(name, flags, 0o600, dir_fd=parent_fd)
        else:
            descriptor = os.open(path, flags, 0o600)
        created = True
        view = memoryview(audio)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Short write while saving speech audio.")
            view = view[written:]
        os.fsync(descriptor)
        succeeded = True
    except FileExistsError:
        raise FileExistsError("Speech output already exists; choose a new data/NAME.wav path.") from None
    except OSError:
        raise OSError("Speech audio could not be saved safely inside the workspace.") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created and not succeeded:
            try:
                if parent_fd is not None:
                    os.unlink(name, dir_fd=parent_fd)
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                pass
        if parent_fd is not None:
            if succeeded:
                try:
                    os.fsync(parent_fd)
                except OSError:
                    pass
            os.close(parent_fd)
    return {"path": output_path, "size_bytes": len(audio), "created": True}
