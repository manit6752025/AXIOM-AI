from __future__ import annotations

import pytest

from axiom.core import ollama_service as service


def _offline_report() -> dict:
    return {
        "backend": "ollama",
        "endpoint": "http://127.0.0.1:11434",
        "runtime_installed": True,
        "status": "offline",
        "models": [],
        "error": "offline",
        "checked_at": "2026-10-24T00:00:00+00:00",
    }


def test_service_status_exposes_control_capabilities_without_process_paths(monkeypatch):
    monkeypatch.setattr(service, "runtime_status", _offline_report)
    monkeypatch.setattr(service, "_managed_state", lambda: None)

    report = service.service_status()

    assert report["service_managed_by_axiom"] is False
    assert report["service_can_start"] is True
    assert report["service_can_stop"] is False
    assert report["service_control_scope"] == "loopback_only"
    assert "service_message" in report
    assert "/home/" not in str(report)


def test_service_status_marks_owned_process_as_stoppable(monkeypatch):
    monkeypatch.setattr(service, "runtime_status", _offline_report)
    monkeypatch.setattr(service, "_managed_state", lambda: {"pid": 456, "executable": "/usr/bin/ollama"})

    report = service.service_status()

    assert report["service_managed_by_axiom"] is True
    assert report["service_can_start"] is False
    assert report["service_can_stop"] is True


def test_start_refuses_when_ollama_executable_is_missing(monkeypatch):
    monkeypatch.setattr(service, "runtime_status", _offline_report)
    monkeypatch.setattr(service, "_managed_state", lambda: None)
    monkeypatch.setattr(service.shutil, "which", lambda _name: None)
    monkeypatch.setattr(service.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("must not launch an absent executable"))

    with pytest.raises(service.OllamaServiceError, match="not found on PATH"):
        service.start_service()


def test_stop_refuses_to_signal_a_process_not_owned_by_axiom(monkeypatch):
    monkeypatch.setattr(service, "_managed_state", lambda: None)
    monkeypatch.setattr(service.os, "kill", lambda *_args, **_kwargs: pytest.fail("must not signal an unowned process"))

    with pytest.raises(service.OllamaServiceError, match="only stop an Ollama process it started"):
        service.stop_service()


def test_process_identity_mismatch_prevents_stopping_pid_reuse(monkeypatch):
    record = {"pid": 123, "executable": "/usr/bin/ollama", "identity": "linux:10"}
    monkeypatch.setattr(service, "_process_info", lambda _pid: {
        "executable": "/usr/bin/ollama",
        "command": "/usr/bin/ollama\0serve",
        "identity": "linux:11",
    })

    assert service._matches_state(record) is False


def test_start_model_pull_rejects_urls_before_checking_runtime(monkeypatch):
    monkeypatch.setattr(service, "runtime_status", lambda: pytest.fail("invalid model names must be rejected first"))

    with pytest.raises(ValueError):
        service.start_model_pull("https://example.com/model")


def test_start_model_pull_requires_an_online_local_service(monkeypatch):
    monkeypatch.setattr(service, "runtime_status", _offline_report)
    monkeypatch.setattr(service.shutil, "which", lambda _name: "/usr/bin/ollama")

    with pytest.raises(service.OllamaServiceError, match="Start the local Ollama service"):
        service.start_model_pull("qwen2.5:0.5b")


def test_progress_output_removes_terminal_controls_and_caps_log_lines():
    line = service._sanitize_line("\x1b[31mPulling layer 100%\x1b[0m\x07" + ("x" * 500))

    assert "\x1b" not in line
    assert "\x07" not in line
    assert "Pulling layer 100%" in line
    assert len(line) <= 320


def test_pull_job_history_is_bounded_and_job_ids_are_validated():
    original = dict(service._PULL_JOBS)
    try:
        service._PULL_JOBS.clear()
        job = service._PullJob(id="a" * 32, model="qwen2.5:0.5b")
        service._PULL_JOBS[job.id] = job

        report = service.get_model_pull(job.id)
        assert report["status"] == "queued"
        assert report["logs"] == []
        with pytest.raises(ValueError, match="Invalid model-download job ID"):
            service.get_model_pull("../not-a-job")
    finally:
        service._PULL_JOBS.clear()
        service._PULL_JOBS.update(original)


def test_start_terminates_process_if_ownership_state_cannot_be_saved(monkeypatch):
    class FakeProcess:
        pid = 12345
        terminated = False
        waited = False
        def poll(self):
            return None
        def terminate(self):
            self.terminated = True
        def wait(self, timeout=None):
            self.waited = True
            return 0

    process = FakeProcess()
    monkeypatch.setattr(service, "service_status", lambda: {**_offline_report(), "service_managed_by_axiom": False, "service_can_start": True})
    monkeypatch.setattr(service.shutil, "which", lambda _name: "/usr/bin/ollama")
    monkeypatch.setattr(service.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(service, "_process_info", lambda _pid: {
        "executable": "/usr/bin/ollama", "command": "/usr/bin/ollama\0serve", "identity": "linux:1",
    })
    monkeypatch.setattr(service, "_write_state", lambda _record: (_ for _ in ()).throw(OSError("disk unavailable")))

    with pytest.raises(service.OllamaServiceError, match="could not safely record ownership"):
        service.start_service()

    assert process.terminated is True
    assert process.waited is True


def test_start_reaps_process_if_process_identity_cannot_be_verified(monkeypatch):
    class FakeProcess:
        pid = 12346
        terminated = False
        waited = False
        def poll(self):
            return None
        def terminate(self):
            self.terminated = True
        def wait(self, timeout=None):
            self.waited = True
            return 0

    process = FakeProcess()
    monkeypatch.setattr(service, "service_status", lambda: {**_offline_report(), "service_managed_by_axiom": False, "service_can_start": True})
    monkeypatch.setattr(service.shutil, "which", lambda _name: "/usr/bin/ollama")
    monkeypatch.setattr(service.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(service, "_process_info", lambda _pid: None)
    monkeypatch.setattr(service, "_SERVICE_START_TIMEOUT_SECONDS", 0)

    with pytest.raises(service.OllamaServiceError, match="could not verify ownership"):
        service.start_service()

    assert process.terminated is True
    assert process.waited is True
