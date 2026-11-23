from __future__ import annotations

import hashlib
import os

import pytest

from axiom.core import prediction_jobs


def _checkpoint(root, payload: bytes, *, rows: int = 1):
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    job_id = "a" * 32
    name = f".axiom-prediction-99999999-{job_id}-{'b' * 32}.tmp"
    path = data / name
    path.write_bytes(payload)
    identity = os.stat(path)
    return {
        "id": job_id,
        "checkpoint_name": name,
        "checkpoint_identity": [identity.st_dev, identity.st_ino],
        "checkpoint_bytes": len(payload),
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "rows_completed": rows,
    }, path


def test_prediction_checkpoint_verifies_committed_prefix_and_truncates_uncommitted_tail(tmp_path):
    committed = b'{"prompt":"first","reference":"A","prediction":"x"}\n'
    record, path = _checkpoint(tmp_path, committed + b'{"partial":"uncommitted')
    stream, name, checkpoint_path, identity, size, digest = prediction_jobs._open_prediction_checkpoint(
        tmp_path, record
    )
    try:
        assert name == record["checkpoint_name"]
        assert checkpoint_path == path
        assert identity == tuple(record["checkpoint_identity"])
        assert size == len(committed)
        assert digest.hexdigest() == record["checkpoint_sha256"]
        stream.seek(0)
        assert stream.read() == committed
        assert os.fstat(stream.fileno()).st_size == len(committed)
    finally:
        stream.close()


def test_prediction_checkpoint_rejects_modified_committed_bytes(tmp_path):
    committed = b'{"prompt":"first","reference":"A","prediction":"x"}\n'
    record, path = _checkpoint(tmp_path, committed)
    path.write_bytes(committed.replace(b'"x"', b'"y"'))
    with pytest.raises(ValueError, match="digest or row count"):
        prediction_jobs._open_prediction_checkpoint(tmp_path, record)


def test_prediction_checkpoint_rejects_symlink_replacement(tmp_path):
    committed = b'{"prompt":"first","reference":"A","prediction":"x"}\n'
    record, path = _checkpoint(tmp_path, committed)
    outside = tmp_path / "outside"
    outside.write_bytes(committed)
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    with pytest.raises(ValueError, match="regular file"):
        prediction_jobs._open_prediction_checkpoint(tmp_path, record)


def test_prediction_asset_fingerprint_changes_when_weights_or_tokenizer_change(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"base weights")
    (model / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (model / "tokenizer.json").write_text('{"version":"one"}', encoding="utf-8")

    original = prediction_jobs._prediction_asset_fingerprint(model, adapter=False)
    assert original and len(original) == 64
    assert prediction_jobs._prediction_asset_fingerprint(model, adapter=False) == original

    (model / "tokenizer.json").write_text('{"version":"two"}', encoding="utf-8")
    assert prediction_jobs._prediction_asset_fingerprint(model, adapter=False) != original
    (model / "model.safetensors").write_bytes(b"changed base weights")
    changed = prediction_jobs._prediction_asset_fingerprint(model, adapter=False)
    assert changed and changed != original


def test_prediction_v1_job_history_migrates_as_non_resumable(tmp_path):
    import json

    job_id = "c" * 32
    legacy = {
        "id": job_id,
        "status": "interrupted",
        "created_at": "2026-11-20T00:00:00+00:00",
        "updated_at": "2026-11-20T00:01:00+00:00",
        "started_at": "2026-11-20T00:00:01+00:00",
        "finished_at": "2026-11-20T00:01:00+00:00",
        "dataset_path": "data/eval.jsonl",
        "output_path": "data/predictions.jsonl",
        "model_path": "models/local",
        "adapter_path": None,
        "max_new_tokens": 32,
        "temperature": 0.7,
        "seed": None,
        "device": "cpu",
        "precision": "fp32",
        "prompt_field": "prompt",
        "reference_field": "reference",
        "prediction_field": "prediction",
        "rows_total": 2,
        "rows_completed": 1,
        "prompts_truncated": 0,
        "error_code": "worker_process_exited",
        "evaluation_run_ids": [],
        "artifact_identity": None,
        "artifact_sha256": None,
        "pending_artifact_identity": None,
        "pending_artifact_sha256": None,
        "runner_pid": 99999999,
        "runner_identity": None,
    }
    state_dir = tmp_path / ".axiom"
    state_dir.mkdir()
    (state_dir / "prediction_jobs.json").write_text(
        json.dumps({"schema_version": 1, "jobs": [legacy]}, indent=2),
        encoding="utf-8",
    )

    [record] = prediction_jobs._read_job_records(tmp_path)
    migrated = prediction_jobs._record_as_job(record, tmp_path)
    summary = prediction_jobs._job_summary(migrated)

    assert record["dataset_sha256"] is None
    assert record["checkpoint_name"] is None
    assert record["resumption_count"] == 0
    assert summary["status"] == "interrupted"
    assert summary["resumable"] is False

def test_resume_prediction_job_revalidates_assets_and_schedules_checkpoint_worker(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    root = tmp_path
    model = root / "models" / "tiny"
    model.mkdir(parents=True)
    data = root / "data"
    data.mkdir()
    dataset = data / "eval.jsonl"
    dataset.write_text(
        '{"prompt":"one","reference":"A"}\n{"prompt":"two","reference":"B"}\n',
        encoding="utf-8",
    )
    committed = b'{"prompt":"one","reference":"A","prediction":"first"}\n'
    record, checkpoint_path = _checkpoint(root, committed)
    job_id = record["id"]
    manifest = prediction_jobs._preflight_dataset(
        "data/eval.jsonl", root, prompt_field="prompt", reference_field="reference"
    )
    record.update({
        "status": "interrupted",
        "created_at": "2026-11-20T00:00:00+00:00",
        "updated_at": "2026-11-20T00:01:00+00:00",
        "started_at": "2026-11-20T00:00:01+00:00",
        "finished_at": "2026-11-20T00:01:00+00:00",
        "dataset_path": "data/eval.jsonl",
        "output_path": "data/predictions.jsonl",
        "model_path": "models/tiny",
        "adapter_path": None,
        "max_new_tokens": 16,
        "temperature": 0.7,
        "seed": 7,
        "device": "cpu",
        "precision": "fp32",
        "prompt_field": "prompt",
        "reference_field": "reference",
        "prediction_field": "prediction",
        "rows_total": 2,
        "prompts_truncated": 0,
        "error_code": "worker_process_exited",
        "evaluation_run_ids": [],
        "artifact_identity": None,
        "artifact_sha256": None,
        "pending_artifact_identity": None,
        "pending_artifact_sha256": None,
        "runner_pid": 99999999,
        "runner_identity": None,
        "dataset_sha256": manifest["sha256"],
        "model_fingerprint": "1" * 64,
        "adapter_fingerprint": None,
        "resumption_count": 0,
    })
    state_dir = root / ".axiom"
    state_dir.mkdir()
    monkeypatch.setattr(prediction_jobs, "_JOBS", {})
    monkeypatch.setattr(prediction_jobs, "resolve_workspace_input", lambda path, workspace, label: model)
    monkeypatch.setattr(prediction_jobs, "validate_local_model_tree", lambda path: None)
    monkeypatch.setattr(prediction_jobs, "_resolve_engine_asset", lambda path, label: path)
    monkeypatch.setattr(prediction_jobs, "_prediction_asset_fingerprint", lambda path, adapter: "1" * 64)
    monkeypatch.setattr(prediction_jobs, "runtime_status", lambda: {"ready": True})
    monkeypatch.setattr(prediction_jobs, "_active_training_job_exists", lambda: False)
    monkeypatch.setattr(prediction_jobs, "acquire_local_compute_job_lock", lambda job: None)
    monkeypatch.setattr(prediction_jobs, "release_local_compute_job_lock", lambda job: None)
    generated_prompts = []

    def fake_generate(model_path, prompts, **kwargs):
        generated_prompts.extend(prompts)
        return {"results": [{
            "response": "second",
            "context_management": {"token_truncation_applied": False},
        } for _prompt in prompts]}

    class SynchronousThread:
        def __init__(self, *, target, args, kwargs, **_options):
            self.target, self.args, self.kwargs = target, args, kwargs

        def start(self):
            self.target(*self.args, **self.kwargs)

    monkeypatch.setattr(prediction_jobs, "generate_local_batch", fake_generate)
    monkeypatch.setattr(prediction_jobs.threading, "Thread", SynchronousThread)
    prediction_jobs._persist_job_record(root, record)

    resumed = prediction_jobs.resume_prediction_job(job_id, root)

    assert resumed is not None
    assert resumed["status"] == "completed"
    assert resumed["resumption_count"] == 1
    assert resumed["rows_completed"] == 2
    assert generated_prompts == ["two"]
    published = data / "predictions.jsonl"
    output_rows = [json.loads(line) for line in published.read_text(encoding="utf-8").splitlines()]
    assert [row["prediction"] for row in output_rows] == ["first", "second"]
    persisted = next(item for item in prediction_jobs._read_job_records(root) if item["id"] == job_id)
    assert persisted["status"] == "completed"
    assert persisted["checkpoint_name"] is None
    assert persisted["artifact_sha256"] == hashlib.sha256(published.read_bytes()).hexdigest()


def test_resume_prediction_job_refuses_dataset_fingerprint_changes(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    root = tmp_path
    model = root / "models" / "tiny"
    model.mkdir(parents=True)
    data = root / "data"
    data.mkdir()
    dataset = data / "eval.jsonl"
    dataset.write_text('{"prompt":"changed","reference":"A"}\n', encoding="utf-8")
    committed = b'{"prompt":"one","reference":"A","prediction":"first"}\n'
    record, _checkpoint_path = _checkpoint(root, committed)
    job_id = record["id"]
    record.update({
        "status": "interrupted",
        "created_at": "2026-11-20T00:00:00+00:00",
        "updated_at": "2026-11-20T00:01:00+00:00",
        "started_at": "2026-11-20T00:00:01+00:00",
        "finished_at": "2026-11-20T00:01:00+00:00",
        "dataset_path": "data/eval.jsonl",
        "output_path": "data/predictions.jsonl",
        "model_path": "models/tiny",
        "adapter_path": None,
        "max_new_tokens": 16,
        "temperature": 0.7,
        "seed": 7,
        "device": "cpu",
        "precision": "fp32",
        "prompt_field": "prompt",
        "reference_field": "reference",
        "prediction_field": "prediction",
        "rows_total": 2,
        "prompts_truncated": 0,
        "error_code": "worker_process_exited",
        "evaluation_run_ids": [],
        "artifact_identity": None,
        "artifact_sha256": None,
        "pending_artifact_identity": None,
        "pending_artifact_sha256": None,
        "runner_pid": 99999999,
        "runner_identity": None,
        "dataset_sha256": hashlib.sha256(b'{"prompt":"original","reference":"A"}\n').hexdigest(),
        "model_fingerprint": "1" * 64,
        "adapter_fingerprint": None,
        "resumption_count": 0,
    })
    (root / ".axiom").mkdir()
    monkeypatch.setattr(prediction_jobs, "_JOBS", {})
    monkeypatch.setattr(prediction_jobs, "resolve_workspace_input", lambda path, workspace, label: model)
    monkeypatch.setattr(prediction_jobs, "validate_local_model_tree", lambda path: None)
    monkeypatch.setattr(prediction_jobs, "_resolve_engine_asset", lambda path, label: path)
    monkeypatch.setattr(prediction_jobs, "_prediction_asset_fingerprint", lambda path, adapter: "1" * 64)
    prediction_jobs._persist_job_record(root, record)

    with pytest.raises(ValueError, match="dataset fingerprint"):
        prediction_jobs.resume_prediction_job(job_id, root)


