import hashlib
import io
import json

from axiom.evaluation.jsonl import evaluate_jsonl_stream
from axiom.evaluation.jsonl import create_output_manifest
from axiom.evaluation.store import EvaluationStore


def test_bounded_evaluator_computes_documented_metrics_without_values():
    payload = (
        b'{"prediction":"a b","reference":"a c","split":"test"}\n'
        b'{"prediction":"Answer","reference":"answer","split":"test"}\n'
        b'{"prediction":"ignored","reference":"ignored","split":"train"}\n'
    )

    report = evaluate_jsonl_stream(
        io.BytesIO(payload),
        path="eval.jsonl",
        file_name="eval.jsonl",
        file_size_bytes=len(payload),
        prediction_field="prediction",
        reference_field="reference",
        split_field="split",
        split_value="test",
    )

    assert report["counts"]["valid_samples"] == 2
    assert report["counts"]["filtered_split_rows"] == 1
    assert report["metrics"]["exact_match"] == 0
    assert report["metrics"]["normalized_exact_match"] == 0.5
    assert report["metrics"]["token_f1_micro"] == 2 / 3
    assert report["dataset_manifest"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert report["dataset_manifest"]["full_content_hashed"] is True
    interval = report["confidence_intervals"]["exact_match"]
    assert 0 <= interval["low"] <= report["metrics"]["exact_match"] <= interval["high"] <= 1
    assert report["confidence_intervals"]["confidence_level"] == 0.95
    assert len(create_output_manifest(report)["sha256"]) == 64
    assert "Answer" not in json.dumps(report)
    assert report["privacy"]["raw_values_persisted"] is False


def test_evaluator_reports_scan_limits_and_fully_hashes_discarded_long_rows():
    payload = b'{"prediction":"' + b"x" * 100 + b'","reference":"x"}\n'
    payload += b'{"prediction":"yes","reference":"yes"}\n'
    report = evaluate_jsonl_stream(
        io.BytesIO(payload),
        path="eval.jsonl",
        file_name="eval.jsonl",
        file_size_bytes=len(payload),
        prediction_field="prediction",
        reference_field="reference",
        max_line_bytes=64,
    )

    assert report["truncated"] is True
    assert report["truncation_reasons"] == ["max_line_bytes"]
    assert report["counts"]["overlong_rows"] == 1
    assert report["counts"]["valid_samples"] == 1
    assert report["dataset_manifest"]["full_content_hashed"] is True
    assert report["dataset_manifest"]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_evaluation_store_is_immutable_except_explicit_deletion(tmp_path):
    store = EvaluationStore(tmp_path)
    run = {"schema_version": 1, "id": "run-one", "report": {"metrics": {"exact_match": 1.0}}}
    store.add(run)
    assert store.get("run-one") == run
    assert store.list_page(limit=1)["runs"] == [run]
    deleted = store.delete("run-one")
    assert deleted == run
    assert store.get("run-one") is None


def test_evaluation_api_persists_runs_compares_baselines_and_deletes(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from axiom.api.server import create_app

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "predictions.jsonl").write_text(
        '{"reference":"yes","baseline":"yes","candidate":"no","split":"test"}\n'
        '{"reference":"same","baseline":"same","candidate":"same","split":"test"}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    with TestClient(create_app()) as client:
        baseline_response = client.post("/api/evaluation/runs", json={
            "path": "predictions.jsonl",
            "prediction_field": "baseline",
            "reference_field": "reference",
            "split_field": "split",
            "split_value": "test",
            "baseline": True,
        })
        assert baseline_response.status_code == 201, baseline_response.text
        baseline = baseline_response.json()["run"]
        assert baseline["execution_status"] == "precomputed_outputs_only"
        assert baseline["report"]["metrics"]["exact_match"] == 1
        assert baseline["report"]["counts"]["valid_samples"] == 2

        candidate_response = client.post("/api/evaluation/runs", json={
            "path": "predictions.jsonl",
            "prediction_field": "candidate",
            "reference_field": "reference",
            "split_field": "split",
            "split_value": "test",
            "baseline_run_id": baseline["id"],
            "regression_metric": "exact_match",
            "regression_tolerance": 0,
        })
        assert candidate_response.status_code == 201, candidate_response.text
        candidate = candidate_response.json()["run"]
        assert candidate["comparison"]["status"] == "regression"
        assert candidate["comparison"]["delta"] == -0.5
        assert "yes" not in json.dumps(candidate)

        page = client.get("/api/evaluation/runs", params={"limit": 1})
        assert page.status_code == 200
        assert page.json()["total"] == 2
        assert page.json()["has_more"] is True
        assert client.get(f"/api/evaluation/runs/{candidate['id']}").status_code == 200

        traversal = client.post("/api/evaluation/runs", json={
            "path": "../outside.jsonl",
            "prediction_field": "prediction",
            "reference_field": "reference",
        })
        assert traversal.status_code == 400

        deleted = client.delete(f"/api/evaluation/runs/{baseline['id']}")
        assert deleted.status_code == 200
        assert deleted.json()["deleted"]["id"] == baseline["id"]
        assert client.get("/api/evaluation/runs").json()["total"] == 1
