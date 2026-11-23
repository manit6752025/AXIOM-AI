from __future__ import annotations

import json

from fastapi.testclient import TestClient

from axiom.api import workspace
from axiom.api.server import create_app
from axiom.core.hardware import HardwareInfo
from axiom.evaluation.experiments import _welch_test


def test_welch_comparison_requires_replicates_and_returns_bounded_statistics():
    insufficient = _welch_test([0.8], [0.9, 0.95])
    assert insufficient["status"] == "insufficient_replicates"
    assert insufficient["p_value"] is None
    assert insufficient["t_statistic"] is None

    result = _welch_test([0.7, 0.75, 0.8], [0.8, 0.85, 0.9])
    assert result["status"] == "computed"
    assert result["minimum_replicates_per_group"] == 2
    assert result["p_value"] is not None
    assert 0 <= result["p_value"] <= 1
    assert result["t_statistic"] > 0
    assert result["degrees_of_freedom"] > 0

    # A Welch p-value is not defined when both groups have no observed
    # within-group variance, even when their sample means differ.
    degenerate = _welch_test([0.4, 0.4, 0.4], [0.6, 0.6, 0.6])
    assert degenerate["status"] == "degenerate_zero_variance"
    assert degenerate["degenerate_zero_variance"] is True
    assert degenerate["t_statistic"] is None
    assert degenerate["degrees_of_freedom"] is None
    assert degenerate["p_value"] is None
    assert "zero within-group variance" in degenerate["note"]

    # A single zero-variance group remains statistically computable when
    # the other group supplies a positive variance estimate.
    one_constant_group = _welch_test([0.4, 0.4, 0.4], [0.5, 0.6, 0.7])
    assert one_constant_group["status"] == "computed"
    assert one_constant_group["p_value"] is not None
    assert 0 <= one_constant_group["p_value"] <= 1


def test_cli_explains_degenerate_zero_variance_welch_result(monkeypatch):
    from typer.testing import CliRunner
    from axiom.cli import evaluation as evaluation_cli
    from axiom.cli.main import app

    note = "Both groups have zero within-group variance; an inferential p-value is undefined."
    report = {
        "metric": "exact_match",
        "baseline": {
            "replicate_count": 3, "mean": 0.4, "sample_standard_deviation": 0.0,
            "minimum": 0.4, "maximum": 0.4,
        },
        "candidate": {
            "replicate_count": 3, "mean": 0.6, "sample_standard_deviation": 0.0,
            "minimum": 0.6, "maximum": 0.6,
        },
        "difference": {"candidate_minus_baseline_mean": 0.2, "direction": "improved"},
        "statistical_test": {
            "status": "degenerate_zero_variance", "p_value": None,
            "t_statistic": None, "degrees_of_freedom": None, "note": note,
        },
        "scope_note": "The comparison is descriptive when an inferential test is undefined.",
    }
    monkeypatch.setattr(evaluation_cli, "compare_experiment_evaluations", lambda *_args, **_kwargs: report)

    result = CliRunner().invoke(app, [
        "evaluation", "compare-experiments", "1" * 32, "2" * 32,
    ])
    assert result.exit_code == 0, result.stdout
    assert "Statistical test undefined:" in result.stdout
    assert note in result.stdout
    assert "insufficient replicates" not in result.stdout.lower()


def test_api_compares_repeated_evaluations_linked_to_saved_plan_snapshots(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    references = ("yes", "yes", "no")
    outputs = {
        "baseline-a.jsonl": ("yes", "yes", "no"),
        "baseline-b.jsonl": ("yes", "no", "no"),
        "candidate-a.jsonl": ("yes", "no", "no"),
        "candidate-b.jsonl": ("no", "no", "no"),
    }
    for filename, predictions in outputs.items():
        rows = [
            {"reference": reference, "prediction": prediction}
            for reference, prediction in zip(references, predictions, strict=True)
        ]
        (data_dir / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
        )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        workspace,
        "detect_hardware",
        lambda: HardwareInfo("TestOS", "x86_64", 8, 32.0, "Test GPU", 16.0, True),
    )

    with TestClient(create_app()) as client:
        baseline_response = client.post("/api/workspace/experiments", json={
            "name": "Baseline snapshot",
            "parameter_billions": 7,
            "method": "qlora",
        })
        candidate_response = client.post("/api/workspace/experiments", json={
            "name": "Candidate snapshot",
            "parameter_billions": 7,
            "method": "lora",
        })
        assert baseline_response.status_code == 201, baseline_response.text
        assert candidate_response.status_code == 201, candidate_response.text
        baseline_id = baseline_response.json()["experiment"]["id"]
        candidate_id = candidate_response.json()["experiment"]["id"]

        run_ids = {baseline_id: [], candidate_id: []}
        for experiment_id, filenames in (
            (baseline_id, ("baseline-a.jsonl", "baseline-b.jsonl")),
            (candidate_id, ("candidate-a.jsonl", "candidate-b.jsonl")),
        ):
            for filename in filenames:
                response = client.post("/api/evaluation/runs", json={
                    "path": filename,
                    "prediction_field": "prediction",
                    "reference_field": "reference",
                    "experiment_id": experiment_id,
                })
                assert response.status_code == 201, response.text
                run = response.json()["run"]
                assert run["experiment_id"] == experiment_id
                run_ids[experiment_id].append(run["id"])

        comparison_response = client.post("/api/evaluation/experiment-comparisons", json={
            "baseline_experiment_id": baseline_id,
            "candidate_experiment_id": candidate_id,
            "metric": "exact_match",
        })
        assert comparison_response.status_code == 200, comparison_response.text
        comparison = comparison_response.json()["comparison"]
        assert comparison["comparison_type"] == "axiom-repeated-offline-evaluation"
        assert comparison["baseline"]["replicate_count"] == 2
        assert comparison["candidate"]["replicate_count"] == 2
        assert set(comparison["baseline"]["run_ids"]) == set(run_ids[baseline_id])
        assert set(comparison["candidate"]["run_ids"]) == set(run_ids[candidate_id])
        assert comparison["corpus"]["sample_count"] == 3
        assert len(comparison["corpus"]["comparison_corpus_sha256"]) == 64
        assert comparison["difference"]["candidate_minus_baseline_mean"] < 0
        assert comparison["difference"]["direction"] == "regressed"
        assert comparison["statistical_test"]["status"] == "computed"
        assert 0 <= comparison["statistical_test"]["p_value"] <= 1

        from typer.testing import CliRunner
        from axiom.cli.main import app

        cli_result = CliRunner().invoke(app, [
            "evaluation", "compare-experiments", baseline_id, candidate_id,
            "--metric", "exact_match", "--json",
        ])
        assert cli_result.exit_code == 0, cli_result.stdout
        cli_comparison = json.loads(cli_result.stdout)
        assert cli_comparison["baseline"]["replicate_count"] == 2
        assert cli_comparison["candidate"]["replicate_count"] == 2
        assert cli_comparison["statistical_test"]["p_value"] is not None

        missing = client.post("/api/evaluation/experiment-comparisons", json={
            "baseline_experiment_id": "0" * 32,
            "candidate_experiment_id": candidate_id,
        })
        assert missing.status_code == 404
