"""Traceable repeated-evaluation comparisons for saved training experiments."""
from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Any

from axiom.core.experiments import ExperimentStore
from axiom.core.plan_reproducibility import reproducible_plan_document
from axiom.evaluation.store import EvaluationStore

_EXPERIMENT_ID = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METRICS = frozenset({"exact_match", "normalized_exact_match", "token_f1_micro"})
_MAX_EVALUATION_RUNS = 1000


class EvaluationExperimentNotFoundError(ValueError):
    """A saved experiment referenced by an evaluation is missing."""


class EvaluationComparisonError(ValueError):
    """Evaluation runs cannot be compared under a common, complete corpus."""


def validate_saved_experiment(experiment_id: str | None) -> dict[str, Any] | None:
    """Resolve an optional experiment association to an immutable saved plan."""
    if experiment_id is None:
        return None
    if not isinstance(experiment_id, str) or not _EXPERIMENT_ID.fullmatch(experiment_id):
        raise ValueError("Experiment ID must be a 32-character hexadecimal identifier.")
    try:
        experiment = ExperimentStore().get(experiment_id)
    except (OSError, ValueError):
        raise EvaluationComparisonError("Saved experiment history could not be read safely.") from None
    if experiment is None:
        raise EvaluationExperimentNotFoundError("Saved experiment was not found.")
    try:
        reproducible_plan_document(experiment)
    except (TypeError, ValueError):
        raise EvaluationComparisonError("The associated experiment has no valid immutable plan snapshot.") from None
    return experiment


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_variance(values: list[float], mean: float) -> float:
    if len(values) < 2:
        raise ValueError("Sample variance requires at least two replicates.")
    return math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the incomplete-beta continued fraction with bounded iterations."""
    max_iterations = 200
    epsilon = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        twice = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + twice) * (a + twice))
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c
        coefficient = -(a + iteration) * (qab + iteration) * x / ((a + twice) * (qap + twice))
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise EvaluationComparisonError("Statistical comparison did not converge within its numeric safety bound.")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if not 0.0 <= x <= 1.0 or a <= 0.0 or b <= 0.0:
        raise ValueError("Incomplete beta parameters are outside their numeric domain.")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    exponent = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(exponent)
    if x < (a + 1.0) / (a + b + 2.0):
        result = front * _beta_continued_fraction(a, b, x) / a
    else:
        result = 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, result))


def _welch_test(baseline: list[float], candidate: list[float]) -> dict[str, Any]:
    """Compute a two-sided Welch t-test only when both groups have replicates."""
    if len(baseline) < 2 or len(candidate) < 2:
        return {
            "method": "Welch two-sided unequal-variance t-test",
            "status": "insufficient_replicates",
            "minimum_replicates_per_group": 2,
            "t_statistic": None,
            "degrees_of_freedom": None,
            "p_value": None,
        }
    baseline_mean = _mean(baseline)
    candidate_mean = _mean(candidate)
    baseline_variance = _sample_variance(baseline, baseline_mean)
    candidate_variance = _sample_variance(candidate, candidate_mean)
    baseline_term = baseline_variance / len(baseline)
    candidate_term = candidate_variance / len(candidate)
    standard_error_squared = baseline_term + candidate_term
    if standard_error_squared == 0.0:
        return {
            "method": "Welch two-sided unequal-variance t-test",
            "status": "degenerate_zero_variance",
            "minimum_replicates_per_group": 2,
            "t_statistic": None,
            "degrees_of_freedom": None,
            "p_value": None,
            "degenerate_zero_variance": True,
            "note": "Both groups have zero within-group variance; a finite t statistic and inferential p-value are undefined.",
        }
    standard_error = math.sqrt(standard_error_squared)
    statistic = (candidate_mean - baseline_mean) / standard_error
    denominator = (
        baseline_term * baseline_term / (len(baseline) - 1)
        + candidate_term * candidate_term / (len(candidate) - 1)
    )
    if denominator <= 0.0:
        raise EvaluationComparisonError("Statistical comparison produced an invalid variance estimate.")
    degrees = standard_error_squared * standard_error_squared / denominator
    beta_argument = degrees / (degrees + statistic * statistic)
    p_value = _regularized_beta(beta_argument, degrees / 2.0, 0.5)
    if not all(math.isfinite(value) for value in (statistic, degrees, p_value)):
        raise EvaluationComparisonError("Statistical comparison produced non-finite output.")
    return {
        "method": "Welch two-sided unequal-variance t-test",
        "status": "computed",
        "minimum_replicates_per_group": 2,
        "t_statistic": statistic,
        "degrees_of_freedom": degrees,
        "p_value": min(1.0, max(0.0, p_value)),
        "degenerate_zero_variance": False,
    }


def _collect_replicates(
    records: list[dict[str, Any]],
    experiment_id: str,
    metric: str,
) -> tuple[list[dict[str, Any]], int]:
    linked = [record for record in records if record.get("experiment_id") == experiment_id]
    eligible: list[dict[str, Any]] = []
    excluded_partial = 0
    for record in linked:
        report = record.get("report")
        manifest = report.get("dataset_manifest") if isinstance(report, dict) else None
        if (
            record.get("status") != "complete"
            or not isinstance(report, dict)
            or report.get("truncated") is not False
            or not isinstance(manifest, dict)
            or manifest.get("full_content_hashed") is not True
            or manifest.get("comparison_corpus_fully_hashed") is not True
        ):
            excluded_partial += 1
            continue
        metrics = report.get("metrics")
        score = metrics.get(metric) if isinstance(metrics, dict) else None
        digest = manifest.get("comparison_corpus_sha256")
        if (
            isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0
            or not isinstance(digest, str) or not _SHA256.fullmatch(digest)
        ):
            raise EvaluationComparisonError("A linked evaluation run has an invalid metric or corpus fingerprint.")
        fields = report.get("fields")
        counts = report.get("counts")
        if not isinstance(fields, dict) or not isinstance(counts, dict):
            raise EvaluationComparisonError("A linked evaluation run is missing comparison metadata.")
        sample_count = counts.get("valid_samples")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise EvaluationComparisonError("A linked evaluation run has an invalid sample count.")
        eligible.append({
            "id": record.get("id"),
            "created_at": record.get("created_at"),
            "model_label": record.get("model_label"),
            "score": float(score),
            "corpus_sha256": digest,
            "fields": {
                "reference": fields.get("reference"),
                "split": fields.get("split"),
                "selected_split": fields.get("selected_split"),
            },
            "sample_count": sample_count,
        })
    if not linked:
        raise EvaluationComparisonError(f"Experiment {experiment_id} has no linked offline evaluation runs.")
    if not eligible:
        raise EvaluationComparisonError(f"Experiment {experiment_id} has no complete comparable evaluation runs.")
    return eligible, excluded_partial


def compare_experiment_evaluations(
    baseline_experiment_id: str,
    candidate_experiment_id: str,
    *,
    metric: str = "token_f1_micro",
) -> dict[str, Any]:
    """Compare repeated, complete evaluations for two immutable experiment snapshots."""
    if metric not in _METRICS:
        raise ValueError(f"Metric must be one of: {', '.join(sorted(_METRICS))}.")
    if baseline_experiment_id == candidate_experiment_id:
        raise ValueError("Baseline and candidate experiment IDs must be different.")
    baseline_experiment = validate_saved_experiment(baseline_experiment_id)
    candidate_experiment = validate_saved_experiment(candidate_experiment_id)
    try:
        page = EvaluationStore().list_page(limit=_MAX_EVALUATION_RUNS, offset=0)
    except (OSError, ValueError):
        raise EvaluationComparisonError("Evaluation history could not be read safely.") from None
    if page["has_more"]:
        raise EvaluationComparisonError("Evaluation history exceeds the comparison scan limit.")
    baseline_runs, baseline_excluded = _collect_replicates(page["runs"], baseline_experiment_id, metric)
    candidate_runs, candidate_excluded = _collect_replicates(page["runs"], candidate_experiment_id, metric)
    combined = [*baseline_runs, *candidate_runs]
    digests = {item["corpus_sha256"] for item in combined}
    fields = {(
        item["fields"]["reference"],
        item["fields"]["split"],
        item["fields"]["selected_split"],
        item["sample_count"],
    ) for item in combined}
    if len(digests) != 1 or len(fields) != 1:
        raise EvaluationComparisonError(
            "Linked evaluation replicates must use the same complete comparison corpus, reference field, split, and sample count."
        )

    baseline_scores = [item["score"] for item in baseline_runs]
    candidate_scores = [item["score"] for item in candidate_runs]
    baseline_mean = _mean(baseline_scores)
    candidate_mean = _mean(candidate_scores)
    baseline_variance = _sample_variance(baseline_scores, baseline_mean) if len(baseline_scores) >= 2 else None
    candidate_variance = _sample_variance(candidate_scores, candidate_mean) if len(candidate_scores) >= 2 else None
    baseline_summary = {
        "experiment_id": baseline_experiment_id,
        "experiment_name": baseline_experiment["name"],
        "plan_fingerprint": reproducible_plan_document(baseline_experiment)["fingerprint"]["value"],
        "replicate_count": len(baseline_scores),
        "excluded_incomplete_count": baseline_excluded,
        "mean": baseline_mean,
        "sample_standard_deviation": math.sqrt(baseline_variance) if baseline_variance is not None else None,
        "minimum": min(baseline_scores),
        "maximum": max(baseline_scores),
        "run_ids": [item["id"] for item in baseline_runs],
    }
    candidate_summary = {
        "experiment_id": candidate_experiment_id,
        "experiment_name": candidate_experiment["name"],
        "plan_fingerprint": reproducible_plan_document(candidate_experiment)["fingerprint"]["value"],
        "replicate_count": len(candidate_scores),
        "excluded_incomplete_count": candidate_excluded,
        "mean": candidate_mean,
        "sample_standard_deviation": math.sqrt(candidate_variance) if candidate_variance is not None else None,
        "minimum": min(candidate_scores),
        "maximum": max(candidate_scores),
        "run_ids": [item["id"] for item in candidate_runs],
    }
    return {
        "schema_version": 1,
        "comparison_type": "axiom-repeated-offline-evaluation",
        "metric": metric,
        "corpus": {
            "comparison_corpus_sha256": next(iter(digests)),
            "reference_field": next(iter(fields))[0],
            "split_field": next(iter(fields))[1],
            "selected_split": next(iter(fields))[2],
            "sample_count": next(iter(fields))[3],
        },
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "difference": {
            "candidate_minus_baseline_mean": candidate_mean - baseline_mean,
            "direction": "improved" if candidate_mean > baseline_mean else "regressed" if candidate_mean < baseline_mean else "unchanged",
        },
        "statistical_test": _welch_test(baseline_scores, candidate_scores),
        "generated_at": datetime.now(UTC).isoformat(),
        "scope_note": (
            "This compares aggregate offline evaluation metrics linked to saved planning snapshots. "
            "Welch's test assumes independent repeated evaluation runs; p-values are omitted until each group "
            "has at least two complete replicates. A plan snapshot is not proof that training was executed, "
            "and a model label or experiment association is user-supplied unless separately evidenced by provenance."
        ),
    }
