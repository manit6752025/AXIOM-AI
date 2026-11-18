from __future__ import annotations

import json

import pytest

from axiom.core import fine_tuning


def test_fine_tuning_job_history_is_bounded_and_pages_are_stable(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    for index in range(7):
        job_id = f"{index + 1:032x}"
        (jobs_dir / f"{job_id}.json").write_text(
            json.dumps({"id": job_id, "created_at": f"2026-11-{18-index:02d}T10:00:00+00:00"}),
            encoding="utf-8",
        )

    monkeypatch.setattr(fine_tuning, "jobs_directory", lambda: jobs_dir)
    monkeypatch.setattr(fine_tuning, "_recover_if_stale", lambda record: record)
    monkeypatch.setattr(fine_tuning, "_public_training_job", lambda record: record)

    first = fine_tuning.list_jobs_page(limit=3, offset=0)
    second = fine_tuning.list_jobs_page(limit=3, offset=3)
    assert first["total"] == 7
    assert first["has_more"] is True
    assert second["has_more"] is True
    assert len(first["jobs"]) == 3
    assert len(second["jobs"]) == 3
    assert set(job["id"] for job in first["jobs"]).isdisjoint(
        job["id"] for job in second["jobs"]
    )


@pytest.mark.parametrize("limit,offset", [(0, 0), (101, 0), (1, -1), (1, 5001), (True, 0)])
def test_fine_tuning_job_history_rejects_invalid_page_bounds(tmp_path, monkeypatch, limit, offset):
    monkeypatch.setattr(fine_tuning, "jobs_directory", lambda: tmp_path / "missing")
    with pytest.raises(ValueError):
        fine_tuning.list_jobs_page(limit=limit, offset=offset)


def test_fine_tuning_job_history_refuses_symlinked_directory(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "jobs"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")
    monkeypatch.setattr(fine_tuning, "jobs_directory", lambda: linked)
    with pytest.raises(fine_tuning.FineTuningError, match="symbolic link"):
        fine_tuning.list_jobs_page()
