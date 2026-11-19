import json

from axiom.core.experiments import ExperimentStore
from axiom.core.hardware import HardwareInfo


def test_experiment_store_persists_newest_records_first(tmp_path):
    store = ExperimentStore(tmp_path)
    first = {"id": "first", "name": "First plan"}
    second = {"id": "second", "name": "Second plan"}

    store.add(first)
    store.add(second)

    assert store.list() == [
        {**second, "schema_version": 1, "tags": []},
        {**first, "schema_version": 1, "tags": []},
    ]
    assert store.get("first") == {**first, "schema_version": 1, "tags": []}


def test_experiment_store_bounds_list_results_and_rejects_bad_limits(tmp_path):
    store = ExperimentStore(tmp_path)
    store.add({"id": "one"})

    assert store.list(1) == [{"id": "one", "schema_version": 1, "tags": []}]
    try:
        store.list(0)
    except ValueError as exc:
        assert "between 1 and 1000" in str(exc)
    else:
        raise AssertionError("invalid history limit should be rejected")


def test_workspace_api_saves_and_reads_hardware_aware_plan(tmp_path, monkeypatch):
    from axiom.api import workspace

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        workspace,
        "detect_hardware",
        lambda: HardwareInfo("TestOS", "x86_64", 8, 32.0, "Test GPU", 16.0, True),
    )

    saved = workspace.workspace_add_experiment(
        workspace.WorkspaceExperimentInput(
            name="First local plan",
            parameter_billions=7,
            method="qlora",
        )
    )["experiment"]

    assert saved["status"] == "planned"
    assert saved["hardware"]["gpu_name"] == "Test GPU"
    assert saved["plan"]["method"] == "qlora"
    assert workspace.workspace_experiments(limit=100, offset=0, q=None, tag=None)["experiments"] == [saved]
    assert workspace.workspace_experiment(saved["id"])["experiment"] == saved


def test_experiment_store_search_metadata_edit_and_deletion(tmp_path):
    store = ExperimentStore(tmp_path)
    first = {"id": "first", "name": "baseline run", "notes": "small", "tags": ["baseline"], "plan": {"method": "auto"}}
    second = {"id": "second", "name": "fine tune", "notes": "large", "tags": ["candidate"], "plan": {"method": "lora"}}
    store.add(first)
    store.add(second)

    page = store.list_page(limit=1, offset=0, query="fine", tag="candidate")
    assert page["total"] == 1
    assert page["experiments"][0]["id"] == "second"
    updated = store.update_metadata("second", name="candidate v2", notes="reviewed", tags=["candidate", "ready"])
    assert updated["plan"] == second["plan"]
    assert updated["name"] == "candidate v2"
    assert store.list_page(tag="ready")["total"] == 1
    deleted = store.delete("second")
    assert deleted["id"] == "second"
    assert store.get("second") is None


def test_experiment_store_migrates_unversioned_history_and_rejects_future_schema(tmp_path):
    store = ExperimentStore(tmp_path)
    store.root.mkdir(exist_ok=True)
    store.records_file.write_text('[{"id":"legacy","name":"Old plan"}]', encoding="utf-8")

    assert store.get("legacy")["schema_version"] == 1
    store.update_metadata("legacy", name="Migrated plan", notes=None, tags=[])
    assert json.loads(store.records_file.read_text(encoding="utf-8"))[0]["schema_version"] == 1

    store.records_file.write_text('[{"id":"future","schema_version":2}]', encoding="utf-8")
    try:
        store.list()
    except ValueError as exc:
        assert "Unsupported experiment schema version" in str(exc)
    else:
        raise AssertionError("future experiment schemas must not be misread")


def test_experiment_api_paginates_updates_metadata_and_deletes(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from axiom.api.server import create_app

    monkeypatch.chdir(tmp_path)
    with TestClient(create_app()) as client:
        create = client.post(
            "/api/workspace/experiments",
            json={"name": "Planning record", "parameter_billions": 3, "tags": ["baseline"]},
        )
        assert create.status_code == 201
        saved = create.json()["experiment"]

        page = client.get("/api/workspace/experiments", params={"q": "planning", "tag": "baseline"})
        assert page.status_code == 200
        assert page.json()["total"] == 1
        assert page.json()["experiments"][0]["id"] == saved["id"]
        assert page.json()["has_more"] is False

        update = client.patch(
            f"/api/workspace/experiments/{saved['id']}",
            json={"name": "Baseline plan", "notes": "edit metadata only", "tags": ["baseline", "approved"]},
        )
        assert update.status_code == 200
        assert update.json()["plan_immutable"] is True
        assert update.json()["experiment"]["plan"] == saved["plan"]

        deleted = client.delete(f"/api/workspace/experiments/{saved['id']}")
        assert deleted.status_code == 200
        assert deleted.json()["run_artifacts_deleted"] is False
        assert client.get("/api/workspace/experiments").json()["total"] == 0


def test_api_saves_only_the_exact_current_plan_snapshot(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from axiom.api import workspace
    from axiom.api.server import create_app
    from axiom.training.planner import create_training_plan

    hardware = HardwareInfo("TestOS", "x86_64", 8, 32.0, "Test GPU", 16.0, True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(workspace, "detect_hardware", lambda: hardware)
    plan = create_training_plan(7, hardware, "qlora").__dict__

    with TestClient(create_app()) as client:
        saved = client.post(
            "/api/workspace/experiments",
            json={
                "name": "Exact displayed plan",
                "parameter_billions": 7,
                "method": "qlora",
                "plan_snapshot": plan,
            },
        )
        assert saved.status_code == 201, saved.text
        record = saved.json()["experiment"]
        assert record["schema_version"] == 1
        assert record["plan"] == plan

        stale_plan = {**plan, "learning_rate": 1}
        stale = client.post(
            "/api/workspace/experiments",
            json={
                "name": "Stale plan",
                "parameter_billions": 7,
                "method": "qlora",
                "plan_snapshot": stale_plan,
            },
        )
        assert stale.status_code == 409
        assert client.get("/api/workspace/experiments").json()["total"] == 1


def test_experiment_cli_search_update_and_delete(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from axiom.cli.main import app

    monkeypatch.chdir(tmp_path)
    store = ExperimentStore()
    store.add({"id": "cli-plan", "name": "Baseline candidate", "notes": None, "tags": ["baseline"], "status": "planned"})
    runner = CliRunner()

    listed = runner.invoke(app, ["experiment", "list", "--query", "candidate", "--tag", "baseline", "--json"])
    assert listed.exit_code == 0, listed.stdout
    assert json.loads(listed.stdout)[0]["id"] == "cli-plan"

    updated = runner.invoke(app, ["experiment", "update", "cli-plan", "--name", "Approved", "--tag", "reviewed"])
    assert updated.exit_code == 0, updated.stdout
    assert json.loads(updated.stdout)["experiment"]["tags"] == ["reviewed"]

    deleted = runner.invoke(app, ["experiment", "delete", "cli-plan", "--yes"])
    assert deleted.exit_code == 0, deleted.stdout
    assert json.loads(deleted.stdout)["run_artifacts_deleted"] is False


def test_cli_plan_can_save_the_exact_generated_snapshot(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from axiom.cli import main
    from axiom.cli.main import app

    hardware = HardwareInfo("TestOS", "x86_64", 8, 32.0, "Test GPU", 16.0, True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "detect_hardware", lambda: hardware)

    result = CliRunner().invoke(
        app,
        ["train", "plan", "7", "--method", "qlora", "--save", "--name", "CLI snapshot", "--tag", "baseline", "--json"],
    )

    assert result.exit_code == 0, result.stdout
    output = json.loads(result.stdout)
    assert output["status"] == "planning_only"
    assert output["executed"] is False
    assert output["experiment"]["schema_version"] == 1
    assert output["experiment"]["plan"] == output["plan"]
    stored = ExperimentStore().get(output["experiment"]["id"])
    assert stored["plan"] == output["plan"]
    assert stored["tags"] == ["baseline"]
