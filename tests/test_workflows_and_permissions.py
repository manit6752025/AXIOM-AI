import json

from axiom.core.tool_permissions import ToolPermissionStore
from axiom.core.workflows import validate_workflow_graph


def test_tool_permissions_deny_by_default_and_follow_active_profile(tmp_path):
    store = ToolPermissionStore(tmp_path)

    assert store.allows("axiom_system_info") is False
    try:
        store.require("axiom_system_info")
    except PermissionError as exc:
        assert "denied" in str(exc)
    else:
        raise AssertionError("an ungranted MCP tool should be denied")

    store.save_profile("read-only", ["axiom_system_info"])
    store.set_active("read-only")

    assert store.allows("axiom_system_info") is True
    assert store.allows("axiom_dataset_clean") is False

    try:
        store.create_profile("read-only")
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("creating a profile must not overwrite an existing profile")


def test_tool_permission_store_rejects_malformed_profile_data(tmp_path):
    store = ToolPermissionStore(tmp_path)
    store.root.mkdir(exist_ok=True)
    store.policy_file.write_text(
        json.dumps({
            "active_profile": "default",
            "profiles": {"default": {"name": "default", "allowed_tools": [["bad"]]}},
        }),
        encoding="utf-8",
    )

    try:
        store.list()
    except ValueError as exc:
        assert "Invalid MCP profile" in str(exc)
    else:
        raise AssertionError("malformed permission data must fail closed")


def test_workflow_graph_rejects_cycles_and_unknown_nodes():
    nodes = [
        {"id": "dataset", "type": "dataset.profile"},
        {"id": "plan", "type": "training.plan"},
    ]

    validate_workflow_graph("Valid plan", nodes, [{"source": "dataset", "target": "plan"}])

    try:
        validate_workflow_graph(
            "Cycle",
            nodes,
            [
                {"source": "dataset", "target": "plan"},
                {"source": "plan", "target": "dataset"},
            ],
        )
    except ValueError as exc:
        assert "acyclic" in str(exc)
    else:
        raise AssertionError("cyclic workflow graphs should be rejected")

    try:
        validate_workflow_graph("Invalid", [{"id": "x", "type": "exec.shell"}], [])
    except ValueError as exc:
        assert "Unsupported workflow node type" in str(exc)
    else:
        raise AssertionError("unsupported workflow nodes should be rejected")


def test_workflow_graph_rejects_invalid_edge_types():
    try:
        validate_workflow_graph(
            "Invalid edge",
            [{"id": "dataset", "type": "dataset.profile"}],
            [{"source": [], "target": "dataset"}],
        )
    except ValueError as exc:
        assert "existing node" in str(exc)
    else:
        raise AssertionError("non-string edge IDs should be rejected")


def test_workflow_and_mcp_profiles_are_available_through_workspace_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from axiom.api.server import create_app

    monkeypatch.chdir(tmp_path)
    with TestClient(create_app()) as client:
        invalid_request = client.get(
            "/api/workflows?limit=not-a-number",
            headers={"X-Request-ID": "test_trace_123"},
        )
        assert invalid_request.status_code == 422
        assert invalid_request.headers["X-Request-ID"] == "test_trace_123"
        assert invalid_request.json()["request_id"] == "test_trace_123"
        assert invalid_request.json()["error"]["code"] == "validation_error"
        assert "input" not in invalid_request.json()["detail"][0]

        policy = client.get("/api/security/tool-permissions")
        assert policy.status_code == 200
        assert policy.json()["active_profile"] == "default"
        assert policy.json()["profiles"] == [{"name": "default", "allowed_tools": []}]

        workflow_payload = {
            "name": "Dataset to plan",
            "nodes": [
                {"id": "dataset", "type": "dataset.profile"},
                {"id": "plan", "type": "training.plan"},
            ],
            "edges": [{"source": "dataset", "target": "plan"}],
        }
        created_workflow = client.post("/api/workflows", json=workflow_payload)
        assert created_workflow.status_code == 201
        workflow = created_workflow.json()["workflow"]
        assert workflow["execution_status"] == "not_executable"
        assert workflow["schema_version"] == 1
        assert workflow["revision"] == 1
        assert client.get("/api/workflows").json()["workflows"] == [workflow]

        dry_run = client.post(
            f"/api/workflows/{workflow['id']}/dry-run",
            json={"parameter_billions": 7, "method": "auto", "seed": 1234},
        )
        assert dry_run.status_code == 200
        dry_run_body = dry_run.json()["dry_run"]
        assert dry_run_body["status"] == "planning_only"
        assert dry_run_body["execution_status"] == "not_executable"
        assert dry_run_body["executed"] is False
        assert dry_run_body["job_created"] is False
        assert dry_run_body["resource_summary"]["estimated_training_plan"]
        assert dry_run_body["unresolved_inputs"][0]["node_id"] == "dataset"
        manifest = dry_run_body["reproducibility_manifest"]
        assert manifest["seed"] == 1234
        assert manifest["seed_applied"] is False
        assert len(manifest["graph_sha256"]) == 64
        assert manifest["software"]["python"]

        template_response = client.get("/api/workflows/templates")
        assert template_response.status_code == 200
        assert {item["id"] for item in template_response.json()["templates"]} >= {
            "dataset-audit",
            "supervised-finetuning",
        }

        invalid_graph = {**workflow_payload, "edges": [
            {"source": "dataset", "target": "plan"},
            {"source": "plan", "target": "dataset"},
        ]}
        assert client.post("/api/workflows/validate", json=invalid_graph).status_code == 422

        update_payload = {**workflow_payload, "name": "Dataset to plan v2"}
        updated = client.put(f"/api/workflows/{workflow['id']}", json=update_payload)
        assert updated.status_code == 200
        updated_workflow = updated.json()["workflow"]
        assert updated_workflow["revision"] == 2
        assert len(updated_workflow["versions"]) == 1
        assert updated_workflow["versions"][0]["name"] == "Dataset to plan"
        assert len(client.get(f"/api/workflows/{workflow['id']}/versions").json()["versions"]) == 1

        duplicate = client.post(f"/api/workflows/{workflow['id']}/duplicate")
        assert duplicate.status_code == 201
        duplicate_workflow = duplicate.json()["workflow"]
        assert duplicate_workflow["id"] != workflow["id"]
        assert duplicate_workflow["revision"] == 1
        assert duplicate_workflow["versions"] == []

        created_profile = client.post("/api/security/tool-profiles/read-only")
        assert created_profile.status_code == 201
        saved_profile = client.put(
            "/api/security/tool-profiles/read-only",
            json={"allowed_tools": ["axiom_system_info"]},
        )
        assert saved_profile.status_code == 200
        assert client.put("/api/security/active-profile", json={"name": "read-only"}).status_code == 200
        assert ToolPermissionStore(tmp_path / ".axiom").allows("axiom_system_info")

        events = client.get("/api/workspace/events").json()["events"]
        assert {event["action"] for event in events} >= {"workflows.saved", "workflows.dry_run", "mcp.profile.created"}


def test_workspace_dataset_profile_accepts_bounded_scan_budgets(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from axiom.api.server import create_app

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        '{"prompt":"one"}\n{"prompt":"two"}\n{"prompt":"three"}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    with TestClient(create_app()) as client:
        response = client.get(
            "/api/workspace/dataset-profile",
            params={
                "path": "train.jsonl",
                "max_rows": 2,
                "max_bytes": 1024,
                "max_line_bytes": 128,
            },
        )
        assert response.status_code == 200
        report = response.json()["profile"]
        assert report["lines_scanned"] == 2
        assert report["valid_rows"] == 2
        assert report["truncated"] is True
        assert report["limits"] == {
            "max_bytes": 1024,
            "max_rows": 2,
            "max_line_bytes": 128,
        }
        assert client.get(
            "/api/workspace/dataset-profile",
            params={"path": "train.jsonl", "max_rows": 500_001},
        ).status_code == 422
