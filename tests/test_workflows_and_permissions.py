import json

from axiom.core.tool_permissions import TOOL_SCOPES, ToolPermissionStore
from axiom.core.workflows import WorkflowStore, validate_workflow_graph


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
    listed = {item["name"]: item for item in store.list()["tools"]}
    assert listed["axiom_dataset_clean"]["scopes"] == ["write", "read"]
    assert listed["axiom_engine_generate"]["scopes"] == ["execution", "read"]
    assert set(TOOL_SCOPES) == set(listed)

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
        assert workflow["schema_version"] == 2
        assert workflow["revision"] == 1
        assert workflow["edges"][0] == {
            "source": "dataset",
            "target": "plan",
            "source_port": "dataset",
            "target_port": "dataset",
        }
        assert workflow["nodes"][1]["ports"]["inputs"][0] == {
            "id": "dataset", "data_type": "dataset", "required": True,
        }
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
        templates = template_response.json()["templates"]
        assert {item["id"] for item in templates} >= {
            "dataset-audit",
            "supervised-finetuning",
        }
        supervised = next(item for item in templates if item["id"] == "supervised-finetuning")
        assert supervised["schema_version"] == 2
        plan_edges = [edge for edge in supervised["edges"] if edge["target"] == "plan"]
        assert {(edge["source"], edge["source_port"], edge["target_port"]) for edge in plan_edges} == {
            ("dataset", "dataset", "dataset"),
            ("model", "model", "model"),
        }

        validation = client.post("/api/workflows/validate", json=workflow_payload)
        assert validation.status_code == 200
        assert validation.json()["valid"] is True
        assert validation.json()["missing_inputs"] == [{
            "node_id": "plan",
            "port_id": "model",
            "reason": "Required input is not connected.",
        }]

        invalid_graph = {**workflow_payload, "edges": [
            {"source": "dataset", "target": "plan"},
            {"source": "plan", "target": "dataset"},
        ]}
        assert client.post("/api/workflows/validate", json=invalid_graph).status_code == 422

        wrong_port = {**workflow_payload, "edges": [{
            "source": "dataset", "target": "plan",
            "source_port": "dataset", "target_port": "model",
        }]}
        assert client.post("/api/workflows/validate", json=wrong_port).status_code == 422

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



def test_workflow_typed_ports_accept_matching_dataset_and_model_inputs():
    nodes = [
        {"id": "dataset", "type": "dataset.profile"},
        {"id": "model", "type": "model.inspect"},
        {"id": "plan", "type": "training.plan"},
    ]
    edges = [
        {
            "source": "dataset", "target": "plan",
            "source_port": "dataset", "target_port": "dataset",
        },
        {
            "source": "model", "target": "plan",
            "source_port": "model", "target_port": "model",
        },
    ]
    assert validate_workflow_graph("Complete training plan", nodes, edges) == []


def test_workflow_graph_reports_missing_required_and_required_any_inputs():
    nodes = [
        {"id": "dataset", "type": "dataset.profile"},
        {"id": "plan", "type": "training.plan"},
        {"id": "report", "type": "report.export"},
    ]
    edges = [{
        "source": "dataset", "target": "plan",
        "source_port": "dataset", "target_port": "dataset",
    }]
    missing = validate_workflow_graph("Incomplete workflow", nodes, edges)
    assert missing == [
        {"node_id": "plan", "port_id": "model", "reason": "Required input is not connected."},
        {
            "node_id": "report",
            "port_id": "one_of:dataset|plan|evaluation",
            "reason": "Connect at least one input to this node.",
        },
    ]


def test_workflow_graph_rejects_unknown_mismatched_and_occupied_ports():
    nodes = [
        {"id": "dataset-a", "type": "dataset.profile"},
        {"id": "dataset-b", "type": "dataset.profile"},
        {"id": "plan", "type": "training.plan"},
    ]
    invalid_graphs = [
        [{"source": "dataset-a", "target": "plan", "source_port": "missing", "target_port": "dataset"}],
        [{"source": "dataset-a", "target": "plan", "source_port": "dataset", "target_port": "model"}],
        [
            {"source": "dataset-a", "target": "plan", "source_port": "dataset", "target_port": "dataset"},
            {"source": "dataset-b", "target": "plan", "source_port": "dataset", "target_port": "dataset"},
        ],
    ]
    for edges in invalid_graphs:
        try:
            validate_workflow_graph("Invalid ports", nodes, edges)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid workflow ports should be rejected: {edges}")


def test_workflow_schema_v1_legacy_edges_migrate_to_schema_v2(tmp_path):
    registry = tmp_path / "workflows.json"
    registry.write_text(json.dumps([{
        "id": "legacy-workflow",
        "name": "Legacy workflow",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "schema_version": 1,
        "revision": 1,
        "versions": [],
        "nodes": [
            {"id": "dataset", "type": "dataset.profile"},
            {"id": "model", "type": "model.inspect"},
        ],
        "edges": [{"source": "dataset", "target": "model"}],
        "execution_status": "not_executable",
    }]), encoding="utf-8")

    store = WorkflowStore(tmp_path)
    migrated = store.get("legacy-workflow")
    assert migrated is not None
    assert migrated["schema_version"] == 2
    assert migrated["nodes"][0]["ports"]["outputs"][0]["id"] == "dataset"
    assert migrated["nodes"][1]["ports"]["inputs"][0]["id"] == "artifact"
    assert migrated["edges"] == [{
        "source": "dataset",
        "target": "model",
        "source_port": "artifact",
        "target_port": "artifact",
    }]

    updated = store.update(
        "legacy-workflow",
        name="Migrated workflow",
        nodes=migrated["nodes"],
        edges=migrated["edges"],
    )
    assert updated is not None
    persisted = json.loads(registry.read_text(encoding="utf-8"))
    assert persisted[0]["schema_version"] == 2
    assert persisted[0]["edges"][0]["source_port"] == "artifact"
    assert persisted[0]["versions"][0]["edges"][0]["target_port"] == "artifact"
