from pathlib import Path

from typer.testing import CliRunner

from axiom.cli.main import app

runner = CliRunner()


def test_guide_detects_complete_project_and_lists_next_commands(
    tmp_path: Path,
    monkeypatch,
):
    for folder in (
        "data",
        "models",
        "experiments",
        "evaluations",
        "outputs",
    ):
        (tmp_path / folder).mkdir()

    (tmp_path / "axiom.yaml").write_text(
        "project:\n  name: demo\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# demo\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["guide"])

    assert result.exit_code == 0
    assert "AXIOM is a local-first command-line tool" in result.stdout
    assert "AXIOM project: detected (axiom.yaml found)" in result.stdout
    assert "standard project paths are present" in result.stdout
    assert "axiom config validate" in result.stdout
    assert "axiom status" in result.stdout
    assert "axiom model list" in result.stdout


def test_guide_suggests_initialization_outside_project(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["guide"])

    assert result.exit_code == 0
    assert "AXIOM project: not detected" in result.stdout
    assert "axiom init my-ai" in result.stdout
    assert "cd my-ai" in result.stdout
    assert "axiom guide" in result.stdout


def test_check_reports_incomplete_project_without_network_or_backend_steps(
    tmp_path: Path,
    monkeypatch,
):
    (tmp_path / "axiom.yaml").write_text(
        "project:\n  name: demo\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["check"])

    assert result.exit_code == 0
    assert "AXIOM project detected" in result.stdout
    assert "Missing standard paths" in result.stdout
    assert "README.md" in result.stdout
    assert "axiom config validate" in result.stdout
    assert "http" not in result.stdout.lower()


def test_help_describes_local_observed_behavior():
    model_result = runner.invoke(app, ["model", "search", "--help"])
    project_result = runner.invoke(app, ["project", "validate", "--help"])

    assert model_result.exit_code == 0
    assert "local model paths" in model_result.stdout
    assert project_result.exit_code == 0
    assert "expected AXIOM project paths" in project_result.stdout


def test_gui_help_exposes_refresh_option():
    result = runner.invoke(app, ["gui", "--help"])
    assert result.exit_code == 0
    assert "--refresh" in result.stdout
    assert "browser storage" in result.stdout


def test_refresh_control_center_runs_lint_then_build_without_deleting_ui(tmp_path, monkeypatch):
    import types
    import axiom.cli.main as main_cli

    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3,"packages":{"":{}}}', encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/copy_webui.mjs").write_text("// copy", encoding="utf-8")
    (tmp_path / "axiom/webui").mkdir(parents=True)
    previous_index = tmp_path / "axiom/webui/index.html"
    previous_index.write_text("previous UI", encoding="utf-8")
    calls = []
    monkeypatch.setattr(main_cli.shutil, "which", lambda command: "/usr/bin/npm")
    monkeypatch.setattr(main_cli, "_frontend_runtime_typescript_mismatches", lambda root: [])

    def fake_run(args, *, cwd, check):
        calls.append((args, cwd, check))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(main_cli.subprocess, "run", fake_run)
    main_cli._refresh_control_center(tmp_path)

    assert [call[0][-1] for call in calls] == ["lint", "build"]
    assert all(call[1] == tmp_path and call[2] is False for call in calls)
    assert previous_index.read_text(encoding="utf-8") == "previous UI"


def test_frontend_dependency_audit_detects_missing_bundled_lucide_types(tmp_path):
    import json
    import axiom.cli.main as main_cli

    package_root = tmp_path / "node_modules" / "lucide-react"
    package_root.mkdir(parents=True)
    (package_root / "package.json").write_text(
        json.dumps({
            "name": "lucide-react",
            "version": "1.77.0",
            "typings": "dist/lucide-react.d.ts",
        }),
        encoding="utf-8",
    )
    lock = {
        "lockfileVersion": 3,
        "packages": {
            "": {"dependencies": {"lucide-react": "^1.77.0"}},
            "node_modules/lucide-react": {"version": "1.77.0"},
        },
    }
    (tmp_path / "package-lock.json").write_text(json.dumps(lock), encoding="utf-8")

    missing = main_cli._frontend_dependency_mismatches(tmp_path)
    assert missing == [
        "lucide-react: bundled TypeScript declarations are missing; restore node_modules with npm ci"
    ]

    declarations = package_root / "dist" / "lucide-react.d.ts"
    declarations.parent.mkdir()
    declarations.write_text("export type LucideIcon = unknown;\n", encoding="utf-8")
    assert main_cli._frontend_dependency_mismatches(tmp_path) == []


def test_refresh_detects_typescript_version_resolved_by_eslint(tmp_path, monkeypatch):
    import json
    import types
    import axiom.cli.main as main_cli

    packages = {
        "node_modules/typescript": {"version": "6.0.3"},
        "node_modules/typescript-eslint": {"version": "8.73.0"},
    }
    (tmp_path / "package-lock.json").write_text(
        json.dumps({"lockfileVersion": 3, "packages": packages}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        main_cli.shutil,
        "which",
        lambda command: "/usr/bin/node" if command in {"node", "nodejs"} else None,
    )

    def fake_run(args, *, cwd, capture_output, text, check, timeout):
        assert args[:2] == ["/usr/bin/node", "-e"]
        assert cwd == tmp_path and capture_output and text and not check
        assert timeout == 10
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps([
                {"importer": "@typescript-eslint/parser", "version": "7.0.0"},
                {"importer": "@typescript-eslint/typescript-estree", "version": "7.0.0"},
            ]),
            stderr="",
        )

    monkeypatch.setattr(main_cli.subprocess, "run", fake_run)
    mismatches = main_cli._frontend_runtime_typescript_mismatches(tmp_path)

    assert mismatches == [
        "@typescript-eslint/parser resolves TypeScript 7.0.0, but package-lock.json pins 6.0.3",
        "@typescript-eslint/typescript-estree resolves TypeScript 7.0.0, but package-lock.json pins 6.0.3",
    ]


def test_refresh_reinstalls_locked_typescript_before_lint_and_build(tmp_path, monkeypatch):
    import types
    import axiom.cli.main as main_cli

    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "package-lock.json").write_text(
        '{"lockfileVersion":3,"packages":{"":{}}}', encoding="utf-8"
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/copy_webui.mjs").write_text("// copy", encoding="utf-8")
    (tmp_path / "axiom/webui").mkdir(parents=True)
    previous_index = tmp_path / "axiom/webui/index.html"
    previous_index.write_text("previous UI", encoding="utf-8")
    calls = []
    runtime_checks = iter([
        ["@typescript-eslint/parser resolves TypeScript 7.0.0, but package-lock.json pins 6.0.3"],
        [],
    ])
    monkeypatch.setattr(
        main_cli.shutil,
        "which",
        lambda command: {
            "npm": "/usr/bin/npm",
            "node": "/usr/bin/node",
            "nodejs": None,
        }.get(command),
    )
    monkeypatch.setattr(
        main_cli,
        "_frontend_runtime_typescript_mismatches",
        lambda root: next(runtime_checks),
    )

    def fake_run(args, *, cwd, check):
        calls.append((args, cwd, check))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(main_cli.subprocess, "run", fake_run)
    main_cli._refresh_control_center(tmp_path)

    assert [call[0][1:] for call in calls] == [
        ["ci", "--include=dev", "--no-audit", "--no-fund"],
        ["run", "lint"],
        ["run", "build"],
    ]
    assert all(call[1] == tmp_path and call[2] is False for call in calls)
    assert previous_index.read_text(encoding="utf-8") == "previous UI"


def test_update_cli_check_never_installs(monkeypatch):
    import axiom.cli.main as main_cli

    monkeypatch.setattr(main_cli.updater, "check_for_update", lambda current, channel: {
        "channel": channel, "current_version": current, "latest_version": "9.9.9",
        "update_available": True, "release": {"html_url": "https://example.invalid/release"},
    })
    monkeypatch.setattr(
        main_cli.updater,
        "download_and_install",
        lambda version: (_ for _ in ()).throw(AssertionError("check must not install")),
    )
    result = runner.invoke(app, ["--no-banner", "update", "--check"])
    assert result.exit_code == 0, result.output
    assert "Update available" in result.stdout
    assert "axiom update --auto" in result.stdout


def test_update_cli_auto_installs_without_prompt(monkeypatch):
    import axiom.cli.main as main_cli

    observed = []
    monkeypatch.setattr(main_cli.updater, "check_for_update", lambda current, channel: {
        "channel": channel, "current_version": current, "latest_version": "9.9.9",
        "update_available": True, "release": {"html_url": "https://example.invalid/release"},
    })
    monkeypatch.setattr(main_cli.updater, "download_and_install", lambda version: (
        observed.append(version) or {
            "installed_version": version, "user_data_preserved": True, "restart_required": True,
        }
    ))
    result = runner.invoke(app, ["--no-banner", "update", "--auto"])
    assert result.exit_code == 0, result.output
    assert observed == ["9.9.9"]
    assert "browser storage were left in place" in result.stdout
