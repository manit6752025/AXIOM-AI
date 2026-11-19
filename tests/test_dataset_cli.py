import hashlib
import json

from typer.testing import CliRunner

from axiom.cli.main import app


runner = CliRunner()


def test_dataset_profile_cli_supports_explicit_redacted_pii_scan(tmp_path):
    source = tmp_path / "private.jsonl"
    secret_email = "person.private@example.com"
    source.write_text(json.dumps({"contact": secret_email}) + "\n", encoding="utf-8")

    result = runner.invoke(app, ["dataset", "profile", str(source), "--scan-pii"])

    assert result.exit_code == 0, result.stdout
    assert "PII pattern matches" in result.stdout
    assert "email 1" in result.stdout
    assert secret_email not in result.stdout


def test_dataset_manifest_cli_outputs_parent_lineage(tmp_path):
    source = tmp_path / "train.jsonl"
    contents = b'{"id":1}\n'
    source.write_bytes(contents)
    parent_digest = hashlib.sha256(b"parent").hexdigest()

    result = runner.invoke(
        app,
        [
            "dataset",
            "manifest",
            str(source),
            "--parent-manifest-sha256",
            parent_digest,
        ],
    )

    assert result.exit_code == 0, result.stdout
    report = json.loads(result.stdout)
    assert report["sha256"] == hashlib.sha256(contents).hexdigest()
    assert report["lineage"]["parent_manifest_sha256"] == parent_digest


def test_dataset_clean_preview_cli_does_not_change_source(tmp_path):
    source = tmp_path / "train.jsonl"
    contents = b'{"id":1}\n{"id":1}\n'
    source.write_bytes(contents)

    result = runner.invoke(app, ["dataset", "preview-clean", str(source)])

    assert result.exit_code == 0, result.stdout
    report = json.loads(result.stdout)
    assert report["preview_only"] is True
    assert report["removed_duplicate_rows"] == 1
    assert report["output_written"] is False
    assert source.read_bytes() == contents
