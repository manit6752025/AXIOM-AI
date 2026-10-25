from __future__ import annotations
import pytest
from axiom.core import model_studio as studio

def test_model_identifiers_and_files_are_validated():
    assert studio.validate_ollama_name("qwen3.5:0.8b") == "qwen3.5:0.8b"
    assert studio.validate_repo_id("example/tiny-gguf") == "example/tiny-gguf"
    assert studio.validate_gguf_filename("model-Q4_K_M.gguf") == "model-Q4_K_M.gguf"
    for value in ("https://example.com/model", "../model", "/tmp/model"):
        with pytest.raises(ValueError): studio.validate_ollama_name(value)
    for value in ("owner", "owner/../private", "https://example.com/a/b"):
        with pytest.raises(ValueError): studio.validate_repo_id(value)
    for value in ("../model.gguf", "folder/model.gguf", "model.safetensors", "https://host/file.gguf"):
        with pytest.raises(ValueError): studio.validate_gguf_filename(value)

def test_search_uses_fixed_public_catalogue_and_filters_gguf(monkeypatch):
    seen = {}
    def fake(url, timeout=12.0):
        seen["url"] = url
        return [
            {"id":"example/model-GGUF", "tags":["gguf","text-generation"],"downloads":100,"likes":4,"lastModified":"2026-01-01"},
            {"id":"example/not-related", "tags":["text-generation"],"downloads":200},
        ]
    monkeypatch.setattr(studio, "_read_public_json", fake)
    result = studio.search_gguf_catalog("test model", 5)
    assert "https://huggingface.co/api/models?" in seen["url"]
    assert result["count"] == 1
    assert result["results"][0]["id"] == "example/model-GGUF"

def test_files_only_returns_single_file_gguf(monkeypatch):
    monkeypatch.setattr(studio, "_read_public_json", lambda *_args, **_kwargs: {"siblings":[
        {"rfilename":"model-Q4_K_M.gguf","size":123},
        {"rfilename":"tokenizer.json"},
        {"rfilename":"subdir/shard.gguf"},
    ]})
    result = studio.list_gguf_files("example/model")
    assert result["count"] == 1
    assert result["files"][0]["filename"] == "model-Q4_K_M.gguf"
    assert result["files"][0]["size_bytes"] == 123

def test_empty_search_is_rejected_without_network(monkeypatch):
    monkeypatch.setattr(studio, "_read_public_json", lambda *_a, **_k: pytest.fail("must not make request"))
    with pytest.raises(ValueError): studio.search_gguf_catalog("  ")

def test_remote_ollama_host_is_rejected(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "https://models.example.net")
    with pytest.raises(studio.ModelStudioError, match="non-loopback"):
        studio._require_loopback_ollama()
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    studio._require_loopback_ollama()

def test_remote_provider_requires_public_https_and_rejects_private_addresses(monkeypatch):
    import socket
    monkeypatch.setattr(studio.socket, "getaddrinfo", lambda *_a, **_k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    assert studio.validate_remote_base_url("https://provider.example/api/v1") == "https://provider.example/api/v1"
    with pytest.raises(ValueError, match="HTTPS"):
        studio.validate_remote_base_url("http://provider.example/v1")
    monkeypatch.setattr(studio.socket, "getaddrinfo", lambda *_a, **_k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))])
    with pytest.raises(ValueError, match="public IP"):
        studio.validate_remote_base_url("https://provider.example/v1")


def test_native_hub_pull_installs_only_supported_safetensors_assets(tmp_path, monkeypatch):
    import json

    config = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 16,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "tie_word_embeddings": True,
    }
    manifest = {
        "siblings": [
            {"rfilename": "config.json", "size": 180},
            {"rfilename": "tokenizer.json", "size": 20},
            {"rfilename": "tokenizer_config.json", "size": 25},
            {"rfilename": "model.safetensors", "size": 64},
            {"rfilename": "remote_model.py", "size": 1000},
            {"rfilename": "../escape.safetensors", "size": 1000},
        ]
    }

    def fake_read(url, timeout=12.0):
        if "/api/models/" in url:
            return manifest
        if url.endswith("/resolve/main/config.json"):
            return config
        raise AssertionError(f"unexpected metadata request: {url}")

    def fake_download(repo_id, filename, destination, **kwargs):
        assert repo_id == "Qwen/Qwen2.5-0.5B-Instruct"
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(config).encode() if filename == "config.json" else b"{}" if filename in {"tokenizer.json", "tokenizer_config.json"} else b"fake-safe-tensor"
        destination.write_bytes(payload)
        callback = kwargs.get("on_progress")
        if callback:
            callback(kwargs.get("progress_offset", 0) + len(payload), kwargs.get("progress_total"))
        return len(payload)

    monkeypatch.setattr(studio, "_read_public_json", fake_read)
    monkeypatch.setattr(studio, "native_model_root", lambda: tmp_path / "native-models")
    monkeypatch.setattr(studio, "_download_native_file", fake_download)
    logs, progress = [], []

    result = studio.pull_native_model(
        "Qwen/Qwen2.5-0.5B-Instruct",
        on_log=logs.append,
        on_progress=lambda current, total: progress.append((current, total)),
    )

    installed = tmp_path / "native-models" / "qwen-qwen2.5-0.5b-instruct"
    assert result["operation"] == "pull_native"
    assert result["model_type"] == "qwen2"
    assert result["architecture"] == "Qwen2ForCausalLM"
    assert result["path"] == str(installed)
    assert (installed / "config.json").is_file()
    assert (installed / "tokenizer.json").is_file()
    assert (installed / "model.safetensors").is_file()
    assert not (installed / "remote_model.py").exists()
    assert not (tmp_path / "native-models" / "escape.safetensors").exists()
    assert logs and progress
    assert result["remote_code_executed"] is False


def test_native_hub_pull_rejects_unsupported_architecture_before_weights(tmp_path, monkeypatch):
    unsupported = {
        "model_type": "gemma3",
        "architectures": ["Gemma3ForConditionalGeneration"],
    }
    monkeypatch.setattr(studio, "_read_public_json", lambda url, timeout=12.0: (
        {"siblings": [{"rfilename": "config.json", "size": 10}, {"rfilename": "tokenizer.json", "size": 10}, {"rfilename": "model.safetensors", "size": 100}]}
        if "/api/models/" in url else unsupported
    ))
    monkeypatch.setattr(studio, "native_model_root", lambda: tmp_path / "native-models")
    monkeypatch.setattr(studio, "_download_native_file", lambda *_a, **_k: pytest.fail("weights must not download for unsupported model"))
    with pytest.raises(studio.ModelStudioError, match="not supported by AXIOM Native Engine"):
        studio.pull_native_model("example/unsupported")
    assert not (tmp_path / "native-models").exists()


def test_unified_local_provider_lists_qwen3_native_models(monkeypatch):
    from axiom.core import llm_client

    monkeypatch.setattr(llm_client, "discover_local_engine_assets", lambda: {
        "models": [
            {"name": "qwen3-local", "path": "/trusted/qwen3", "model_type": "qwen3", "architecture": "Qwen3ForCausalLM"},
            {"name": "llama-local", "path": "/trusted/llama", "model_type": "llama", "architecture": "LlamaForCausalLM"},
            {"name": "unsupported-local", "path": "/trusted/other", "model_type": "gemma", "architecture": "GemmaForCausalLM"},
        ]
    })

    result = llm_client.list_models("axiom-local")
    assert result["execution"] == "local"
    assert result["models"] == ["qwen3-local", "llama-local"]
    assert {item["model_type"] for item in result["model_details"]} == {"qwen3", "llama"}


def test_install_gguf_directly_into_native_model_root(tmp_path, monkeypatch):
    native_root = tmp_path / "trusted-native-models"
    monkeypatch.setattr(studio, "native_model_root", lambda: native_root)

    def fake_download(repo_id, filename, destination, on_progress=None):
        assert repo_id == "example/tiny-gguf"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"gguf-fixture")
        if on_progress:
            on_progress(len(b"gguf-fixture"), len(b"gguf-fixture"))
        return {"repo_id": repo_id, "filename": filename, "path": str(destination), "size_bytes": len(b"gguf-fixture")}

    monkeypatch.setattr(studio, "_download_gguf", fake_download)
    logs = []
    result = studio.install_hf_gguf(
        "example/tiny-gguf",
        "tiny-Q4_K_M.gguf",
        "AXIOM Tiny",
        register_ollama=False,
        on_log=logs.append,
        install_native=True,
    )

    installed = native_root / "axiom-tiny" / "tiny-Q4_K_M.gguf"
    assert installed.is_file()
    assert installed.read_bytes() == b"gguf-fixture"
    assert result["path"] == str(installed)
    assert result["installed_for_native_engine"] is True
    assert result["registered_with_ollama"] is False
    assert result["checkpoint_format"] == "gguf"
    assert any("trusted native model root" in message for message in logs)


def test_install_gguf_native_target_rejects_conflicting_ollama_or_existing_checkpoint(tmp_path, monkeypatch):
    native_root = tmp_path / "trusted-native-models"
    monkeypatch.setattr(studio, "native_model_root", lambda: native_root)
    with pytest.raises(ValueError, match="not both"):
        studio.install_hf_gguf(
            "example/tiny-gguf", "tiny.gguf", "tiny",
            register_ollama=True, install_native=True,
        )

    target_dir = native_root / "tiny"
    target_dir.mkdir(parents=True)
    (target_dir / "previous.gguf").write_bytes(b"existing")
    monkeypatch.setattr(studio, "_download_gguf", lambda *_args, **_kwargs: pytest.fail("must not overwrite/ambiguate an existing GGUF model"))
    with pytest.raises(studio.ModelStudioError, match="already contains a GGUF file"):
        studio.install_hf_gguf(
            "example/tiny-gguf", "tiny.gguf", "tiny",
            register_ollama=False, install_native=True,
        )



def test_cli_can_install_gguf_directly_for_native_engine(monkeypatch):
    from typer.testing import CliRunner
    from axiom.cli.model_studio import studio_app

    observed = {}

    def fake_install(repo_id, filename, model_name, register_ollama, **kwargs):
        observed.update(
            repo_id=repo_id,
            filename=filename,
            model_name=model_name,
            register_ollama=register_ollama,
            install_native=kwargs.get("install_native"),
        )
        return {
            "path": "/trusted/models/tiny/model-Q4_K_M.gguf",
            "model_name": model_name,
            "registered_with_ollama": False,
            "installed_for_native_engine": True,
            "checkpoint_format": "gguf",
        }

    monkeypatch.setattr(studio, "install_hf_gguf", fake_install)
    result = CliRunner().invoke(studio_app, [
        "install-gguf", "example/tiny-gguf", "model-Q4_K_M.gguf",
        "--name", "tiny", "--native",
    ])

    assert result.exit_code == 0, f"{result.output}\\n{result.exception!r}"
    assert observed == {
        "repo_id": "example/tiny-gguf",
        "filename": "model-Q4_K_M.gguf",
        "model_name": "tiny",
        "register_ollama": False,
        "install_native": True,
    }
    assert "Installed for AXIOM Native Engine" in result.output
    assert "axiom engine run --model-path" in result.output
