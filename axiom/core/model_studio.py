"""Local model acquisition and GGUF conversion helpers for AXIOM.

External network access is limited to explicit public Hugging Face Hub search/file
requests. Model inference remains local through Ollama. GGUF conversion only reads
an existing local Transformers model directory and a local llama.cpp checkout.
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener, urlopen

_MAX_RESULTS = 20
_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024 * 1024
_MAX_NATIVE_MODEL_BYTES = 160 * 1024**3
_MAX_NATIVE_MODEL_FILE_BYTES = 20 * 1024**3
_NATIVE_MODEL_ROOT = Path.home() / ".local" / "share" / "axiom" / "models"
_NATIVE_OPTIONAL_FILES = {"tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "added_tokens.json"}
_NATIVE_SHARD_RE = re.compile(r"^model-[0-9]{5}-of-[0-9]{5}\.safetensors$")
_MAX_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,119}$")
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+() -]{0,240}\.gguf$", re.IGNORECASE)
_SLUG = re.compile(r"[^a-z0-9._-]+")
_ACTIVE_LOCK = threading.RLock()

class ModelStudioError(RuntimeError):
    """Model acquisition/conversion could not be completed."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def studio_home() -> Path:
    configured = os.environ.get("AXIOM_MODEL_STUDIO_HOME", "").strip()
    return (Path(configured).expanduser() if configured else Path.home() / ".local" / "share" / "axiom" / "model-studio").resolve()


def jobs_directory() -> Path:
    return studio_home() / "jobs"


def validate_ollama_name(name: str) -> str:
    value = name.strip() if isinstance(name, str) else ""
    if not value or not _MAX_MODEL_NAME.fullmatch(value) or "://" in value or ".." in value or value.startswith("/"):
        raise ValueError("Model name must be a safe Ollama model identifier, not a URL or filesystem path.")
    return value


def validate_repo_id(repo_id: str) -> str:
    value = repo_id.strip() if isinstance(repo_id, str) else ""
    if not _REPO_ID.fullmatch(value) or ".." in value:
        raise ValueError("Repository ID must look like owner/repository.")
    return value


def validate_remote_base_url(base_url: str) -> str:
    """Accept only explicit public HTTPS OpenAI-compatible API base URLs."""
    value = base_url.strip() if isinstance(base_url, str) else ""
    if not value or len(value) > 512 or any(ord(char) < 32 for char in value):
        raise ValueError("Enter a valid public HTTPS API base URL ending in /v1.")
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port or 443
    except ValueError:
        raise ValueError("API base URL is malformed.") from None
    if parsed.scheme != "https" or not host or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Online model connections require HTTPS without embedded credentials, query strings, or fragments.")
    if not parsed.path.rstrip("/").endswith("/v1"):
        raise ValueError("Enter the provider's OpenAI-compatible API base URL ending in /v1 (for example https://provider.example/api/v1).")
    if host == "localhost" or host.endswith(".local"):
        raise ValueError("Remote model connections must use a public host; local runtimes should use the local Ollama connection.")
    try:
        addresses = {ipaddress.ip_address(info[4][0].split("%", 1)[0]) for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)}
    except (OSError, ValueError, IndexError):
        raise ModelStudioError("The configured public model endpoint hostname could not be resolved.") from None
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Online model connections must resolve exclusively to public IP addresses; private/loopback targets are blocked.")
    authority = f"[{host}]" if ":" in host else host
    if parsed.port and parsed.port != 443:
        authority += f":{parsed.port}"
    return f"https://{authority}{parsed.path.rstrip('/')}"


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, new_url):
        return None


def _remote_request(base_url: str, api_key: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    base = validate_remote_base_url(base_url)
    key = api_key.strip() if isinstance(api_key, str) else ""
    if not key or len(key) > 512 or any(ord(char) < 32 for char in key):
        raise ValueError("API key is required and must be a single-line value no longer than 512 characters.")
    if endpoint not in {"/models", "/chat/completions"}:
        raise ValueError("Unsupported remote-model API operation.")
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(base + endpoint, data=body, method="GET" if body is None else "POST", headers={
        "Accept": "application/json", "Authorization": f"Bearer {key}",
        **({"Content-Type": "application/json"} if body is not None else {}),
    })
    # Do not use environment-configured proxies or follow redirects: the API key
    # goes only to the HTTPS host the user explicitly validated.
    opener = build_opener(ProxyHandler({}), _RejectRedirects())
    try:
        with opener.open(req, timeout=30 if body is None else 180) as response:
            raw = response.read(2_000_000)
    except HTTPError as exc:
        if exc.code in {301, 302, 303, 307, 308}:
            raise ModelStudioError("The online provider redirected the request. Redirects are blocked to protect the API key; enter its final HTTPS /v1 endpoint.") from None
        if exc.code in {401, 403}:
            raise ModelStudioError("The online provider rejected the API key or denied access (HTTP 401/403).") from None
        raise ModelStudioError(f"Online provider request failed (HTTP {exc.code}). Check the endpoint, model access, and provider status.") from None
    except (URLError, TimeoutError, OSError):
        raise ModelStudioError("Could not connect to the configured public HTTPS model endpoint.") from None
    try:
        result = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ModelStudioError("The online provider returned an invalid JSON response.") from None
    if not isinstance(result, dict):
        raise ModelStudioError("The online provider returned an unexpected response shape.")
    return result


def list_remote_models(base_url: str, api_key: str) -> dict[str, Any]:
    """Check an explicitly configured OpenAI-compatible endpoint without saving credentials."""
    base = validate_remote_base_url(base_url)
    data = _remote_request(base, api_key, "/models")
    raw = data.get("data", data.get("models", []))
    models = []
    if isinstance(raw, list):
        for item in raw[:100]:
            name = item.get("id") if isinstance(item, dict) else item if isinstance(item, str) else None
            if isinstance(name, str) and name.strip() and len(name) <= 256 and not any(ord(char) < 32 for char in name):
                models.append(name)
    return {"provider": "openai-compatible", "base_url": base, "models": models, "count": len(models),
            "credential_storage": "not_saved", "note": "Credentials are used only for this request and are not written to the AXIOM workspace."}


def run_remote_chat(base_url: str, api_key: str, model: str, prompt: str,
                    max_tokens: int = 256, temperature: float = 0.7) -> dict[str, Any]:
    """Send one explicitly requested prompt to a public remote provider."""
    if not isinstance(model, str) or not model.strip() or len(model) > 256 or "://" in model or any(ord(char) < 32 for char in model):
        raise ValueError("Enter a model ID returned by the selected provider.")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12_000:
        raise ValueError("Prompt must contain 1–12,000 characters.")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 2_048:
        raise ValueError("Maximum completion tokens must be between 1 and 2,048.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= float(temperature) <= 2:
        raise ValueError("Temperature must be between 0 and 2.")
    base = validate_remote_base_url(base_url)
    data = _remote_request(base, api_key, "/chat/completions", {
        "model": model.strip(), "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": float(temperature), "stream": False,
    })
    choices = data.get("choices")
    message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str))
    if not isinstance(content, str):
        raise ModelStudioError("The remote provider response did not contain text content.")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return {"provider": "openai-compatible", "base_url": base, "model": model.strip(), "response": content[:100_000],
            "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
            "completion_tokens": usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None,
            "credential_storage": "not_saved", "generated_at": _now()}


def validate_gguf_filename(filename: str) -> str:
    value = filename.strip() if isinstance(filename, str) else ""
    if not _FILE_NAME.fullmatch(value) or "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError("Choose a single .gguf file from the selected repository; folder paths are not allowed.")
    return value


def _read_public_json(url: str, timeout: float = 12.0) -> Any:
    # Callers construct URLs only from fixed Hugging Face origins and validated IDs.
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "AXIOM-Model-Studio/1"})
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read(4_000_000)
    except HTTPError as exc:
        if exc.code == 401 or exc.code == 403:
            raise ModelStudioError("This public catalogue item requires access that is not available anonymously.") from None
        if exc.code == 404:
            raise ModelStudioError("The selected model or file was not found in the public catalogue.") from None
        raise ModelStudioError(f"Hugging Face catalogue request failed (HTTP {exc.code}).") from None
    except (URLError, TimeoutError, OSError):
        raise ModelStudioError("Could not reach the public Hugging Face catalogue. Check your connection and retry.") from None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ModelStudioError("The model catalogue returned an invalid response.") from None


def search_gguf_catalog(query: str, limit: int = 12) -> dict[str, Any]:
    """Search public Hub model metadata and only return GGUF-related repositories.

    The Hub search endpoint primarily matches repository IDs. Strip generic
    descriptor tokens (GGUF, instruct, quantization) before the query, then fall
    back to popular GGUF-tagged repositories if that narrow search finds nothing.
    """
    q = query.strip() if isinstance(query, str) else ""
    if not q or len(q) > 120 or any(ord(char) < 32 for char in q):
        raise ValueError("Enter a model search query between 1 and 120 characters.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_RESULTS:
        raise ValueError(f"Result limit must be between 1 and {_MAX_RESULTS}.")
    generic = {"gguf", "instruct", "instruction", "model", "models", "quant", "quantization", "small", "low", "memory", "q4", "q4_k_m", "q5", "q5_k_m", "q8", "up", "to"}
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9._+-]*", q)
    search_term = next((term for term in terms if term.lower() not in generic), "")

    def fetch(term: str) -> list[dict[str, Any]]:
        fields: dict[str, str] = {"filter": "gguf", "sort": "downloads", "direction": "-1", "limit": str(min(limit * 4, 80))}
        if term:
            fields["search"] = term
        data = _read_public_json(f"https://huggingface.co/api/models?{urlencode(fields)}")
        return data if isinstance(data, list) else []

    data = fetch(search_term)
    def select(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected = []
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            tags = item.get("tags", [])
            tags = [str(tag)[:80] for tag in tags[:80]] if isinstance(tags, list) else []
            model_id = item["id"][:192]
            if "gguf" not in [tag.lower() for tag in tags] and "gguf" not in model_id.lower():
                continue
            downloads = item.get("downloads", 0)
            likes = item.get("likes", 0)
            selected.append({
                "id": model_id,
                "author": model_id.split("/", 1)[0],
                "downloads": downloads if isinstance(downloads, int) and downloads >= 0 else 0,
                "likes": likes if isinstance(likes, int) and likes >= 0 else 0,
                "pipeline_tag": str(item.get("pipeline_tag", "unknown"))[:80],
                "last_modified": str(item.get("lastModified", ""))[:48],
                "tags": tags,
                "url": f"https://huggingface.co/{quote(model_id, safe='/')}",
            })
        return selected
    results = select(data)
    if not results and search_term:
        # Still show useful GGUF candidates for semantic terms not present in IDs.
        results = select(fetch(""))
    results.sort(key=lambda item: item["downloads"], reverse=True)
    results = results[:limit]
    return {"provider": "huggingface", "query": q, "count": len(results), "results": results,
            "note": "Public catalogue metadata only. Check the model card, license, quantization, and required disk/RAM before installing."}

def list_gguf_files(repo_id: str) -> dict[str, Any]:
    repo = validate_repo_id(repo_id)
    data = _read_public_json(f"https://huggingface.co/api/models/{quote(repo, safe='/')}?blobs=true")
    siblings = data.get("siblings", []) if isinstance(data, dict) else []
    if not isinstance(siblings, list):
        raise ModelStudioError("The selected model repository has no readable file inventory.")
    files = []
    for item in siblings:
        if not isinstance(item, dict) or not isinstance(item.get("rfilename"), str):
            continue
        filename = item["rfilename"]
        # This GUI deliberately supports a single GGUF blob at a time; split shards
        # are still listed but must be installed as a compatible complete set later.
        if filename.lower().endswith(".gguf") and "/" not in filename and "\\" not in filename:
            size = item.get("size")
            files.append({"filename": filename, "size_bytes": size if isinstance(size, int) and size >= 0 else None,
                          "quantization": _quantization_hint(filename)})
    files.sort(key=lambda item: (0 if item["quantization"] == "Q4_K_M" else 1, item["filename"].lower()))
    return {"repo_id": repo, "files": files[:300], "count": len(files),
            "url": f"https://huggingface.co/{quote(repo, safe='/')}",
            "note": "Only single-file GGUF entries are listed here. Large or split models may need additional files."}


def _quantization_hint(filename: str) -> str:
    match = re.search(r"(?i)(Q[2-8](?:_K_M|_K_S|_K_L|_K|_0)?|IQ[0-9]_[A-Z0-9_]+|F16|F32|BF16)", filename)
    return match.group(1).upper() if match else "unknown"


def _slug(value: str) -> str:
    safe = _SLUG.sub("-", value.lower()).strip("-._")
    return (safe or "local-model")[:80]


def _require_loopback_ollama() -> None:
    configured = os.environ.get("OLLAMA_HOST", "").strip()
    if not configured:
        return
    candidate = configured if "://" in configured else "http://" + configured
    try:
        hostname = (urlsplit(candidate).hostname or "").lower().strip("[]")
    except ValueError:
        hostname = ""
    if hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ModelStudioError("OLLAMA_HOST is configured to a non-loopback host. Model Studio is local-only; unset OLLAMA_HOST or point it to localhost first.")


def ollama_status() -> dict[str, Any]:
    executable = shutil.which("ollama")
    return {"ollama_installed": executable is not None,
            "ollama_endpoint": "http://127.0.0.1:11434",
            "gguf_library": str(studio_home() / "gguf"),
            "model_download_limit_bytes": _MAX_DOWNLOAD_BYTES,
            "conversion_available": False}


def _run_checked(command: list[str], *, label: str, timeout: int = 14_400) -> str:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout, check=False)
    except FileNotFoundError:
        raise ModelStudioError(f"Required local tool for {label} was not found. Check the selected llama.cpp/Ollama installation.") from None
    except subprocess.TimeoutExpired:
        raise ModelStudioError(f"{label} exceeded the {timeout // 60}-minute timeout.") from None
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        detail = " ".join(output.split())[-1200:]
        raise ModelStudioError(f"{label} failed (exit {proc.returncode}). {detail}".strip())
    return " ".join(output.split())[-1200:]


def pull_ollama_model(model_name: str, on_log: Callable[[str], None] | None = None) -> dict[str, Any]:
    model = validate_ollama_name(model_name)
    _require_loopback_ollama()
    executable = shutil.which("ollama")
    if not executable:
        raise ModelStudioError("Ollama is not installed. Install Ollama first or download a GGUF file without registering it.")
    if on_log:
        on_log(f"Starting local Ollama pull: {model}")
    summary = _run_checked([executable, "pull", model], label="Ollama model pull", timeout=7200)
    if on_log:
        on_log("Ollama pull finished.")
    return {"operation": "pull", "model_name": model, "runtime": "ollama", "summary": summary}


def _download_gguf(repo_id: str, filename: str, destination: Path,
                    on_progress: Callable[[int, int | None], None] | None = None) -> dict[str, Any]:
    repo = validate_repo_id(repo_id)
    file_name = validate_gguf_filename(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ModelStudioError(f"Refusing to overwrite an existing model file: {destination.name}")
    try:
        free = shutil.disk_usage(destination.parent).free
    except OSError:
        free = 0
    url = f"https://huggingface.co/{quote(repo, safe='/')}/resolve/main/{quote(file_name, safe='') }?download=true"
    request = Request(url, headers={"User-Agent": "AXIOM-Model-Studio/1", "Accept": "application/octet-stream"})
    temp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    written = 0
    try:
        with urlopen(request, timeout=60) as response:
            raw_length = response.headers.get("Content-Length")
            total = int(raw_length) if raw_length and raw_length.isdigit() else None
            if total is not None and total > _MAX_DOWNLOAD_BYTES:
                raise ModelStudioError("Selected GGUF exceeds AXIOM's 20 GiB per-file download limit.")
            if total is not None and free and total > free:
                raise ModelStudioError("There is not enough free disk space for this GGUF file.")
            with temp.open("xb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > _MAX_DOWNLOAD_BYTES:
                        raise ModelStudioError("GGUF download exceeded AXIOM's 20 GiB per-file limit.")
                    if free and written > free:
                        raise ModelStudioError("The download reached the available disk-space limit.")
                    handle.write(chunk)
                    if on_progress:
                        on_progress(written, total)
        if written <= 0:
            raise ModelStudioError("The catalogue returned an empty GGUF file.")
        os.replace(temp, destination)
    except ModelStudioError:
        temp.unlink(missing_ok=True)
        raise
    except HTTPError as exc:
        temp.unlink(missing_ok=True)
        if exc.code == 404:
            raise ModelStudioError("The GGUF file was not found. Refresh the model's file list.") from None
        raise ModelStudioError(f"GGUF download failed (HTTP {exc.code}). Check the model's access/license requirements.") from None
    except (URLError, TimeoutError, OSError) as exc:
        temp.unlink(missing_ok=True)
        raise ModelStudioError("GGUF download was interrupted. A partial file was removed; retry when the connection is stable.") from None
    return {"repo_id": repo, "filename": file_name, "path": str(destination), "size_bytes": written}


def install_hf_gguf(repo_id: str, filename: str, model_name: str, register_ollama: bool = True,
                    on_log: Callable[[str], None] | None = None,
                    on_progress: Callable[[int, int | None], None] | None = None,
                    install_native: bool = False) -> dict[str, Any]:
    """Download a public GGUF either to Model Studio storage or AXIOM's trusted native root."""
    repo = validate_repo_id(repo_id)
    file_name = validate_gguf_filename(filename)
    if not isinstance(register_ollama, bool) or not isinstance(install_native, bool):
        raise ValueError("GGUF installation targets must be boolean flags.")
    if install_native and register_ollama:
        raise ValueError("Choose direct AXIOM Native Engine installation or Ollama registration, not both.")
    model = validate_ollama_name(model_name) if register_ollama else _slug(model_name)
    if install_native:
        native_root = native_model_root().resolve()
        base = native_root / _slug(model)
        if base.is_symlink():
            raise ModelStudioError("The native model destination must not be a symlink.")
        base.mkdir(parents=True, exist_ok=True)
        try:
            resolved_base = base.resolve(strict=True)
            resolved_base.relative_to(native_root)
        except (OSError, ValueError):
            raise ModelStudioError("The native model destination escaped AXIOM's trusted model root.") from None
        if any(item.is_file() and item.suffix.lower() == ".gguf" for item in resolved_base.iterdir()):
            raise ModelStudioError("This native model directory already contains a GGUF file; refusing to create an ambiguous checkpoint.")
    else:
        base = studio_home() / "gguf" / _slug(model)
        base.mkdir(parents=True, exist_ok=True)
    target = base / file_name
    downloaded = _download_gguf(repo, file_name, target, on_progress)
    result: dict[str, Any] = {
        "operation": "install_gguf",
        **downloaded,
        "model_name": model,
        "registered_with_ollama": False,
        "installed_for_native_engine": install_native,
        "checkpoint_format": "gguf",
    }
    if install_native:
        if on_log:
            on_log(
                f"Downloaded GGUF into AXIOM's trusted native model root at {target}. "
                "AXIOM will validate its architecture/tokenizer and dequantize tensors to floating-point when loaded."
            )
    if register_ollama:
        _require_loopback_ollama()
        executable = shutil.which("ollama")
        if not executable:
            raise ModelStudioError(f"GGUF downloaded to {target}, but Ollama is not installed to register it. Install Ollama, then run `ollama create {model} -f {base / 'Modelfile'}` after creating a Modelfile with this path.")
        modelfile = base / "Modelfile"
        text = f"FROM {target}\nPARAMETER num_ctx 4096\n"
        modelfile.write_text(text, encoding="utf-8")
        if on_log:
            on_log(f"Downloaded GGUF ({downloaded['size_bytes']:,} bytes); registering {model} with local Ollama.")
        summary = _run_checked([executable, "create", model, "-f", str(modelfile)], label="Ollama GGUF registration", timeout=7200)
        result.update(registered_with_ollama=True, modelfile=str(modelfile), registration_summary=summary)
        if on_log:
            on_log("GGUF registered with Ollama. Inference is now available through the local runtime.")
    return result



def native_model_root() -> Path:
    """Return AXIOM's trusted root for downloaded native model checkpoints."""
    return _NATIVE_MODEL_ROOT


def _is_native_asset_name(filename: str) -> bool:
    if not isinstance(filename, str) or Path(filename).name != filename or filename in {"", ".", ".."}:
        return False
    return filename in {
        "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "generation_config.json", "added_tokens.json", "model.safetensors",
        "model.safetensors.index.json",
    } or bool(_NATIVE_SHARD_RE.fullmatch(filename))


def _download_native_file(
    repo_id: str,
    filename: str,
    destination: Path,
    *,
    on_progress: Callable[[int, int | None], None] | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
    free_bytes: int | None = None,
) -> int:
    """Download one whitelisted public Hub file into staging."""
    if not _is_native_asset_name(filename):
        raise ModelStudioError("The Hub manifest contains a native-model filename AXIOM does not allow.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{quote(repo_id, safe='/')}/resolve/main/{quote(filename, safe='')}?download=true"
    request = Request(url, headers={"User-Agent": "AXIOM-Native-Model-Studio/1", "Accept": "application/octet-stream"})
    temp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    written = 0
    try:
        with urlopen(request, timeout=60) as response:
            raw_length = response.headers.get("Content-Length")
            total = int(raw_length) if raw_length and raw_length.isdigit() else None
            if total is not None and total > _MAX_NATIVE_MODEL_FILE_BYTES:
                raise ModelStudioError("A native-model file exceeds AXIOM's 20 GiB per-file limit.")
            if free_bytes is not None and total is not None and total + progress_offset > free_bytes:
                raise ModelStudioError("There is not enough free disk space for this native model.")
            with temp.open("xb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > _MAX_NATIVE_MODEL_FILE_BYTES:
                        raise ModelStudioError("A native-model file exceeded AXIOM's 20 GiB per-file limit.")
                    if progress_offset + written > _MAX_NATIVE_MODEL_BYTES:
                        raise ModelStudioError("Native-model download exceeded AXIOM's 160 GiB total limit.")
                    if free_bytes is not None and progress_offset + written > free_bytes:
                        raise ModelStudioError("The native-model download reached the available disk-space limit.")
                    handle.write(chunk)
                    if on_progress:
                        on_progress(progress_offset + written, progress_total)
        if written <= 0:
            raise ModelStudioError(f"The Hub returned an empty native-model file: {filename}.")
        os.replace(temp, destination)
        return written
    except ModelStudioError:
        temp.unlink(missing_ok=True)
        raise
    except HTTPError as exc:
        temp.unlink(missing_ok=True)
        if exc.code in {401, 403}:
            raise ModelStudioError("This repository requires authentication. AXIOM native pull currently supports public Hub repositories only.") from None
        if exc.code == 404:
            raise ModelStudioError(f"Required native-model file {filename!r} was not found in the public repository.") from None
        raise ModelStudioError(f"Native-model download failed for {filename!r} (HTTP {exc.code}).") from None
    except (URLError, TimeoutError, OSError):
        temp.unlink(missing_ok=True)
        raise ModelStudioError(f"Native-model download was interrupted while retrieving {filename!r}; partial files were removed.") from None


def pull_native_model(
    repo_id: str,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int | None], None] | None = None,
) -> dict[str, Any]:
    """Acquire public Llama/Mistral/Qwen2 safetensors assets for AXIOM's native runtime.

    Only allowlisted data/config files are downloaded; model-provided Python is never fetched or executed.
    Existing installs are left untouched.
    """
    repo = validate_repo_id(repo_id)
    repo_url = quote(repo, safe="/")
    metadata = _read_public_json(f"https://huggingface.co/api/models/{repo_url}?blobs=true")
    siblings = metadata.get("siblings", []) if isinstance(metadata, dict) else []
    if not isinstance(siblings, list):
        raise ModelStudioError("The Hub did not return a usable file inventory for this model.")
    sizes: dict[str, int | None] = {}
    for item in siblings:
        if not isinstance(item, dict) or not isinstance(item.get("rfilename"), str):
            continue
        name = item["rfilename"]
        if not _is_native_asset_name(name):
            continue
        raw_size = item.get("size")
        if not isinstance(raw_size, int) or isinstance(raw_size, bool) or raw_size < 0:
            lfs = item.get("lfs")
            raw_size = lfs.get("size") if isinstance(lfs, dict) else None
        sizes[name] = raw_size if isinstance(raw_size, int) and not isinstance(raw_size, bool) and raw_size >= 0 else None

    if "config.json" not in sizes or "tokenizer.json" not in sizes:
        raise ModelStudioError("Native pull requires top-level config.json and tokenizer.json files.")
    if on_log:
        on_log(f"Inspecting public Hub metadata for {repo} before downloading weights.")
    config = _read_public_json(f"https://huggingface.co/{repo_url}/resolve/main/config.json")
    if not isinstance(config, dict):
        raise ModelStudioError("The repository config.json is not a JSON object.")
    try:
        from axiom.core.native_engine import NativeLlamaConfig
        native_config = NativeLlamaConfig.from_mapping(config)
    except (ImportError, OSError):
        raise ModelStudioError("Install AXIOM's native-inference extra before pulling native checkpoints.") from None
    except (ValueError, TypeError, KeyError) as exc:
        raise ModelStudioError(f"This checkpoint is not supported by AXIOM Native Engine: {exc}") from None

    selected = ["config.json", "tokenizer.json"]
    selected.extend(name for name in sorted(_NATIVE_OPTIONAL_FILES) if name in sizes)
    if "model.safetensors.index.json" in sizes:
        index = _read_public_json(f"https://huggingface.co/{repo_url}/resolve/main/model.safetensors.index.json")
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ModelStudioError("The native safetensors index has no valid weight_map.")
        shards: set[str] = set()
        for tensor_name, filename in weight_map.items():
            if (
                not isinstance(tensor_name, str) or not tensor_name
                or not isinstance(filename, str)
                or not _is_native_asset_name(filename)
                or not filename.endswith(".safetensors")
            ):
                raise ModelStudioError("The native safetensors index contains an unsafe or unsupported shard filename.")
            shards.add(filename)
        if not shards or len(shards) > 512:
            raise ModelStudioError("The native checkpoint must contain between 1 and 512 safetensors shards.")
        missing = sorted(shard for shard in shards if shard not in sizes)
        if missing:
            raise ModelStudioError("The native safetensors index references a shard not listed by the Hub: " + ", ".join(missing[:4]))
        selected.append("model.safetensors.index.json")
        selected.extend(sorted(shards))
    elif "model.safetensors" in sizes:
        selected.append("model.safetensors")
    else:
        raise ModelStudioError("This repository has no supported model.safetensors file or shard index.")

    selected = list(dict.fromkeys(selected))
    for filename in selected:
        if not _is_native_asset_name(filename) or filename not in sizes:
            raise ModelStudioError(f"Required native-model file {filename!r} is missing from the public Hub inventory.")
        size = sizes[filename]
        if size is not None and size > _MAX_NATIVE_MODEL_FILE_BYTES:
            raise ModelStudioError(f"Native-model file {filename!r} exceeds the 20 GiB per-file limit.")
    known_sizes = [sizes[name] for name in selected]
    total_size = sum(size for size in known_sizes if size is not None)
    expected_total = total_size if all(size is not None for size in known_sizes) else None
    if total_size > _MAX_NATIVE_MODEL_BYTES:
        raise ModelStudioError("This native checkpoint exceeds AXIOM's 160 GiB total model limit.")

    root = native_model_root().expanduser().resolve(strict=False)
    model_name = _slug(repo.replace("/", "-"))
    target = root / model_name
    if target.exists() or target.is_symlink():
        raise ModelStudioError(f"AXIOM refuses to overwrite the existing native model directory: {target}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        free_bytes = shutil.disk_usage(root).free
    except OSError:
        free_bytes = None
    if expected_total is not None and free_bytes is not None and expected_total > free_bytes:
        raise ModelStudioError("There is not enough free disk space for the complete native checkpoint.")
    staging = root / f".{model_name}.{uuid.uuid4().hex}.staging"
    staging.mkdir(mode=0o700)
    downloaded = 0
    try:
        if on_log:
            summary = f"{expected_total:,} bytes estimated" if expected_total is not None else "total size unknown"
            on_log(f"Downloading {len(selected)} allowlisted model/tokenizer files ({summary}).")
        for filename in selected:
            written = _download_native_file(
                repo, filename, staging / filename,
                on_progress=on_progress,
                progress_offset=downloaded,
                progress_total=expected_total,
                free_bytes=free_bytes,
            )
            downloaded += written
            if downloaded > _MAX_NATIVE_MODEL_BYTES:
                raise ModelStudioError("Native-model download exceeded AXIOM's 160 GiB total limit.")
        local_config = json.loads((staging / "config.json").read_text(encoding="utf-8"))
        from axiom.core.native_engine import NativeLlamaConfig
        verified = NativeLlamaConfig.from_mapping(local_config)
        if verified.model_type != native_config.model_type:
            raise ModelStudioError("The downloaded model config changed during the pull; installation was cancelled.")
        if not (staging / "tokenizer.json").is_file():
            raise ModelStudioError("The completed download is missing tokenizer.json.")
        if not (staging / "model.safetensors").is_file() and not (staging / "model.safetensors.index.json").is_file():
            raise ModelStudioError("The completed download is missing its native safetensors weights.")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if on_log:
        on_log(f"Installed native checkpoint at {target}. Select it on the Runtime page to run AXIOM inference.")
    architectures = config.get("architectures")
    architecture_name = architectures[0] if isinstance(architectures, list) and architectures and isinstance(architectures[0], str) else native_config.model_type
    return {
        "operation": "pull_native",
        "repo_id": repo,
        "model_name": model_name,
        "model_type": native_config.model_type,
        "architecture": architecture_name,
        "path": str(target),
        "files": len(selected),
        "file_names": selected,
        "size_bytes": downloaded,
        "expected_size_bytes": expected_total,
        "runtime": "axiom-native",
        "remote_code_executed": False,
    }



def convert_local_model_to_gguf(source_dir: str, llama_cpp_dir: str, output_path: str,
                                 outtype: str = "f16", quantization: str = "Q4_K_M",
                                 on_log: Callable[[str], None] | None = None) -> dict[str, Any]:
    try:
        source = Path(source_dir).expanduser().resolve(strict=True)
        llama_dir = Path(llama_cpp_dir).expanduser().resolve(strict=True)
        output = Path(output_path).expanduser().resolve(strict=False)
        output_parent = output.parent.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("Source model, llama.cpp directory, and output parent must be existing local paths.") from None
    if not source.is_dir() or not (source / "config.json").is_file():
        raise ValueError("Source must be a local Transformers model directory containing config.json.")
    converter = llama_dir / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise ValueError("The chosen llama.cpp directory must contain convert_hf_to_gguf.py.")
    if output_parent != output.parent or not output_parent.is_dir() or output.suffix.lower() != ".gguf":
        raise ValueError("Output must be a .gguf file inside an existing local directory.")
    if output.exists():
        raise ValueError("Refusing to overwrite an existing output file.")
    if output == source or source in output.parents or output.parent == source:
        raise ValueError("GGUF output must not overlap the source model directory.")
    allowed_outtypes = {"auto", "f16", "f32", "bf16"}
    if outtype not in allowed_outtypes:
        raise ValueError("Output type must be auto, f16, f32, or bf16.")
    allowed_quant = {"none", "Q2_K", "Q3_K_M", "Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"}
    if quantization not in allowed_quant:
        raise ValueError("Unsupported quantization. Choose none, Q2_K, Q3_K_M, Q4_0, Q4_K_M, Q5_K_M, Q6_K, or Q8_0.")
    if on_log:
        on_log("Converting local Transformers weights to GGUF. This can use substantial RAM and disk space.")
    if quantization == "none":
        summary = _run_checked([sys.executable, str(converter), str(source), "--outfile", str(output), "--outtype", outtype], label="HF-to-GGUF conversion")
        return {"operation": "convert_gguf", "source_dir": str(source), "output_path": str(output), "outtype": outtype, "quantization": "none", "summary": summary}
    quant_candidates = [llama_dir / "build" / "bin" / "llama-quantize", llama_dir / "build" / "bin" / "llama-quantize.exe", llama_dir / "llama-quantize", llama_dir / "llama-quantize.exe"]
    quantizer = next((path for path in quant_candidates if path.is_file()), None)
    if quantizer is None:
        raise ValueError("Quantization requested, but llama-quantize was not found. Build llama.cpp first, or choose quantization 'none'.")
    interim = output.with_name(f".{output.stem}.unquantized.{uuid.uuid4().hex}.gguf")
    try:
        _run_checked([sys.executable, str(converter), str(source), "--outfile", str(interim), "--outtype", "f16"], label="HF-to-GGUF conversion")
        if on_log:
            on_log(f"Converted weights; quantizing to {quantization}.")
        summary = _run_checked([str(quantizer), str(interim), str(output), quantization], label="GGUF quantization")
    except Exception:
        interim.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise
    interim.unlink(missing_ok=True)
    return {"operation": "convert_gguf", "source_dir": str(source), "output_path": str(output), "outtype": "f16", "quantization": quantization, "summary": summary}


def _job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("Invalid Model Studio job ID.")
    return jobs_directory() / f"{job_id}.json"


def _write_job(job: dict[str, Any]) -> None:
    folder = jobs_directory()
    folder.mkdir(parents=True, exist_ok=True)
    target = _job_path(job["id"])
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, target)


def _read_job(job_id: str) -> dict[str, Any] | None:
    path = _job_path(job_id)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ModelStudioError("Saved Model Studio job state could not be read.") from None
    return value if isinstance(value, dict) else None


def _pid_alive(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _recover(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("status") in {"queued", "running"} and job.get("runner_pid") != os.getpid() and not _pid_alive(job.get("runner_pid")):
        job = dict(job)
        job.update(status="failed", error="The Model Studio process exited before the operation completed.", updated_at=_now(), finished_at=_now())
        job["logs"] = [*job.get("logs", []), "Recovered stale job after process exit."][-80:]
        _write_job(job)
    return job


def get_job(job_id: str) -> dict[str, Any] | None:
    value = _read_job(job_id)
    return _recover(value) if value else None


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("Job limit must be between 1 and 100.")
    folder = jobs_directory()
    if not folder.is_dir():
        return []
    records = []
    for path in folder.glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and re.fullmatch(r"[0-9a-f]{32}", str(value.get("id", ""))):
                records.append(_recover(value))
        except (OSError, json.JSONDecodeError):
            continue
    records.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return records[:limit]


def _update(job_id: str, **changes: Any) -> None:
    with _ACTIVE_LOCK:
        job = _read_job(job_id)
        if not job:
            return
        logs = changes.pop("append_log", None)
        if logs:
            job["logs"] = [*job.get("logs", []), str(logs)[:600]][-80:]
        job.update(changes, updated_at=_now())
        _write_job(job)


def _worker(job_id: str) -> None:
    job = _read_job(job_id)
    if not job:
        return
    _update(job_id, status="running", started_at=_now(), append_log="Job started.")
    config = job["config"]
    try:
        def log(message: str) -> None:
            _update(job_id, append_log=message)
        last_percent = [-1]
        last_unknown_bytes = [0]
        def progress(current: int, total: int | None) -> None:
            if total:
                percent = min(100, int(current * 100 / total))
                if percent // 5 == last_percent[0] // 5 and percent < 100:
                    return
                last_percent[0] = percent
            elif current - last_unknown_bytes[0] < 32 * 1024 * 1024 and current > 1024 * 1024:
                return
            last_unknown_bytes[0] = current
            _update(job_id, progress_bytes=current, total_bytes=total, append_log=(f"Downloaded {current:,} bytes" + (f" / {total:,} bytes" if total else "")))
        operation = config["operation"]
        if operation == "pull":
            result = pull_ollama_model(config["model_name"], on_log=log)
        elif operation == "pull_native":
            result = pull_native_model(config["repo_id"], on_log=log, on_progress=progress)
        elif operation == "install_gguf":
            result = install_hf_gguf(
                config["repo_id"], config["filename"], config["model_name"],
                config["register_ollama"], on_log=log, on_progress=progress,
                install_native=config.get("install_native", False),
            )
        elif operation == "convert_gguf":
            result = convert_local_model_to_gguf(config["source_dir"], config["llama_cpp_dir"], config["output_path"], config["outtype"], config["quantization"], on_log=log)
        else:
            raise ValueError("Unsupported Model Studio operation.")
        _update(job_id, status="completed", result=result, error=None, finished_at=_now(), append_log="Operation completed.")
    except Exception as exc:
        message = str(exc).strip() or "Operation failed."
        _update(job_id, status="failed", error=message[:1200], finished_at=_now(), append_log=f"Failed: {message[:450]}")


def start_job(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("Model Studio job configuration must be an object.")
    operation = config.get("operation")
    if operation == "pull":
        clean = {"operation": operation, "model_name": validate_ollama_name(config.get("model_name", ""))}
        if not shutil.which("ollama"):
            raise ModelStudioError("Ollama is not installed. Install it before pulling from the Ollama library.")
    elif operation == "pull_native":
        clean = {"operation": operation, "repo_id": validate_repo_id(config.get("repo_id", ""))}
    elif operation == "install_gguf":
        repo = validate_repo_id(config.get("repo_id", ""))
        filename = validate_gguf_filename(config.get("filename", ""))
        register = config.get("register_ollama", True)
        install_native = config.get("install_native", False)
        if not isinstance(register, bool):
            raise ValueError("register_ollama must be true or false.")
        if not isinstance(install_native, bool):
            raise ValueError("install_native must be true or false.")
        if install_native and register:
            raise ValueError("Direct native installation must disable register_ollama.")
        if register:
            name = validate_ollama_name(config.get("model_name", ""))
            _require_loopback_ollama()
            if not shutil.which("ollama"):
                raise ModelStudioError("Ollama is not installed. Uncheck register with Ollama to download the GGUF file only.")
        else:
            raw_name = config.get("model_name", "")
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError("Enter a local model label for the downloaded file.")
            name = _slug(raw_name)
        clean = {
            "operation": operation, "repo_id": repo, "filename": filename,
            "model_name": name, "register_ollama": register, "install_native": install_native,
        }
    elif operation == "convert_gguf":
        required = ("source_dir", "llama_cpp_dir", "output_path")
        if any(not isinstance(config.get(key), str) or not config[key].strip() for key in required):
            raise ValueError("Source model, llama.cpp directory, and output path are required.")
        outtype = config.get("outtype", "f16")
        quantization = config.get("quantization", "Q4_K_M")
        if outtype not in {"auto", "f16", "f32", "bf16"}:
            raise ValueError("Unsupported output type.")
        if quantization not in {"none", "Q2_K", "Q3_K_M", "Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"}:
            raise ValueError("Unsupported quantization type.")
        clean = {"operation": operation, "source_dir": config["source_dir"], "llama_cpp_dir": config["llama_cpp_dir"], "output_path": config["output_path"], "outtype": outtype, "quantization": quantization}
    else:
        raise ValueError("Choose pull, pull_native, install_gguf, or convert_gguf.")
    with _ACTIVE_LOCK:
        active = [job for job in list_jobs(100) if job.get("status") in {"queued", "running"}]
        if active:
            raise ModelStudioError("Another Model Studio operation is active. Wait for it to finish before starting another.")
        job_id = uuid.uuid4().hex
        job = {"schema_version": 1, "id": job_id, "status": "queued", "operation": operation,
               "created_at": _now(), "updated_at": _now(), "started_at": None, "finished_at": None,
               "runner_pid": os.getpid(), "config": clean, "progress_bytes": 0, "total_bytes": None,
               "logs": ["Job queued."], "error": None, "result": None}
        _write_job(job)
        thread = threading.Thread(target=_worker, args=(job_id,), name=f"axiom-model-studio-{job_id[:8]}", daemon=True)
        thread.start()
        return job


def model_studio_status() -> dict[str, Any]:
    executable = shutil.which("ollama")
    status = {"schema_version": 1, "ollama_installed": executable is not None, "ollama_endpoint": "http://127.0.0.1:11434",
              "download_limit_bytes": _MAX_DOWNLOAD_BYTES, "gguf_directory": str(studio_home() / "gguf"),
              "native_model_root": str(native_model_root()), "native_model_download_limit_bytes": _MAX_NATIVE_MODEL_BYTES,
              "native_model_file_limit_bytes": _MAX_NATIVE_MODEL_FILE_BYTES, "native_pull_available": True,
              "catalogue": "Hugging Face public model API", "network_search_is_explicit": True}
    return status
