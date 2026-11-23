"""Bounded local-only text embeddings through trusted safetensors models."""
from __future__ import annotations

import gc
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from axiom.core.engine import _allowed_model_roots, _trusted_file_entry, resolve_local_model_directory

MAX_EMBEDDING_INPUTS = 32
MAX_EMBEDDING_TEXT_CHARS = 65_536
MAX_EMBEDDING_TOTAL_CHARS = 262_144
MAX_EMBEDDING_CHUNKS = 128
MAX_EMBEDDING_DIMENSION = 16_384
MAX_EMBEDDING_VECTOR_VALUES = 2_000_000
MAX_EMBEDDING_BATCH_SIZE = 16
_RUNTIME_LOCK = threading.RLock()
_RUNTIME_CACHE: dict[str, Any] | None = None
_RUNTIME_CACHE_KEY: tuple[str, str] | None = None


class EmbeddingEngineError(RuntimeError):
    """A local model, device, or embedding operation failed safely."""


def validate_embedding_inputs(texts, *, batch_size=8, max_length=256, chunk_size_chars=2048,
                              chunk_overlap_chars=128, device="auto", normalize=True) -> None:
    if not isinstance(texts, (list, tuple)) or not 1 <= len(texts) <= MAX_EMBEDDING_INPUTS:
        raise ValueError(f"Provide between 1 and {MAX_EMBEDDING_INPUTS} text inputs.")
    total = 0
    for index, text in enumerate(texts, 1):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Text input {index} must contain non-whitespace characters.")
        if len(text) > MAX_EMBEDDING_TEXT_CHARS:
            raise ValueError(f"Text input {index} exceeds the per-input {MAX_EMBEDDING_TEXT_CHARS}-character limit.")
        total += len(text)
    if total > MAX_EMBEDDING_TOTAL_CHARS:
        raise ValueError(f"Combined input exceeds the {MAX_EMBEDDING_TOTAL_CHARS}-character limit.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= MAX_EMBEDDING_BATCH_SIZE:
        raise ValueError(f"Batch size must be an integer from 1 through {MAX_EMBEDDING_BATCH_SIZE}.")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or not 8 <= max_length <= 512:
        raise ValueError("Maximum token length must be an integer from 8 through 512.")
    if isinstance(chunk_size_chars, bool) or not isinstance(chunk_size_chars, int) or not 256 <= chunk_size_chars <= 8192:
        raise ValueError("Chunk size must be an integer from 256 through 8192 characters.")
    if (isinstance(chunk_overlap_chars, bool) or not isinstance(chunk_overlap_chars, int)
        or not 0 <= chunk_overlap_chars <= 1024 or chunk_overlap_chars >= chunk_size_chars):
        raise ValueError("Chunk overlap must be bounded and smaller than the chunk size.")
    if not isinstance(device, str) or device not in {"auto", "cpu", "cuda", "rocm", "mps", "xpu"}:
        raise ValueError("Device must be one of auto, cpu, cuda, rocm, mps, or xpu.")
    if not isinstance(normalize, bool):
        raise ValueError("Embedding normalization must be a boolean.")


def split_text_into_chunks(texts, *, chunk_size_chars=2048, chunk_overlap_chars=128):
    """Split inputs and retain offsets, never source text in the public result."""
    if not isinstance(texts, (list, tuple)) or not texts:
        raise ValueError("At least one text input is required.")
    if isinstance(chunk_size_chars, bool) or not isinstance(chunk_size_chars, int) or not 256 <= chunk_size_chars <= 8192:
        raise ValueError("Chunk size must be from 256 through 8192 characters.")
    if (isinstance(chunk_overlap_chars, bool) or not isinstance(chunk_overlap_chars, int)
        or not 0 <= chunk_overlap_chars <= 1024 or chunk_overlap_chars >= chunk_size_chars):
        raise ValueError("Chunk overlap must be bounded and smaller than the chunk size.")
    result = []
    for input_index, text in enumerate(texts, 1):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Text input {input_index} must contain non-whitespace characters.")
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(len(text), start + chunk_size_chars)
            if end < len(text):
                floor = start + chunk_size_chars // 2
                boundary = max(text.rfind(separator, floor, end) for separator in ("\n", " ", "\t"))
                if boundary > start:
                    end = boundary
            left, right = start, end
            while left < right and text[left].isspace():
                left += 1
            while right > left and text[right - 1].isspace():
                right -= 1
            if right > left:
                chunk_index += 1
                result.append({"text": text[left:right], "input_index": input_index,
                    "chunk_index": chunk_index, "start_char": left, "end_char": right,
                    "characters": right - left})
                if len(result) > MAX_EMBEDDING_CHUNKS:
                    raise ValueError(f"Input expands beyond the {MAX_EMBEDDING_CHUNKS}-chunk request limit.")
            if end >= len(text):
                break
            next_start = max(start + 1, end - chunk_overlap_chars)
            while next_start < len(text) and text[next_start].isspace():
                next_start += 1
            start = next_start
    if not result:
        raise ValueError("The supplied text did not produce any embeddable chunks.")
    return result


def _trusted_file(model_dir: Path, relative: str, roots: tuple[Path, ...]) -> Path:
    root = next((candidate for candidate in roots if model_dir.is_relative_to(candidate)), None)
    path = model_dir / relative
    if root is None or not _trusted_file_entry(str(path), str(root)):
        raise EmbeddingEngineError("The selected embedding checkpoint contains an untrusted or missing asset.")
    return path


def _validate_safetensors_checkpoint(model_dir: Path) -> None:
    roots = _allowed_model_roots()
    root = next((candidate for candidate in roots if model_dir.is_relative_to(candidate)), None)
    if root is None:
        raise EmbeddingEngineError("The selected embedding checkpoint is outside configured trusted roots.")
    config = _trusted_file(model_dir, "config.json", roots)
    try:
        if config.stat().st_size > 1_048_576 or not isinstance(json.loads(config.read_text(encoding="utf-8")), dict):
            raise ValueError("invalid config")
    except (OSError, UnicodeError, ValueError):
        raise EmbeddingEngineError("The model configuration is invalid or unreadable.") from None
    directories = file_count = 0
    for directory, subdirectories, filenames in os.walk(model_dir, topdown=True, followlinks=False):
        directories += 1
        if directories > 2_000:
            raise EmbeddingEngineError("The embedding checkpoint contains too many directories.")
        if any((Path(directory) / name).is_symlink() for name in subdirectories):
            raise EmbeddingEngineError("The embedding checkpoint contains a directory symlink.")
        for name in filenames:
            file_count += 1
            if file_count > 20_000:
                raise EmbeddingEngineError("The embedding checkpoint contains too many files.")
            _trusted_file(model_dir, str((Path(directory) / name).relative_to(model_dir)), roots)
    single = model_dir / "model.safetensors"
    if _trusted_file_entry(str(single), str(root)):
        return
    index_name = next((name for name in ("model.safetensors.index.json", "pytorch_model.safetensors.index.json")
        if _trusted_file_entry(str(model_dir / name), str(root))), None)
    if index_name is None:
        raise EmbeddingEngineError("Embedding models must contain safetensors weights; pickle checkpoints are not loaded.")
    index = _trusted_file(model_dir, index_name, roots)
    try:
        if index.stat().st_size > 1_048_576:
            raise ValueError("index too large")
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("missing weight map")
        shards = set(weight_map.values())
        if not 1 <= len(shards) <= 32:
            raise ValueError("shard count")
        for shard in shards:
            if (not isinstance(shard, str) or not shard.endswith(".safetensors")
                or Path(shard).is_absolute() or ".." in Path(shard).parts or "\\" in shard):
                raise ValueError("unsafe shard path")
            _trusted_file(model_dir, shard, roots)
    except EmbeddingEngineError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError):
        raise EmbeddingEngineError("The safetensors shard index is invalid or unreadable.") from None


def _select_device(torch, requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"cuda", "rocm"}:
        if not torch.cuda.is_available():
            raise EmbeddingEngineError("The requested CUDA-compatible device is unavailable.")
        is_rocm = bool(getattr(torch.version, "hip", None))
        if requested == "rocm" and not is_rocm:
            raise EmbeddingEngineError("The requested ROCm device is unavailable in this PyTorch build.")
        if requested == "cuda" and is_rocm:
            raise EmbeddingEngineError("This PyTorch build uses ROCm; request --device rocm instead.")
        return torch.device("cuda")
    if requested == "mps":
        backend = getattr(getattr(torch, "backends", None), "mps", None)
        if backend is None or not backend.is_available():
            raise EmbeddingEngineError("The requested Metal/MPS device is unavailable.")
        return torch.device("mps")
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not xpu.is_available():
        raise EmbeddingEngineError("The requested Intel XPU device is unavailable.")
    return torch.device("xpu")


def _load_runtime(model_path, requested_device):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise EmbeddingEngineError("Install local embedding dependencies with: python -m pip install -e '.[embeddings]'") from exc
    try:
        canonical = Path(resolve_local_model_directory(model_path, "Base model"))
        _validate_safetensors_checkpoint(canonical)
        device = _select_device(torch, requested_device)
        tokenizer = AutoTokenizer.from_pretrained(str(canonical), local_files_only=True,
            trust_remote_code=False, use_fast=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise EmbeddingEngineError("The local tokenizer has no padding token or EOS token.")
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(str(canonical), local_files_only=True,
            trust_remote_code=False, use_safetensors=True)
        model.to(device)
        model.eval()
        dimension = getattr(model.config, "hidden_size", None)
        if isinstance(dimension, bool) or not isinstance(dimension, int) or not 1 <= dimension <= MAX_EMBEDDING_DIMENSION:
            raise EmbeddingEngineError("The model embedding dimension is missing or exceeds the safe limit.")
        return {"torch": torch, "tokenizer": tokenizer, "model": model, "device": device,
            "dimension": dimension, "name": canonical.name}
    except EmbeddingEngineError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        raise EmbeddingEngineError("The selected checkpoint could not be loaded as a safe local embedding model.") from None


def _unload_runtime_locked():
    global _RUNTIME_CACHE, _RUNTIME_CACHE_KEY
    runtime = _RUNTIME_CACHE
    _RUNTIME_CACHE = None
    _RUNTIME_CACHE_KEY = None
    if runtime is None:
        return
    try:
        runtime["model"].to("cpu")
    except Exception:
        pass
    torch = runtime.get("torch")
    del runtime
    gc.collect()
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def unload_local_embeddings() -> dict[str, Any]:
    """Unload AXIOM's cached embedding model without changing checkpoint files."""
    global _RUNTIME_CACHE
    with _RUNTIME_LOCK:
        was_loaded = _RUNTIME_CACHE is not None
        _unload_runtime_locked()
    return {"schema_version": 1, "unloaded": was_loaded,
        "scope_note": "Only AXIOM's cached local embedding model was unloaded; model files were not changed."}


def embed_local_texts(model_path: str, texts, *, batch_size=8, max_length=256,
                      chunk_size_chars=2048, chunk_overlap_chars=128, device="auto",
                      normalize=True, _runtime=None) -> dict[str, Any]:
    """Embed bounded text inputs with attention-mask mean pooling."""
    validate_embedding_inputs(texts, batch_size=batch_size, max_length=max_length,
        chunk_size_chars=chunk_size_chars, chunk_overlap_chars=chunk_overlap_chars,
        device=device, normalize=normalize)
    chunks = split_text_into_chunks(texts, chunk_size_chars=chunk_size_chars,
        chunk_overlap_chars=chunk_overlap_chars)
    if len(chunks) > MAX_EMBEDDING_CHUNKS:
        raise ValueError(f"Input expands beyond the {MAX_EMBEDDING_CHUNKS}-chunk request limit.")
    global _RUNTIME_CACHE, _RUNTIME_CACHE_KEY
    with _RUNTIME_LOCK:
        if _runtime is None:
            canonical = resolve_local_model_directory(model_path, "Base model")
            key = (canonical, device)
            if _RUNTIME_CACHE is None or _RUNTIME_CACHE_KEY != key:
                _unload_runtime_locked()
                _RUNTIME_CACHE = _load_runtime(canonical, device)
                _RUNTIME_CACHE_KEY = key
            runtime = _RUNTIME_CACHE
        else:
            runtime = _runtime
        torch, tokenizer, model = runtime["torch"], runtime["tokenizer"], runtime["model"]
        selected_device, dimension = runtime["device"], runtime["dimension"]
        if isinstance(dimension, bool) or not isinstance(dimension, int) or not 1 <= dimension <= MAX_EMBEDDING_DIMENSION:
            raise EmbeddingEngineError("The loaded model exposes an unsupported embedding dimension.")
        if len(chunks) * dimension > MAX_EMBEDDING_VECTOR_VALUES:
            raise ValueError("The output would exceed the bounded embedding vector size.")
        vectors = []
        try:
            with torch.inference_mode():
                for offset in range(0, len(chunks), batch_size):
                    batch = chunks[offset:offset + batch_size]
                    encoded = tokenizer([item["text"] for item in batch], return_tensors="pt",
                        padding=True, truncation=True, max_length=max_length)
                    encoded = {key: value.to(selected_device) for key, value in encoded.items()}
                    attention = encoded.get("attention_mask")
                    if attention is None:
                        raise EmbeddingEngineError("The tokenizer did not provide an attention mask.")
                    output = model(**encoded)
                    hidden = getattr(output, "last_hidden_state", None)
                    if hidden is None:
                        raise EmbeddingEngineError("The model did not return token-level hidden states.")
                    mask = attention.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    if normalize:
                        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    pooled = pooled.float()
                    if not bool(torch.isfinite(pooled).all()):
                        raise EmbeddingEngineError("The model returned non-finite embedding values.")
                    rows = pooled.cpu().tolist()
                    if not isinstance(rows, list) or any(not isinstance(row, list) or len(row) != dimension for row in rows):
                        raise EmbeddingEngineError("The model returned an unexpected embedding tensor shape.")
                    vectors.extend([[float(value) for value in row] for row in rows])
        except EmbeddingEngineError:
            raise
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, IndexError):
            raise EmbeddingEngineError("Local embedding generation failed; source text was not retained.") from None
    return {
        "schema_version": 1,
        "engine": "axiom-local-embeddings",
        "model": {"name": str(runtime.get("name") or Path(model_path).name), "dimension": dimension},
        "device": str(selected_device),
        "request": {"input_count": len(texts), "chunk_count": len(chunks), "batch_size": batch_size,
            "max_length": max_length, "chunk_size_chars": chunk_size_chars,
            "chunk_overlap_chars": chunk_overlap_chars, "normalized": bool(normalize),
            "input_characters": sum(len(text) for text in texts)},
        "chunks": [{"input_index": item["input_index"], "chunk_index": item["chunk_index"],
            "start_char": item["start_char"], "end_char": item["end_char"],
            "characters": item["characters"], "embedding_index": index}
            for index, item in enumerate(chunks, 1)],
        "embeddings": vectors,
        "privacy": {"source_text_returned": False, "source_text_persisted": False,
            "model_files_uploaded": False, "network_used": False, "results_persisted": False},
        "scope_note": "Generated locally from a trusted safetensors checkpoint. Source text is not returned, logged, uploaded, or persisted.",
        "generated_at": datetime.now(UTC).isoformat(),
    }
