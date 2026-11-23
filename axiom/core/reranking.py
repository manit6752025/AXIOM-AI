"""Bounded local cross-encoder reranking with deterministic score contracts."""
from __future__ import annotations

import gc
import math
import threading
from typing import Any
from pathlib import Path

from axiom.core.engine import resolve_local_model_directory
from axiom.core.embeddings import (
    EmbeddingEngineError,
    _select_device,
    _validate_safetensors_checkpoint,
)

MAX_RERANK_DOCUMENTS = 64
MAX_RERANK_TEXT_CHARS = 8_192
MAX_RERANK_TOTAL_CHARS = 262_144
MAX_RERANK_BATCH_SIZE = 16
MAX_RERANK_LENGTH = 512

_LOCK = threading.RLock()
_CACHE: dict[str, Any] | None = None
_CACHE_KEY: tuple[str, str] | None = None


class RerankerError(RuntimeError):
    """A local model, device, or scoring operation failed safely."""


def validate_rerank_inputs(query: str, documents, *, batch_size=8, max_length=256, device="auto") -> None:
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_RERANK_TEXT_CHARS:
        raise ValueError(f"Query must contain 1 to {MAX_RERANK_TEXT_CHARS} characters.")
    if not isinstance(documents, (list, tuple)) or not 1 <= len(documents) <= MAX_RERANK_DOCUMENTS:
        raise ValueError(f"Provide between 1 and {MAX_RERANK_DOCUMENTS} documents.")
    total = len(query)
    for index, document in enumerate(documents, 1):
        if not isinstance(document, str) or not document.strip():
            raise ValueError(f"Document {index} must contain non-whitespace characters.")
        if len(document) > MAX_RERANK_TEXT_CHARS:
            raise ValueError(f"Document {index} exceeds the per-document {MAX_RERANK_TEXT_CHARS}-character limit.")
        total += len(document)
    if total > MAX_RERANK_TOTAL_CHARS:
        raise ValueError(f"Combined query and documents exceed {MAX_RERANK_TOTAL_CHARS} characters.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= MAX_RERANK_BATCH_SIZE:
        raise ValueError(f"Batch size must be an integer from 1 through {MAX_RERANK_BATCH_SIZE}.")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or not 32 <= max_length <= MAX_RERANK_LENGTH:
        raise ValueError(f"Maximum token length must be an integer from 32 through {MAX_RERANK_LENGTH}.")
    if not isinstance(device, str) or device not in {"auto", "cpu", "cuda", "rocm", "mps", "xpu"}:
        raise ValueError("Device must be one of auto, cpu, cuda, rocm, mps, or xpu.")


def _load_reranker(model_path: str, requested_device: str) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RerankerError(
            "Local reranking needs the optional dependencies. Install with: python -m pip install -e '.[embeddings]'"
        ) from exc
    try:
        canonical = Path(resolve_local_model_directory(model_path, "Base model"))
        _validate_safetensors_checkpoint(canonical)
        device = _select_device(torch, requested_device)
        tokenizer = AutoTokenizer.from_pretrained(
            str(canonical), local_files_only=True, trust_remote_code=False, use_fast=True
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RerankerError("The local tokenizer has no padding token or EOS token.")
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            str(canonical), local_files_only=True, trust_remote_code=False, use_safetensors=True
        )
        model.to(device)
        model.eval()
        labels = getattr(model.config, "num_labels", None)
        if labels not in {1, 2}:
            raise RerankerError("The reranker must expose one regression logit or two classification logits.")
        return {"torch": torch, "tokenizer": tokenizer, "model": model, "device": device,
                "labels": labels, "name": canonical.name}
    except RerankerError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        raise RerankerError("The selected local checkpoint could not be loaded as a safe cross-encoder reranker.") from None


def _clear_cache_locked() -> None:
    global _CACHE, _CACHE_KEY
    cached = _CACHE
    _CACHE = None
    _CACHE_KEY = None
    if cached is None:
        return
    try:
        cached["model"].to("cpu")
    except Exception:
        pass
    torch = cached.get("torch")
    del cached
    gc.collect()
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def unload_local_reranker() -> dict[str, Any]:
    """Unload the cached cross-encoder from device memory without touching model files."""
    global _CACHE
    with _LOCK:
        was_loaded = _CACHE is not None
        _clear_cache_locked()
    return {"schema_version": 1, "unloaded": was_loaded,
            "scope_note": "Only AXIOM's cached reranker was unloaded; model files were not changed."}


def _ranked_results(scores: list[float], documents: list[str]) -> list[dict[str, Any]]:
    """Return score metadata in descending order with stable ties."""
    if len(scores) != len(documents):
        raise RerankerError("The reranker returned a different number of scores than input documents.")
    if any(not isinstance(score, (int, float)) or not math.isfinite(score) for score in scores):
        raise RerankerError("The reranker produced an invalid score.")
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return [
        {"rank": rank, "document_index": index + 1, "score": round(float(scores[index]), 8),
         "document_characters": len(documents[index])}
        for rank, index in enumerate(order, start=1)
    ]


def rerank_local_documents(
    model_path: str,
    query: str,
    documents,
    *,
    batch_size: int = 8,
    max_length: int = 256,
    device: str = "auto",
    _runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score query/document pairs locally and sort with stable, deterministic ties."""
    validate_rerank_inputs(query, documents, batch_size=batch_size, max_length=max_length, device=device)
    global _CACHE, _CACHE_KEY
    with _LOCK:
        if _runtime is None:
            canonical = resolve_local_model_directory(model_path, "Base model")
            key = (canonical, device)
            if _CACHE is None or _CACHE_KEY != key:
                _clear_cache_locked()
                _CACHE = _load_reranker(canonical, device)
                _CACHE_KEY = key
            runtime = _CACHE
        else:
            runtime = _runtime
        torch, tokenizer, model = runtime["torch"], runtime["tokenizer"], runtime["model"]
        selected_device, labels = runtime["device"], runtime["labels"]
        scores: list[float] = []
        try:
            with torch.inference_mode():
                for offset in range(0, len(documents), batch_size):
                    candidates = documents[offset:offset + batch_size]
                    encoded = tokenizer(
                        [query] * len(candidates),
                        candidates,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    )
                    encoded = {key: value.to(selected_device) for key, value in encoded.items()}
                    output = model(**encoded)
                    logits = getattr(output, "logits", None)
                    if logits is None or logits.ndim != 2 or logits.shape[1] != labels:
                        raise RerankerError("The cross-encoder returned an unexpected score tensor.")
                    if not bool(torch.isfinite(logits).all()):
                        raise RerankerError("The cross-encoder returned non-finite scores.")
                    if labels == 1:
                        batch_scores = torch.sigmoid(logits[:, 0])
                    else:
                        batch_scores = torch.softmax(logits, dim=1)[:, 1]
                    scores.extend(float(value) for value in batch_scores.cpu().tolist())
        except RerankerError:
            raise
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, IndexError):
            raise RerankerError("Local reranking failed; query and document text were not retained.") from None

    return {
        "schema_version": 1,
        "engine": "axiom-local-cross-encoder",
        "model": {"name": str(runtime.get("name") or Path(model_path).name), "score_mode": "sigmoid" if labels == 1 else "positive_class_softmax"},
        "device": str(selected_device),
        "request": {"document_count": len(documents), "batch_size": batch_size, "max_length": max_length,
                    "query_characters": len(query), "total_document_characters": sum(len(value) for value in documents)},
        "results": _ranked_results(scores, documents),
        "privacy": {"query_returned": False, "documents_returned": False, "text_persisted": False,
                    "model_files_uploaded": False, "network_used": False},
        "scope_note": (
            "Scores are ranking signals from a local cross-encoder, not calibrated probabilities or relevance guarantees. "
            "Ties preserve original input order; query and document text are not returned or persisted."
        ),
    }
