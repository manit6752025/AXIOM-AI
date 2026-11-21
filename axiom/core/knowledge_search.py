"""Bounded lexical/vector retrieval over a workspace knowledge base."""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from axiom.core.embeddings import EmbeddingEngineError, embed_local_texts
from axiom.core.knowledge import (
    KnowledgeBaseStore,
    KnowledgeStoreCapacityError,
    KnowledgeStoreError,
    _ALLOWED_SUFFIXES,
    _BASE_ID,
    _workspace_path,
)

MAX_SEARCH_DOCUMENTS = 64
MAX_SEARCH_CHUNKS = 128
MAX_SEARCH_SOURCE_BYTES = 16 * 1024 * 1024
MAX_SEARCH_SOURCE_CHARS = 1_048_576
MAX_SEARCH_QUERY_CHARS = 8_192
MAX_SEARCH_RESULT_LIMIT = 20
MAX_SEARCH_PREFIX_FILTERS = 16
MAX_SEARCH_ID_FILTERS = 64
MAX_SEARCH_FORMAT_FILTERS = len(_ALLOWED_SUFFIXES)
_TOKEN_PATTERN = re.compile(r"[^\W_]{2,}(?:[-'][^\W_]+)*", re.UNICODE)


class KnowledgeSearchError(ValueError):
    """A bounded knowledge retrieval request failed validation or execution."""


def _tokens(value: str) -> list[str]:
    return [item.casefold() for item in _TOKEN_PATTERN.findall(value)]


def _validate_filters(
    *,
    formats: list[str] | tuple[str, ...] | None,
    source_prefixes: list[str] | tuple[str, ...] | None,
    document_ids: list[str] | tuple[str, ...] | None,
) -> tuple[set[str] | None, tuple[str, ...], set[str] | None]:
    allowed_formats = {suffix[1:] for suffix in _ALLOWED_SUFFIXES}
    normalized_formats: set[str] | None = None
    if formats is not None:
        if not isinstance(formats, (list, tuple)) or len(formats) > MAX_SEARCH_FORMAT_FILTERS:
            raise KnowledgeSearchError("Provide at most seven format filters.")
        normalized_formats = set()
        for item in formats:
            if not isinstance(item, str):
                raise KnowledgeSearchError("Format filters must be text.")
            value = item.strip().lower().lstrip(".")
            if value not in allowed_formats:
                raise KnowledgeSearchError(f"Unsupported format filter: {item}.")
            normalized_formats.add(value)
        if not normalized_formats:
            normalized_formats = None

    prefixes: list[str] = []
    if source_prefixes is not None:
        if not isinstance(source_prefixes, (list, tuple)) or len(source_prefixes) > MAX_SEARCH_PREFIX_FILTERS:
            raise KnowledgeSearchError(f"Provide at most {MAX_SEARCH_PREFIX_FILTERS} source-path prefixes.")
        for value in source_prefixes:
            try:
                prefixes.append(_workspace_path(value).rstrip("/"))
            except KnowledgeStoreError as exc:
                raise KnowledgeSearchError(str(exc)) from None

    normalized_ids: set[str] | None = None
    if document_ids is not None:
        if not isinstance(document_ids, (list, tuple)) or len(document_ids) > MAX_SEARCH_ID_FILTERS:
            raise KnowledgeSearchError(f"Provide at most {MAX_SEARCH_ID_FILTERS} document ID filters.")
        normalized_ids = set()
        for value in document_ids:
            if not isinstance(value, str) or not _BASE_ID.fullmatch(value):
                raise KnowledgeSearchError("Document ID filters must be 32-character hexadecimal IDs.")
            normalized_ids.add(value)
        if not normalized_ids:
            normalized_ids = None
    return normalized_formats, tuple(prefixes), normalized_ids


def _matches_filters(
    document: dict[str, Any],
    *,
    formats: set[str] | None,
    source_prefixes: tuple[str, ...],
    document_ids: set[str] | None,
) -> bool:
    if formats is not None and document["format"] not in formats:
        return False
    if document_ids is not None and document["id"] not in document_ids:
        return False
    if source_prefixes and not any(
        document["source_path"] == prefix or document["source_path"].startswith(prefix + "/")
        for prefix in source_prefixes
    ):
        return False
    return True


def _chunk_document(
    text: str,
    *,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    document: dict[str, Any],
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + chunk_size_chars)
        end = hard_end
        if hard_end < len(text):
            boundary_floor = start + chunk_size_chars // 2
            candidates = [
                text.rfind(separator, boundary_floor, hard_end)
                for separator in ("\n\n", "\n", ". ", " ")
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (2 if text.startswith("\n\n", boundary) else 1 if text.startswith(". ", boundary) else 0)
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if right > left:
            chunks.append({
                "text": text[left:right],
                "document_id": document["id"],
                "source_path": document["source_path"],
                "file_name": document["file_name"],
                "format": document["format"],
                "content_sha256": document["content_sha256"],
                "chunk_index": len(chunks) + 1,
                "start_char": left,
                "end_char": right,
                "character_count": right - left,
            })
        if end >= len(text):
            break
        start = max(start + 1, end - chunk_overlap_chars)
        while start < len(text) and text[start].isspace():
            start += 1
        if len(chunks) > MAX_SEARCH_CHUNKS:
            raise KnowledgeStoreCapacityError(
                f"Search exceeds the {MAX_SEARCH_CHUNKS}-chunk limit; narrow the metadata filters."
            )
    return chunks


def _bm25_scores(query_tokens: list[str], chunks: list[dict[str, Any]]) -> list[float]:
    if not chunks:
        return []
    query_terms = set(query_tokens)
    token_lists = [_tokens(item["text"]) for item in chunks]
    term_counts = [Counter(tokens) for tokens in token_lists]
    document_frequency = {
        term: sum(1 for counts in term_counts if counts.get(term, 0))
        for term in query_terms
    }
    average_length = sum(len(tokens) for tokens in token_lists) / len(token_lists) or 1.0
    k1, b = 1.2, 0.75
    scores = []
    for tokens, counts in zip(token_lists, term_counts, strict=True):
        score = 0.0
        length = len(tokens)
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse = math.log(1.0 + (len(chunks) - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1.0 - b + b * length / average_length)
            score += inverse * (frequency * (k1 + 1.0) / denominator)
        scores.append(score)
    maximum = max(scores, default=0.0)
    if maximum <= 0:
        return [0.0] * len(scores)
    return [score / maximum for score in scores]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise KnowledgeSearchError("Embedding vectors have incompatible dimensions.")
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if not math.isfinite(dot) or left_norm <= 0 or right_norm <= 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _vector_scores(
    model_path: str,
    query: str,
    chunks: list[dict[str, Any]],
    *,
    device: str,
) -> list[float]:
    query_result = embed_local_texts(model_path, [query], device=device, batch_size=1, max_length=256)
    query_vector = query_result["embeddings"][0]
    scores: list[float] = []
    for offset in range(0, len(chunks), 31):
        batch = chunks[offset:offset + 31]
        result = embed_local_texts(
            model_path,
            [item["text"] for item in batch],
            device=device,
            batch_size=min(16, len(batch)),
            max_length=256,
            chunk_size_chars=4096,
            chunk_overlap_chars=0,
        )
        for vector in result["embeddings"]:
            cosine = _cosine_similarity(query_vector, vector)
            scores.append((cosine + 1.0) / 2.0)
    return scores


def search_knowledge_base(
    store: KnowledgeBaseStore,
    base_id: str,
    query: str,
    *,
    limit: int = 8,
    max_documents: int = 64,
    chunk_size_chars: int = 1200,
    chunk_overlap_chars: int = 160,
    formats: list[str] | tuple[str, ...] | None = None,
    source_prefixes: list[str] | tuple[str, ...] | None = None,
    document_ids: list[str] | tuple[str, ...] | None = None,
    embedding_model_path: str | None = None,
    vector_weight: float = 0.5,
    device: str = "auto",
) -> dict[str, Any]:
    """Perform deterministic bounded lexical retrieval, optionally fused with local vectors."""
    if not isinstance(base_id, str) or not _BASE_ID.fullmatch(base_id):
        raise KnowledgeSearchError("Knowledge base ID is invalid.")
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_SEARCH_QUERY_CHARS:
        raise KnowledgeSearchError(f"Query must contain 1 to {MAX_SEARCH_QUERY_CHARS} characters.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SEARCH_RESULT_LIMIT:
        raise KnowledgeSearchError(f"Result limit must be an integer from 1 through {MAX_SEARCH_RESULT_LIMIT}.")
    if isinstance(max_documents, bool) or not isinstance(max_documents, int) or not 1 <= max_documents <= MAX_SEARCH_DOCUMENTS:
        raise KnowledgeSearchError(f"Document limit must be an integer from 1 through {MAX_SEARCH_DOCUMENTS}.")
    if isinstance(chunk_size_chars, bool) or not isinstance(chunk_size_chars, int) or not 256 <= chunk_size_chars <= 4096:
        raise KnowledgeSearchError("Chunk size must be an integer from 256 through 4096 characters.")
    if (
        isinstance(chunk_overlap_chars, bool)
        or not isinstance(chunk_overlap_chars, int)
        or not 0 <= chunk_overlap_chars <= 512
        or chunk_overlap_chars >= chunk_size_chars
    ):
        raise KnowledgeSearchError("Chunk overlap must be from 0 through 512 and smaller than chunk size.")
    if isinstance(vector_weight, bool) or not isinstance(vector_weight, (int, float)) or not 0 <= vector_weight <= 1:
        raise KnowledgeSearchError("Vector weight must be between 0 and 1.")
    if device not in {"auto", "cpu", "cuda", "rocm", "mps", "xpu"}:
        raise KnowledgeSearchError("Device must be one of auto, cpu, cuda, rocm, mps, or xpu.")
    if embedding_model_path is not None and (not isinstance(embedding_model_path, str) or not embedding_model_path.strip()):
        raise KnowledgeSearchError("Embedding model path must be non-empty text when supplied.")

    query_tokens = _tokens(query)
    if not query_tokens:
        raise KnowledgeSearchError("Query must contain at least one searchable word.")

    selected_formats, prefixes, selected_ids = _validate_filters(
        formats=formats, source_prefixes=source_prefixes, document_ids=document_ids,
    )
    listing = store.list_documents(base_id)
    candidates = [
        item for item in listing["documents"]
        if _matches_filters(
            item, formats=selected_formats, source_prefixes=prefixes, document_ids=selected_ids,
        )
    ][:max_documents]
    if len(candidates) > MAX_SEARCH_DOCUMENTS:
        raise KnowledgeStoreCapacityError("Search exceeds the candidate document limit.")
    source_bytes = sum(item["size_bytes"] for item in candidates)
    if source_bytes > MAX_SEARCH_SOURCE_BYTES:
        raise KnowledgeStoreCapacityError(
            "Filtered documents exceed the 16 MiB source-read limit; narrow the filters or reduce max_documents."
        )

    chunks: list[dict[str, Any]] = []
    scanned_chars = 0
    for document in candidates:
        text = store.read_document(base_id, document["id"])
        scanned_chars += len(text)
        if scanned_chars > MAX_SEARCH_SOURCE_CHARS:
            raise KnowledgeStoreCapacityError(
                "Filtered documents exceed the 1 Mi-character search limit; narrow the filters."
            )
        remaining = MAX_SEARCH_CHUNKS - len(chunks)
        if remaining <= 0:
            raise KnowledgeStoreCapacityError("Search exceeds the chunk limit; narrow the metadata filters.")
        generated = _chunk_document(
            text,
            chunk_size_chars=chunk_size_chars,
            chunk_overlap_chars=chunk_overlap_chars,
            document=document,
        )
        if len(generated) > remaining:
            raise KnowledgeStoreCapacityError("Search exceeds the chunk limit; narrow the metadata filters.")
        chunks.extend(generated)

    lexical = _bm25_scores(query_tokens, chunks)
    vector: list[float] | None = None
    effective_vector_weight = 0.0
    if embedding_model_path is not None and chunks:
        try:
            vector = _vector_scores(embedding_model_path, query, chunks, device=device)
        except EmbeddingEngineError:
            raise
        effective_vector_weight = float(vector_weight)

    ranked = []
    for index, chunk in enumerate(chunks):
        lexical_score = lexical[index]
        vector_score = vector[index] if vector is not None else None
        if vector_score is None:
            score = lexical_score
        else:
            score = (1.0 - effective_vector_weight) * lexical_score + effective_vector_weight * vector_score
        if score <= 0:
            continue
        snippet = chunk["text"].strip()
        ranked.append({
            "score": round(score, 8),
            "lexical_score": round(lexical_score, 8),
            "vector_score": None if vector_score is None else round(vector_score, 8),
            "snippet": snippet[:320],
            "citation": {
                "document_id": chunk["document_id"],
                "file_name": chunk["file_name"],
                "source_path": chunk["source_path"],
                "format": chunk["format"],
                "content_sha256": chunk["content_sha256"],
                "chunk_index": chunk["chunk_index"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "character_count": chunk["character_count"],
            },
        })
    ranked.sort(key=lambda item: (
        -item["score"], -item["lexical_score"], item["citation"]["source_path"].casefold(),
        item["citation"]["chunk_index"], item["citation"]["document_id"],
    ))
    results = ranked[:limit]
    return {
        "schema_version": 1,
        "engine": "axiom-hybrid-knowledge-search" if vector is not None else "axiom-lexical-knowledge-search",
        "knowledge_base": listing["knowledge_base"],
        "query": {
            "character_count": len(query),
            "candidate_documents": len(candidates),
            "chunks_scanned": len(chunks),
            "result_limit": limit,
            "chunk_size_chars": chunk_size_chars,
            "chunk_overlap_chars": chunk_overlap_chars,
            "vector_enabled": vector is not None,
            "vector_weight": effective_vector_weight,
            "formats": sorted(selected_formats) if selected_formats is not None else [],
            "source_prefixes": list(prefixes),
            "document_ids": sorted(selected_ids) if selected_ids is not None else [],
        },
        "results": results,
        "total_results": len(ranked),
        "privacy": {
            "source_documents_modified": False,
            "results_persisted": False,
            "network_used": False,
            "full_source_documents_returned": False,
            "snippets_returned": True,
        },
        "scope_note": (
            "Results are ranked within this workspace knowledge base. Each result includes a content hash and "
            "character offsets for verification. Search is bounded and does not modify, upload, or persist source documents."
        ),
    }
