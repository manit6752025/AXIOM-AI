"""GGUF ingestion for AXIOM's own dense transformer runtime.

GGUF tensors are dequantized to a selected floating-point dtype and executed
through AXIOM's own PyTorch implementation. This module does not use llama.cpp,
Transformers generation, or Ollama, and does not provide low-bit kernels.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

_MAX_GGUF_BYTES = 160 * 1024**3
_MAX_GGUF_TENSORS = 100_000
_MAX_TOKENIZER_BYTES = 64 * 1024**2
_ARCHITECTURES = {"llama": "LlamaForCausalLM", "mistral": "MistralForCausalLM",
                  "qwen2": "Qwen2ForCausalLM", "qwen3": "Qwen3ForCausalLM"}
_BASE_MAP = {"token_embd.weight": "model.embed_tokens.weight",
             "output_norm.weight": "model.norm.weight", "output.weight": "lm_head.weight",
             "output.bias": "lm_head.bias"}
_LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.bias": "self_attn.o_proj.bias",
    "ffn_gate.bias": "mlp.gate_proj.bias",
    "ffn_up.bias": "mlp.up_proj.bias",
    "ffn_down.bias": "mlp.down_proj.bias",
}


def _field(reader: Any, key: str, default: Any = None) -> Any:
    field = reader.fields.get(key)
    if field is None:
        return default
    try:
        return field.contents()
    except Exception:
        raise ValueError("GGUF metadata could not be decoded safely.") from None


def _int(reader: Any, key: str, default: int | None = None) -> int:
    value = _field(reader, key, default)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"GGUF checkpoint is missing valid integer metadata for {key}.")
    return int(value)


def _float(reader: Any, key: str, default: float) -> float:
    value = _field(reader, key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"GGUF checkpoint has invalid numeric metadata for {key}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"GGUF checkpoint has non-finite metadata for {key}.")
    return result


def _logical_shape(tensor: Any) -> tuple[int, ...]:
    # GGUF records GGML dimensions in reverse NumPy/PyTorch order.
    return tuple(reversed(tuple(int(value) for value in tensor.shape)))


_MAX_GGUF_SHARDS = 256
_MAX_GGUF_TOTAL_BYTES = 1024 * 1024**3


def _find_files(model_dir: Path, roots: tuple[Path, ...]) -> tuple[Path, ...]:
    """Resolve all GGUF files under one trusted model directory."""
    from axiom.core.native_engine import _resolve_trusted_file

    try:
        candidates = sorted(
            (item for item in model_dir.iterdir() if item.suffix.lower() == ".gguf" and item.is_file()),
            key=lambda item: item.name.casefold(),
        )
    except OSError:
        raise ValueError("Trusted model directory could not be enumerated.") from None
    if not candidates:
        raise ValueError("Trusted model directory contains no GGUF checkpoint.")
    if len(candidates) > _MAX_GGUF_SHARDS:
        raise ValueError("GGUF checkpoint contains more than AXIOM's 256-shard safety limit.")
    resolved: list[Path] = []
    total_size = 0
    for candidate in candidates:
        path = _resolve_trusted_file(candidate, roots, "GGUF checkpoint")
        try:
            size = path.stat().st_size
        except OSError:
            raise ValueError("Trusted GGUF checkpoint could not be inspected.") from None
        if not 0 < size <= _MAX_GGUF_BYTES:
            raise ValueError("Each GGUF shard must be between 1 byte and 160 GiB.")
        total_size += size
        resolved.append(path)
    if total_size > _MAX_GGUF_TOTAL_BYTES:
        raise ValueError("Combined GGUF shards exceed AXIOM's 1 TiB checkpoint safety limit.")
    return tuple(resolved)


class _MergedGGUFReader:
    """Keep all shard mmap owners alive while presenting the combined tensors."""

    def __init__(self, readers: tuple[Any, ...]):
        self.readers = readers
        self.fields = readers[0].fields
        self.tensors = [tensor for reader in readers for tensor in reader.tensors]


def _open_reader(paths: tuple[Path, ...]) -> tuple[Any, Any, Any]:
    try:
        import gguf
        from gguf.constants import GGMLQuantizationType
        from gguf.quants import dequantize
    except (ImportError, OSError):
        raise RuntimeError("GGUF support requires gguf-py. Reinstall AXIOM with the native-inference extra.") from None
    readers: list[Any] = []
    try:
        for path in paths:
            reader = gguf.GGUFReader(str(path), mode="r")
            if len(reader.tensors) > _MAX_GGUF_TENSORS:
                raise ValueError("GGUF shard contains too many tensors for AXIOM's safety limit.")
            readers.append(reader)
    except Exception:
        raise ValueError("Local GGUF checkpoint or one of its shards is malformed or uses an unsupported GGUF version.") from None
    if len(readers) == 1:
        reader = readers[0]
        reader._axiom_readers = tuple(readers)
        return reader, dequantize, GGMLQuantizationType

    shard_info: list[tuple[int, int, int]] = []
    for reader in readers:
        index = _field(reader, "split.no")
        count = _field(reader, "split.count")
        total_tensors = _field(reader, "split.tensors.count")
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in (index, count, total_tensors)):
            raise ValueError("Multiple GGUF files are present but do not contain valid split-shard metadata.")
        shard_info.append((int(index), int(count), int(total_tensors)))
    expected_count = len(readers)
    if any(count != expected_count for _index, count, _total in shard_info):
        raise ValueError("GGUF split metadata does not match the number of shard files.")
    indices = [index for index, _count, _total in shard_info]
    if sorted(indices) != list(range(expected_count)):
        raise ValueError("GGUF split metadata contains missing or duplicate shard indices.")
    total_counts = {total for _index, _count, total in shard_info}
    if len(total_counts) != 1:
        raise ValueError("GGUF shards disagree about the total tensor count.")
    ordered_readers = tuple(reader for _index, reader in sorted(zip(indices, readers), key=lambda pair: pair[0]))
    merged = _MergedGGUFReader(ordered_readers)
    declared_total = next(iter(total_counts))
    if declared_total != len(merged.tensors):
        raise ValueError("GGUF split metadata tensor count does not match the combined shards.")
    names = [tensor.name for tensor in merged.tensors]
    if len(names) != len(set(names)):
        raise ValueError("GGUF shards contain duplicate tensor names.")
    if _field(merged, "general.architecture") is None:
        raise ValueError("First GGUF shard is missing model architecture metadata.")
    merged._axiom_readers = ordered_readers
    return merged, dequantize, GGMLQuantizationType


def _configuration(reader: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from axiom.core.native_engine import NativeEngineError, NativeLlamaConfig, NativeLlamaRuntime

    arch = _field(reader, "general.architecture")
    if not isinstance(arch, str) or arch not in _ARCHITECTURES:
        raise NativeEngineError(
            "AXIOM Native Engine GGUF support is limited to dense llama, mistral, qwen2 and qwen3 architectures."
        )
    tensors = {item.name: item for item in reader.tensors}
    embedding = tensors.get("token_embd.weight")
    if embedding is None:
        raise NativeEngineError("GGUF checkpoint is missing token_embd.weight.")
    embedding_shape = _logical_shape(embedding)
    if len(embedding_shape) != 2:
        raise NativeEngineError("GGUF token embedding tensor must be a matrix.")

    prefix = arch + "."
    hidden = _int(reader, prefix + "embedding_length")
    intermediate = _int(reader, prefix + "feed_forward_length")
    layers = _int(reader, prefix + "block_count")
    heads = _int(reader, prefix + "attention.head_count")
    kv_heads = _int(reader, prefix + "attention.head_count_kv", heads)
    context = _int(reader, prefix + "context_length", 4096)
    rms_eps = _float(reader, prefix + "attention.layer_norm_rms_epsilon", 1e-5)
    theta = _float(reader, prefix + "rope.freq_base", 10000.0)
    if embedding_shape[1] != hidden:
        raise NativeEngineError("GGUF embedding shape does not match embedding_length metadata.")

    config: dict[str, Any] = {
        "model_type": arch, "architectures": [_ARCHITECTURES[arch]],
        "hidden_size": hidden, "intermediate_size": intermediate,
        "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": kv_heads, "vocab_size": embedding_shape[0],
        "max_position_embeddings": context, "rms_norm_eps": rms_eps, "rope_theta": theta,
        "bos_token_id": _field(reader, "tokenizer.ggml.bos_token_id"),
        "eos_token_id": _field(reader, "tokenizer.ggml.eos_token_id"),
        "tie_word_embeddings": "output.weight" not in tensors,
    }
    if arch == "mistral":
        sliding = _field(reader, prefix + "attention.sliding_window")
        if sliding is not None:
            if isinstance(sliding, bool) or not isinstance(sliding, (int, np.integer)) or int(sliding) < 1:
                raise NativeEngineError("GGUF Mistral sliding-window metadata must be a positive integer.")
            config["sliding_window"] = int(sliding)

    rope_type = _field(reader, prefix + "rope.scaling.type")
    if isinstance(rope_type, str) and rope_type.lower() not in {"", "none", "default"}:
        rope: dict[str, Any] = {
            "rope_type": rope_type.lower(),
            "factor": _float(reader, prefix + "rope.scaling.factor", 1.0),
        }
        if rope_type.lower() in {"llama3", "llama-3"}:
            original = _field(reader, prefix + "rope.scaling.original_context_length")
            if original is not None:
                rope["original_max_position_embeddings"] = original
            rope["low_freq_factor"] = _float(reader, prefix + "rope.scaling.low_freq_factor", 1.0)
            rope["high_freq_factor"] = _float(reader, prefix + "rope.scaling.high_freq_factor", 4.0)
            rope["rope_type"] = "llama3"
        config["rope_scaling"] = rope

    native_config = NativeLlamaConfig.from_mapping(config)
    expected = {"model.embed_tokens.weight": (native_config.vocab_size, hidden), "model.norm.weight": (hidden,)}
    head_dim = hidden // heads
    bias_dims = {"self_attn.q_proj": heads * head_dim, "self_attn.k_proj": kv_heads * head_dim,
                 "self_attn.v_proj": kv_heads * head_dim, "self_attn.o_proj": hidden,
                 "mlp.gate_proj": intermediate, "mlp.up_proj": intermediate, "mlp.down_proj": hidden}
    bias_source = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
                   "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
                   "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
    for index in range(layers):
        p = f"model.layers.{index}."
        expected.update({
            p + "self_attn.q_proj.weight": (heads * head_dim, hidden),
            p + "self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
            p + "self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
            p + "self_attn.o_proj.weight": (hidden, hidden),
            p + "mlp.gate_proj.weight": (intermediate, hidden),
            p + "mlp.up_proj.weight": (intermediate, hidden),
            p + "mlp.down_proj.weight": (hidden, intermediate),
            p + "input_layernorm.weight": (hidden,),
            p + "post_attention_layernorm.weight": (hidden,),
        })
        if native_config.qk_norm:
            expected[p + "self_attn.q_norm.weight"] = (head_dim,)
            expected[p + "self_attn.k_norm.weight"] = (head_dim,)
        for module, out_dim in bias_dims.items():
            if f"blk.{index}.{bias_source[module]}.bias" in tensors:
                expected[p + module + ".bias"] = (out_dim,)
    if "output.weight" in tensors:
        expected["lm_head.weight"] = (native_config.vocab_size, hidden)
    if "output.bias" in tensors:
        expected["lm_head.bias"] = (native_config.vocab_size,)

    source_map = dict(_BASE_MAP)
    for index in range(layers):
        for gguf_suffix, native_suffix in _LAYER_MAP.items():
            source_map[f"blk.{index}.{gguf_suffix}"] = f"model.layers.{index}.{native_suffix}"
    mapped: dict[str, Any] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for source, native_name in source_map.items():
        tensor = tensors.get(source)
        if tensor is None:
            continue
        if native_name in mapped:
            raise NativeEngineError(f"GGUF checkpoint maps more than one tensor to {native_name!r}.")
        shape = _logical_shape(tensor)
        if native_name in expected and shape != expected[native_name]:
            raise NativeEngineError(f"GGUF tensor {source!r} has a shape inconsistent with model metadata.")
        mapped[native_name] = tensor
        shapes[native_name] = shape
    missing = [name for name in NativeLlamaRuntime._required_weight_names(native_config) if name not in mapped]
    if missing:
        raise NativeEngineError("GGUF checkpoint is missing native weights: " + ", ".join(missing[:5]) + ".")
    for name, shape in expected.items():
        if name in mapped and shapes[name] != shape:
            raise NativeEngineError(f"GGUF tensor {name!r} has a shape inconsistent with model metadata.")
    return config, mapped, {
        "native_config": native_config, "architecture": _ARCHITECTURES[arch],
        "weight_shapes": shapes, "expected_shapes": expected,
    }


def _tokenizer_source(reader: Any, model_dir: Path, roots: tuple[Path, ...]) -> str:
    from axiom.core.native_engine import _resolve_trusted_file

    sidecar = model_dir / "tokenizer.json"
    try:
        if sidecar.is_file():
            path = _resolve_trusted_file(sidecar, roots, "Tokenizer file")
            if path.stat().st_size > _MAX_TOKENIZER_BYTES:
                raise ValueError("GGUF tokenizer.json sidecar exceeds the 64 MiB limit.")
            return "tokenizer.json"
    except OSError:
        raise ValueError("Trusted GGUF tokenizer sidecar is not readable.") from None
    embedded = _field(reader, "tokenizer.huggingface.json")
    if isinstance(embedded, str) and embedded:
        if len(embedded.encode("utf-8")) > _MAX_TOKENIZER_BYTES:
            raise ValueError("Embedded GGUF Hugging Face tokenizer exceeds the 64 MiB limit.")
        return "tokenizer.huggingface.json"
    tokens, model = _field(reader, "tokenizer.ggml.tokens"), _field(reader, "tokenizer.ggml.model")
    if isinstance(tokens, list) and tokens and isinstance(model, str) and model.lower() in {
        "gpt2", "bpe", "llama", "unigram", "spm", "sentencepiece"
    }:
        return "tokenizer.ggml.tokens"
    return "unsupported"


def inspect_gguf_checkpoint(model_dir: Path, trusted_roots: tuple[Path, ...] | None = None) -> dict[str, Any]:
    """Inspect metadata and shapes without dequantizing model weights."""
    from axiom.core.native_engine import NativeEngineError

    roots = tuple(root.resolve() for root in (trusted_roots or (model_dir.resolve(),)))
    paths = _find_files(model_dir, roots)
    reader, _dequantize, _qtypes = _open_reader(paths)
    try:
        config, _mapped, details = _configuration(reader)
        tokenizer_source = _tokenizer_source(reader, model_dir, roots)
        if tokenizer_source == "unsupported":
            raise NativeEngineError(
                "GGUF checkpoint has no supported tokenizer source. Add tokenizer.json, embed tokenizer.huggingface.json, or use compatible embedded tokenizer metadata."
            )
        type_counts: dict[str, int] = {}
        for tensor in reader.tensors:
            name = getattr(tensor.tensor_type, "name", str(tensor.tensor_type))
            type_counts[name] = type_counts.get(name, 0) + 1
        return {
            "config": config, "model_type": config["model_type"],
            "architecture": details["architecture"], "checkpoint_format": "gguf",
            "gguf_path": str(paths[0]) if len(paths) == 1 else None,
            "gguf_paths": [str(path) for path in paths],
            "tokenizer_source": tokenizer_source,
            "tensor_type_counts": type_counts,
            # Count native-mapped tensor elements without dequantizing them.
            # Memory planning multiplies this by the chosen floating-point dtype,
            # because the compact GGUF source size is not its runtime footprint.
            "gguf_weight_elements": sum(math.prod(shape) for shape in details["weight_shapes"].values()),
            "scope_note": "GGUF tensors are dequantized for AXIOM's own PyTorch runtime; this release does not include low-bit kernels.",
        }
    except NativeEngineError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError):
        raise NativeEngineError("GGUF checkpoint metadata is incompatible with AXIOM Native Engine.") from None


def _load_tokenizer(reader: Any, model_dir: Path, roots: tuple[Path, ...]) -> tuple[Any, str]:
    from axiom.core.native_engine import _resolve_trusted_file

    try:
        from tokenizers import Tokenizer, AddedToken, models, pre_tokenizers, decoders, normalizers
    except (ImportError, OSError):
        raise RuntimeError("AXIOM GGUF inference requires tokenizers. Install the native-inference extra.") from None
    sidecar = model_dir / "tokenizer.json"
    if sidecar.is_file():
        path = _resolve_trusted_file(sidecar, roots, "Tokenizer file")
        if path.stat().st_size > _MAX_TOKENIZER_BYTES:
            raise ValueError("GGUF tokenizer.json sidecar exceeds the 64 MiB limit.")
        try:
            return Tokenizer.from_file(str(path)), "tokenizer.json"
        except Exception:
            raise ValueError("GGUF tokenizer.json sidecar is not a valid Hugging Face tokenizer.") from None
    embedded = _field(reader, "tokenizer.huggingface.json")
    if isinstance(embedded, str) and embedded:
        if len(embedded.encode("utf-8")) > _MAX_TOKENIZER_BYTES:
            raise ValueError("Embedded GGUF Hugging Face tokenizer exceeds the 64 MiB limit.")
        try:
            return Tokenizer.from_str(embedded), "tokenizer.huggingface.json"
        except Exception:
            raise ValueError("Embedded GGUF Hugging Face tokenizer JSON is malformed.") from None

    from axiom.core.native_engine import NativeEngineError

    raw_tokens = _field(reader, "tokenizer.ggml.tokens")
    model_name = _field(reader, "tokenizer.ggml.model")
    pre_name = _field(reader, "tokenizer.ggml.pre", "")
    if not isinstance(raw_tokens, list) or not raw_tokens or len(raw_tokens) > 1_000_000:
        raise NativeEngineError("GGUF checkpoint has no usable embedded tokenizer vocabulary.")
    tokens = [token if isinstance(token, str) else str(token) for token in raw_tokens]
    if sum(len(token.encode("utf-8")) for token in tokens) > _MAX_TOKENIZER_BYTES:
        raise NativeEngineError("Embedded GGUF tokenizer vocabulary exceeds the 64 MiB limit.")
    if len(set(tokens)) != len(tokens):
        raise NativeEngineError("Embedded GGUF tokenizer has duplicate token strings; provide tokenizer.json.")
    raw_unk = _field(reader, "tokenizer.ggml.unknown_token_id", -1)
    unk_id = int(raw_unk) if isinstance(raw_unk, (int, np.integer)) and not isinstance(raw_unk, bool) and 0 <= int(raw_unk) < len(tokens) else None
    unk_token = tokens[unk_id] if unk_id is not None else None
    model_name = model_name.lower() if isinstance(model_name, str) else ""
    pre_name = pre_name.lower() if isinstance(pre_name, str) else ""
    if model_name in {"gpt2", "bpe"} or pre_name in {"llama-bpe", "gpt-2", "qwen2"}:
        if pre_name not in {"", "llama-bpe", "gpt-2", "gpt2", "qwen2"}:
            raise NativeEngineError(f"GGUF BPE pre-tokenizer {pre_name!r} is not supported; provide tokenizer.json.")
        vocab = {token: index for index, token in enumerate(tokens)}
        raw_merges = _field(reader, "tokenizer.ggml.merges", [])
        if not isinstance(raw_merges, list):
            raise NativeEngineError("GGUF BPE tokenizer merges must be a string list.")
        merges: list[tuple[str, str]] = []
        for merge in raw_merges:
            if not isinstance(merge, str):
                raise NativeEngineError("GGUF BPE tokenizer contains a malformed merge entry.")
            pair = merge.split(" ", 1)
            if len(pair) != 2:
                raise NativeEngineError("GGUF BPE tokenizer contains a malformed merge entry.")
            merges.append((pair[0], pair[1]))
        try:
            bpe = models.BPE(vocab=vocab, merges=merges, unk_token=unk_token, byte_fallback=True)
        except TypeError:
            bpe = models.BPE(vocab=vocab, merges=merges, unk_token=unk_token)
        tokenizer = Tokenizer(bpe)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
    elif model_name in {"llama", "unigram", "spm", "sentencepiece"}:
        if _field(reader, "tokenizer.ggml.precompiled_charsmap"):
            raise NativeEngineError(
                "This GGUF SentencePiece tokenizer has a precompiled normalizer that AXIOM cannot reconstruct exactly; place the matching tokenizer.json next to the GGUF file."
            )
        scores = _field(reader, "tokenizer.ggml.scores", [])
        if not isinstance(scores, list) or (scores and len(scores) != len(tokens)):
            raise NativeEngineError("GGUF Unigram tokenizer scores do not match its vocabulary.")
        scores = [float(value) for value in scores] if scores else [0.0] * len(tokens)
        tokenizer = Tokenizer(models.Unigram(list(zip(tokens, scores)), unk_id=unk_id, byte_fallback=True))
        try:
            tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
            tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
        except TypeError:
            tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True)
            tokenizer.decoder = decoders.Metaspace(replacement="▁", add_prefix_space=True)
        if _field(reader, "tokenizer.ggml.normalizer.nfc", False) is True:
            tokenizer.normalizer = normalizers.NFC()
    else:
        raise NativeEngineError(f"Embedded GGUF tokenizer model {model_name!r} is not supported; provide tokenizer.json.")

    token_types = _field(reader, "tokenizer.ggml.token_type", [])
    if isinstance(token_types, list) and len(token_types) == len(tokens):
        specials = [
            AddedToken(token, special=True)
            for index, (token, token_type) in enumerate(zip(tokens, token_types))
            if int(token_type) != 1 and tokenizer.token_to_id(token) == index
        ]
        if specials:
            tokenizer.add_special_tokens(specials)
    for index, token in enumerate(tokens):
        if tokenizer.token_to_id(token) != index:
            raise NativeEngineError("Embedded GGUF tokenizer cannot preserve token IDs; provide tokenizer.json.")
    return tokenizer, "tokenizer.ggml.tokens"


def _compute_dtype(torch_module: Any, device: Any, precision: str) -> Any:
    if precision not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("Precision must be auto, fp32, fp16, or bf16.")
    device_type = str(getattr(device, "type", device)).split(":", 1)[0]
    if precision == "auto":
        dtype = torch_module.float32 if device_type == "cpu" else getattr(torch_module, "float16", torch_module.float32)
    else:
        dtype = {"fp32": getattr(torch_module, "float32", None),
                 "fp16": getattr(torch_module, "float16", None),
                 "bf16": getattr(torch_module, "bfloat16", None)}[precision]
    if dtype is None:
        raise ValueError(f"Precision {precision!r} is not available in the installed PyTorch runtime.")
    if precision == "bf16" and device_type == "mps":
        raise ValueError("BF16 compute precision is not supported by the AXIOM MPS backend contract.")
    if precision == "bf16":
        backend = getattr(torch_module, device_type, None)
        probe = getattr(backend, "is_bf16_supported", None)
        if device_type in {"cuda", "xpu"} and callable(probe):
            try:
                supported = probe()
            except TypeError:
                supported = probe(device=device)
            except Exception:
                supported = None
            if supported is False:
                raise ValueError(f"BF16 compute precision is not supported by the selected {device_type.upper()} device.")
    return dtype


def load_native_gguf_runtime(
    model_dir: Path, torch_module: Any, device: Any, *,
    adapter_dir: Path | None = None, trusted_roots: tuple[Path, ...] | None = None,
    memory_mode: str = "full", precision: str = "auto",
) -> tuple[Any, Any, dict[str, Any]]:
    """Build the AXIOM-owned runtime from a trusted local GGUF checkpoint."""
    from axiom.core.native_engine import NativeEngineError, NativeLlamaRuntime

    if memory_mode not in {"full", "layer_streaming", "disk_streaming"}:
        raise ValueError("Native memory mode must be full, layer_streaming, or disk_streaming.")
    roots = tuple(root.resolve() for root in (trusted_roots or (model_dir.resolve(),)))
    paths = _find_files(model_dir, roots)
    reader, dequantize, quant_types = _open_reader(paths)
    _config_values, mapped, meta = _configuration(reader)
    config = meta["native_config"]
    tokenizer, tokenizer_source = _load_tokenizer(reader, model_dir, roots)
    if tokenizer.get_vocab_size(with_added_tokens=True) > config.vocab_size:
        raise NativeEngineError("GGUF tokenizer vocabulary is larger than the model embedding table.")

    dtype = _compute_dtype(torch_module, device, precision)
    shapes, expected = meta["weight_shapes"], meta["expected_shapes"]
    qtraits = getattr(__import__("gguf.quants", fromlist=["_type_traits"]), "_type_traits", {})
    supported = {quant_types.F32, quant_types.F16, *qtraits.keys()}
    for name, tensor in mapped.items():
        if tensor.tensor_type not in supported:
            raise NativeEngineError(f"GGUF tensor {tensor.name!r} uses unsupported type {tensor.tensor_type!s}.")
        if name in expected and shapes[name] != expected[name]:
            raise NativeEngineError(f"GGUF tensor {tensor.name!r} has an invalid native shape.")

    def make_loader(tensor: Any, name: str) -> Callable[[], Any]:
        shape = shapes[name]
        def load() -> Any:
            try:
                array = dequantize(tensor.data, tensor.tensor_type)
                if tuple(array.shape) != shape:
                    raise ValueError("dequantized shape mismatch")
                owned = np.array(array, dtype=np.float32, order="C", copy=True)
                return torch_module.from_numpy(owned).to(dtype=dtype)
            except Exception:
                raise NativeEngineError(f"GGUF tensor {tensor.name!r} could not be dequantized safely by AXIOM Native Engine.") from None
        # Preserve the source byte count for accurate read telemetry on streamed layers.
        load._axiom_source_bytes = int(getattr(tensor, "n_bytes", 0))
        return load

    weights: dict[str, Any] = {}
    layer_loaders: dict[str, Callable[[], Any]] = {}
    for name, tensor in mapped.items():
        if name.startswith("model.layers.") and memory_mode in {"layer_streaming", "disk_streaming"}:
            layer_loaders[name] = make_loader(tensor, name)
        else:
            weights[name] = make_loader(tensor, name)().to(device=device, dtype=dtype)
    if "lm_head.weight" not in weights:
        if not config.tie_word_embeddings:
            raise NativeEngineError("GGUF checkpoint has no output.weight and does not declare tied embeddings.")
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        shapes["lm_head.weight"] = tuple(weights["model.embed_tokens.weight"].shape)

    adapters: dict[str, dict[str, Any]] = {}
    scaling: dict[str, float] = {}
    if adapter_dir is not None:
        adapters, scaling = NativeLlamaRuntime._load_lora_adapter(adapter_dir, config, torch_module, device, dtype, roots)
        for module_path, pair in adapters.items():
            base_shape = shapes.get(module_path + ".weight")
            if base_shape is None or pair["A"].shape[1] != base_shape[1] or pair["B"].shape != (base_shape[0], pair["A"].shape[0]):
                raise NativeEngineError(f"LoRA weight shapes do not match base module {module_path!r}.")

    runtime = NativeLlamaRuntime(
        config, weights, torch_module, device, adapter=adapters, adapter_scaling=scaling,
        memory_mode=memory_mode, weight_shapes=shapes, layer_weight_loaders=layer_loaders,
    )
    runtime._gguf_reader = reader
    tensor_types = sorted({getattr(item.tensor_type, "name", str(item.tensor_type)) for item in reader.tensors})
    return runtime, tokenizer, {
        "model_type": config.model_type, "architecture": meta["architecture"],
        "model_directory_name": model_dir.name, "model_directory_path": str(model_dir.resolve()),
        "adapter_directory_name": adapter_dir.name if adapter_dir else None,
        "add_bos_token": bool(_field(reader, "tokenizer.ggml.add_bos_token", config.bos_token_id is not None)),
        "checkpoint_format": "gguf", "execution_backend": "pytorch-native-axiom",
        "transformers_generation_used": False, "ollama_used": False,
        "dtype": str(runtime.dtype),
        "precision": {str(getattr(torch_module, "float32", "float32")): "fp32",
                      str(getattr(torch_module, "float16", "float16")): "fp16",
                      str(getattr(torch_module, "bfloat16", "bfloat16")): "bf16"}.get(str(runtime.dtype), str(runtime.dtype)),
        "precision_requested": precision, "device": str(device), "memory_mode": runtime.memory_mode,
        "gguf_tensor_types": tensor_types,
        "gguf_paths": [str(path) for path in paths],
        "gguf_shard_count": len(paths),
        "gguf_low_bit_kernels_used": False,
        "gguf_tensors_dequantized_to": "floating-point compute dtype", "tokenizer_source": tokenizer_source,
        "scope_note": (
            "GGUF executes through AXIOM's own transformer/decode path; tensors are dequantized on load or per-layer. "
            "Dedicated low-bit inference kernels are not implemented."
        ),
    }
