"""AXIOM-native Llama/Mistral/Qwen2/Qwen3 inference.

AXIOM implements transformer forward operations, RMSNorm, RoPE, grouped-query
attention, KV caching, LoRA projection deltas, autoregressive decoding, and
full/in-RAM/disk-resident weight loading strategies. PyTorch supplies tensor
kernels; Transformers model.generate and Ollama are not used here. The native
loader accepts supported architectures with tokenizer.json and floating-point
safetensors. Unsupported formats/configurations are explicitly rejected.
"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import json, math, re, threading, time
from pathlib import Path
from typing import Any, Iterator, Mapping

_MAX_CONFIG_BYTES = 1_048_576
_MAX_TOKENIZER_BYTES = 64 * 1024 * 1024
_MAX_INDEX_BYTES = 64 * 1024 * 1024
_MAX_WEIGHT_SHARDS = 512
_SUPPORTED_MODEL_TYPES = {"llama", "mistral", "qwen2", "qwen3"}

class NativeEngineError(ValueError):
    """An unsupported, unsafe, or invalid native-engine checkpoint."""

class NativeGenerationCancelled(RuntimeError):
    """Internal cooperative cancellation signal for native inference."""

class NativeDependencyError(RuntimeError):
    """A required runtime dependency for the native engine is unavailable."""


def _apply_top_k_filter(scores: Any, top_k: int, torch_module: Any) -> Any:
    """Keep at most exactly k logits per row, including when boundary scores tie."""
    if top_k <= 0 or top_k >= scores.shape[-1]:
        return scores
    indices = torch_module.topk(scores, top_k, dim=-1).indices
    selected = torch_module.zeros_like(scores, dtype=torch_module.bool)
    selected.scatter_(-1, indices, True)
    return scores.masked_fill(~selected, float("-inf"))


def _apply_min_p_filter(scores: Any, min_p: float) -> Any:
    """Discard tokens whose score is too far below the highest logit."""
    if min_p <= 0:
        return scores
    threshold = scores.max(dim=-1, keepdim=True).values + math.log(min_p)
    return scores.masked_fill(scores < threshold, float("-inf"))


def _sanitize_sampling_scores(scores: Any, torch_module: Any) -> Any:
    """Make decoder logits safe for argmax/softmax without inventing a fallback token."""
    if bool(torch_module.isfinite(scores).all().item()):
        return scores

    positive_infinity = torch_module.isposinf(scores)
    if bool(positive_infinity.any().item()):
        # An overflowed +inf logit dominates all ordinary finite logits. Keep
        # only those tokens and assign equal finite scores among them.
        return torch_module.where(
            positive_infinity,
            torch_module.zeros_like(scores),
            torch_module.full_like(scores, float("-inf")),
        )

    finite = torch_module.isfinite(scores)
    if not bool(finite.any().item()):
        raise NativeEngineError(
            "Native decoder produced no usable token logits; check checkpoint weights, device, and precision."
        )
    # NaN logits are unusable; -inf is the standard mask value for samplers.
    return scores.masked_fill(torch_module.isnan(scores), float("-inf"))

@dataclass(frozen=True)
class NativeLlamaConfig:
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    bos_token_id: int | None
    eos_token_ids: tuple[int, ...]
    tie_word_embeddings: bool
    sliding_window: int | None
    qk_norm: bool
    rope_scaling_type: str = "default"
    rope_scaling_factor: float = 1.0
    rope_original_max_position_embeddings: int | None = None
    rope_low_freq_factor: float | None = None
    rope_high_freq_factor: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NativeLlamaConfig":
        model_type = value.get("model_type")
        if model_type not in _SUPPORTED_MODEL_TYPES:
            raise NativeEngineError("AXIOM Native Engine supports llama, mistral, qwen2, and qwen3 model_type only; there is no Transformers/Ollama fallback.")
        expected_architecture = {
            "llama": "LlamaForCausalLM",
            "mistral": "MistralForCausalLM",
            "qwen2": "Qwen2ForCausalLM",
            "qwen3": "Qwen3ForCausalLM",
        }[model_type]
        architectures = value.get("architectures")
        if architectures is not None and (
            not isinstance(architectures, list)
            or any(not isinstance(name, str) for name in architectures)
            or (architectures and expected_architecture not in architectures)
        ):
            raise NativeEngineError(
                "Checkpoint architecture does not match the supported causal-LM class for its model_type."
            )
        def config_integer(name: str, default: Any = None) -> int:
            raw = value[name] if name in value else default
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise TypeError
            return raw

        def config_number(name: str, default: Any) -> float:
            raw = value.get(name, default)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError
            return float(raw)

        try:
            hidden = config_integer("hidden_size")
            intermediate = config_integer("intermediate_size")
            layers = config_integer("num_hidden_layers")
            heads = config_integer("num_attention_heads")
            kv_heads = config_integer("num_key_value_heads", heads)
            vocab = config_integer("vocab_size")
            max_positions = config_integer("max_position_embeddings", 4096)
            rms_eps = config_number("rms_norm_eps", 1e-6)
            theta = config_number("rope_theta", 10000.0)
        except (KeyError, TypeError, ValueError, OverflowError):
            raise NativeEngineError("Local config.json is missing valid native Llama model dimensions.") from None
        if not (1 <= hidden <= 32768 and 1 <= intermediate <= 131072):
            raise NativeEngineError("Checkpoint hidden/intermediate dimensions exceed native safety limits.")
        if not (1 <= layers <= 256 and 1 <= heads <= 256):
            raise NativeEngineError("Checkpoint layer/head counts exceed native safety limits.")
        if hidden % heads or (hidden // heads) % 2:
            raise NativeEngineError("hidden_size must be divisible into even-sized attention heads.")
        configured_head_dim = value.get("head_dim")
        if configured_head_dim is not None and (
            isinstance(configured_head_dim, bool)
            or not isinstance(configured_head_dim, int)
            or configured_head_dim != hidden // heads
        ):
            raise NativeEngineError("Custom head_dim values are not supported unless they equal hidden_size / num_attention_heads.")
        if not (1 <= kv_heads <= heads) or heads % kv_heads:
            raise NativeEngineError("num_key_value_heads must divide num_attention_heads.")
        if not (1 <= vocab <= 1_000_000 and 1 <= max_positions <= 1_000_000):
            raise NativeEngineError("Checkpoint vocabulary/context dimensions exceed native safety limits.")
        if not math.isfinite(rms_eps) or rms_eps <= 0 or not math.isfinite(theta) or theta <= 0:
            raise NativeEngineError("Checkpoint normalization/RoPE settings are invalid.")
        rope = value.get("rope_scaling")
        rope_type = "default"
        rope_factor = 1.0
        rope_original: int | None = None
        rope_low: float | None = None
        rope_high: float | None = None
        if rope is not None:
            if not isinstance(rope, dict):
                raise NativeEngineError("This RoPE scaling configuration is not implemented by AXIOM Native Engine.")
            rope_type = rope.get("rope_type", rope.get("type", "default"))
            raw_factor = rope.get("factor", 1.0)
            if (
                isinstance(raw_factor, bool)
                or not isinstance(raw_factor, (int, float))
                or not math.isfinite(float(raw_factor))
            ):
                raise NativeEngineError("RoPE scaling factor must be a finite number.")
            rope_factor = float(raw_factor)
            if rope_type == "default":
                if rope_factor != 1.0:
                    raise NativeEngineError("Default RoPE scaling requires factor 1.")
            elif rope_type == "linear":
                if not 1.0 <= rope_factor <= 128.0:
                    raise NativeEngineError("Linear RoPE scaling factor must be between 1 and 128.")
            elif rope_type == "llama3":
                if not 1.0 <= rope_factor <= 128.0:
                    raise NativeEngineError("Llama 3 RoPE scaling factor must be between 1 and 128.")
                raw_original = rope.get("original_max_position_embeddings")
                raw_low, raw_high = rope.get("low_freq_factor"), rope.get("high_freq_factor")
                if (
                    isinstance(raw_original, bool)
                    or not isinstance(raw_original, int)
                    or not 1 <= raw_original <= max_positions
                    or isinstance(raw_low, bool)
                    or not isinstance(raw_low, (int, float))
                    or not math.isfinite(float(raw_low))
                    or isinstance(raw_high, bool)
                    or not isinstance(raw_high, (int, float))
                    or not math.isfinite(float(raw_high))
                    or float(raw_low) <= 0
                    or float(raw_high) <= float(raw_low)
                ):
                    raise NativeEngineError("Llama 3 RoPE scaling requires valid original context and low/high frequency factors.")
                rope_original = raw_original
                rope_low, rope_high = float(raw_low), float(raw_high)
            else:
                raise NativeEngineError("This RoPE scaling configuration is not implemented by AXIOM Native Engine.")
        sliding = value.get("sliding_window") if model_type == "mistral" else None
        if sliding is not None and (isinstance(sliding, bool) or not isinstance(sliding, int) or sliding < 1):
            raise NativeEngineError("Mistral sliding_window must be a positive integer.")
        bos0, eos0 = value.get("bos_token_id"), value.get("eos_token_id")
        if bos0 is not None and (isinstance(bos0, bool) or not isinstance(bos0, int)):
            raise NativeEngineError("Configured BOS token ID must be an integer or null.")
        if eos0 is None:
            eos: tuple[int, ...] = ()
        elif isinstance(eos0, int) and not isinstance(eos0, bool):
            eos = (eos0,)
        elif isinstance(eos0, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in eos0):
            eos = tuple(eos0)
        else:
            raise NativeEngineError("Configured EOS token IDs must be integers or a list of integers.")
        bos = bos0
        if any(x < 0 or x >= vocab for x in (*eos, *((bos,) if bos is not None else ()))):
            raise NativeEngineError("Configured BOS/EOS token ID is outside vocab_size.")
        tied = value.get("tie_word_embeddings", False)
        if not isinstance(tied, bool):
            raise NativeEngineError("tie_word_embeddings must be a boolean.")
        return cls(
            model_type, hidden, intermediate, layers, heads, kv_heads, vocab,
            max_positions, rms_eps, theta, bos, eos, tied, sliding, model_type == "qwen3",
            rope_scaling_type=rope_type,
            rope_scaling_factor=rope_factor,
            rope_original_max_position_embeddings=rope_original,
            rope_low_freq_factor=rope_low,
            rope_high_freq_factor=rope_high,
        )

def _within_trusted_roots(path: Path, trusted_roots: tuple[Path, ...]) -> bool:
    for root in trusted_roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _resolve_trusted_file(path: Path, trusted_roots: tuple[Path, ...], label: str) -> Path:
    """Resolve a model asset and allow symlinks only when the final target remains trusted."""
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise NativeEngineError(f"{label} must be an existing local file.") from None
    if not resolved.is_file() or not _within_trusted_roots(resolved, trusted_roots):
        raise NativeEngineError(f"{label} resolves outside the configured trusted model roots.")
    return resolved

def _read_json_object(
    path: Path,
    max_bytes: int,
    label: str,
    trusted_roots: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    roots = trusted_roots or (path.parent.resolve(),)
    safe_path = _resolve_trusted_file(path, roots, label)
    try:
        if safe_path.stat().st_size > max_bytes:
            raise NativeEngineError(f"{label} exceeds its size limit.")
        payload = json.loads(safe_path.read_text(encoding="utf-8"))
    except NativeEngineError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise NativeEngineError(f"{label} could not be read as valid UTF-8 JSON.") from None
    if not isinstance(payload, dict):
        raise NativeEngineError(f"{label} must contain a JSON object.")
    return payload

def _resolve_safetensor_files(
    model_dir: Path,
    trusted_roots: tuple[Path, ...] | None = None,
) -> tuple[list[Path], dict[str, str]]:
    trusted_roots = trusted_roots or (model_dir.resolve(),)
    index_path, single_path = model_dir / "model.safetensors.index.json", model_dir / "model.safetensors"
    if index_path.is_file():
        index = _read_json_object(index_path, _MAX_INDEX_BYTES, "Safetensors shard index", trusted_roots)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise NativeEngineError("Safetensors index must contain a non-empty weight_map object.")
        shards: set[str] = set()
        normalized: dict[str, str] = {}
        for tensor_name, filename in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name or not isinstance(filename, str):
                raise NativeEngineError("Safetensors index contains an invalid tensor mapping.")
            shard = Path(filename)
            if shard.name != filename or "/" in filename or "\\" in filename or filename in {"", ".", ".."} or shard.suffix != ".safetensors":
                raise NativeEngineError("Safetensors index contains an unsafe or unsupported shard path.")
            shards.add(filename)
            normalized[tensor_name] = filename
        if len(shards) > _MAX_WEIGHT_SHARDS:
            raise NativeEngineError("Checkpoint has more than 512 safetensors shards.")
        files = [model_dir / name for name in sorted(shards)]
        for path in files:
            _resolve_trusted_file(path, trusted_roots, "Safetensors shard")
        return files, normalized
    if single_path.is_file():
        _resolve_trusted_file(single_path, trusted_roots, "Safetensors checkpoint")
        return [single_path], {}
    if (model_dir / "pytorch_model.bin").exists() or any(model_dir.glob("pytorch_model-*.bin")):
        raise NativeEngineError("Pickle-based .bin weights are unsupported; use safetensors. AXIOM will not load arbitrary pickle payloads.")
    raise NativeEngineError("No model.safetensors or model.safetensors.index.json was found in the model directory.")

def _safetensor_keys(files: list[Path], trusted_roots: tuple[Path, ...]) -> dict[str, str]:
    try:
        from safetensors import safe_open
    except (ImportError, OSError):
        raise NativeDependencyError("AXIOM Native Engine requires safetensors. Install the native-inference extra.") from None
    result: dict[str, str] = {}
    for path in files:
        try:
            with safe_open(str(_resolve_trusted_file(path, trusted_roots, "Safetensors shard")), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
        except Exception:
            raise NativeEngineError(f"Safetensors shard {path.name!r} could not be inspected safely.") from None
        for key in keys:
            if key in result:
                raise NativeEngineError(f"Tensor {key!r} is duplicated across safetensors shards.")
            result[key] = path.name
    return result


def inspect_native_adapter_compatibility(
    model_dir: Path,
    adapter_dir: Path,
    *,
    trusted_roots: tuple[Path, ...],
) -> dict[str, Any]:
    """Preflight native LoRA structure without loading the full base-model weights."""
    issues: list[str] = []
    config: NativeLlamaConfig | None = None
    model_type: str | None = None
    architecture: str | None = None
    base_weight_names: set[str] = set()
    layer_count = 0
    target_set: set[str] = set()
    rank: int | None = None
    alpha: float | None = None
    adapter_tensor_count = 0
    pair_shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    roots = tuple(root.resolve() for root in trusted_roots)

    try:
        mapping = _read_json_object(model_dir / "config.json", _MAX_CONFIG_BYTES, "Model config.json", roots)
        config = NativeLlamaConfig.from_mapping(mapping)
        model_type = config.model_type
        architecture = {"llama": "LlamaForCausalLM", "mistral": "MistralForCausalLM", "qwen2": "Qwen2ForCausalLM", "qwen3": "Qwen3ForCausalLM"}[model_type]
        layer_count = config.num_hidden_layers
        tokenizer_path = _resolve_trusted_file(model_dir / "tokenizer.json", roots, "Tokenizer file")
        if tokenizer_path.stat().st_size > _MAX_TOKENIZER_BYTES:
            issues.append("The local tokenizer.json exceeds the 64 MiB native-engine limit.")
        base_files, _ = _resolve_safetensor_files(model_dir, roots)
        base_weight_names = set(_safetensor_keys(base_files, roots))
        missing = [
            name for name in NativeLlamaRuntime._required_weight_names(config)
            if name not in base_weight_names
        ]
        if missing:
            issues.append("Base checkpoint is missing required native weights: " + ", ".join(missing[:5]) + ".")
    except (NativeEngineError, NativeDependencyError, OSError, ValueError) as exc:
        issues.append("Base model: " + str(exc))

    try:
        adapter_config = _read_json_object(
            adapter_dir / "adapter_config.json", _MAX_CONFIG_BYTES, "LoRA adapter config", roots
        )
        for flag in ("use_dora", "use_qalora", "use_rslora"):
            if adapter_config.get(flag) is True:
                issues.append(f"LoRA option {flag} is not supported by AXIOM Native Engine.")
        if adapter_config.get("bias", "none") != "none":
            issues.append("Bias-enabled LoRA adapters are unsupported.")
        if (
            adapter_config.get("modules_to_save")
            or adapter_config.get("rank_pattern")
            or adapter_config.get("alpha_pattern")
        ):
            issues.append("Per-module rank/alpha patterns and modules_to_save are unsupported.")
        if adapter_config.get("fan_in_fan_out") is True:
            issues.append("fan_in_fan_out adapters are unsupported.")
        if adapter_config.get("task_type") not in (None, "CAUSAL_LM"):
            issues.append("Only CAUSAL_LM LoRA adapters are supported.")
        raw_targets = adapter_config.get("target_modules")
        if isinstance(raw_targets, str):
            target_set = {raw_targets}
        elif isinstance(raw_targets, list) and all(isinstance(item, str) and item for item in raw_targets):
            target_set = set(raw_targets)
        if not target_set:
            issues.append("LoRA adapter must declare target_modules as non-empty strings.")
        raw_rank, raw_alpha = adapter_config.get("r"), adapter_config.get("lora_alpha")
        if isinstance(raw_rank, int) and not isinstance(raw_rank, bool) and 1 <= raw_rank <= 4096:
            rank = raw_rank
        else:
            issues.append("LoRA adapter rank r must be an integer from 1 through 4096.")
        if (
            isinstance(raw_alpha, (int, float))
            and not isinstance(raw_alpha, bool)
            and math.isfinite(float(raw_alpha))
            and raw_alpha > 0
        ):
            alpha = float(raw_alpha)
        else:
            issues.append("LoRA adapter lora_alpha must be a finite positive number.")
    except (NativeEngineError, NativeDependencyError, OSError, ValueError) as exc:
        issues.append("LoRA adapter config: " + str(exc))

    try:
        adapter_file = _resolve_trusted_file(
            adapter_dir / "adapter_model.safetensors", roots, "LoRA safetensors checkpoint"
        )
        from safetensors import safe_open
        with safe_open(str(adapter_file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                match = re.fullmatch(r"(.+)\.lora_([AB])(?:\.default)?\.weight", key)
                if not match:
                    issues.append(f"Unsupported LoRA tensor key {key!r}.")
                    continue
                module_path, side = match.group(1), match.group(2)
                if module_path.startswith("base_model.model."):
                    module_path = module_path[len("base_model.model."):]
                elif module_path.startswith("base_model."):
                    module_path = module_path[len("base_model."):]
                if module_path.split(".")[-1] not in target_set:
                    issues.append(f"LoRA tensor {key!r} targets a module not declared in target_modules.")
                module_match = re.fullmatch(
                    r"model\.layers\.(\d+)\.(self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))",
                    module_path,
                )
                if config is None or module_match is None:
                    issues.append(f"LoRA tensor {key!r} does not map to a supported native projection.")
                    continue
                layer_index = int(module_match.group(1))
                if not 0 <= layer_index < config.num_hidden_layers:
                    issues.append(f"LoRA tensor {key!r} references a layer outside the base model.")
                    continue
                if module_path + ".weight" not in base_weight_names:
                    issues.append(f"LoRA target {module_path!r} has no matching base projection tensor.")
                    continue
                projection = module_match.group(2).split(".")[-1]
                head_dim = config.hidden_size // config.num_attention_heads
                if projection == "q_proj":
                    output_dim, input_dim = config.num_attention_heads * head_dim, config.hidden_size
                elif projection in {"k_proj", "v_proj"}:
                    output_dim, input_dim = config.num_key_value_heads * head_dim, config.hidden_size
                elif projection == "o_proj":
                    output_dim, input_dim = config.hidden_size, config.hidden_size
                elif projection in {"gate_proj", "up_proj"}:
                    output_dim, input_dim = config.intermediate_size, config.hidden_size
                else:
                    output_dim, input_dim = config.hidden_size, config.intermediate_size
                shape = tuple(int(dimension) for dimension in handle.get_slice(key).get_shape())
                pair_shapes.setdefault(module_path, {})[side] = shape
                if side == "A" and (rank is None or shape != (rank, input_dim)):
                    issues.append(f"LoRA A tensor shape {shape} for {module_path!r} must be ({rank}, {input_dim}).")
                if side == "B" and (rank is None or shape != (output_dim, rank)):
                    issues.append(f"LoRA B tensor shape {shape} for {module_path!r} must be ({output_dim}, {rank}).")
                adapter_tensor_count += 1
    except (NativeEngineError, NativeDependencyError, OSError, ValueError, RuntimeError) as exc:
        issues.append("LoRA safetensors inspection failed: " + str(exc))

    if not pair_shapes:
        issues.append("No supported LoRA A/B projection pairs were found.")
    for module_path, sides in pair_shapes.items():
        if "A" not in sides or "B" not in sides:
            issues.append(f"LoRA module {module_path!r} must contain both A and B tensors.")

    try:
        import torch
    except (ImportError, OSError):
        torch = None

    cpu_available = torch is not None
    cuda_available = False
    rocm_available = False
    mps_available = False
    xpu_available = False
    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())
            rocm_available = cuda_available and bool(getattr(torch.version, "hip", None))
            cuda_available = cuda_available and not rocm_available
        except (RuntimeError, AttributeError):
            cuda_available = False
            rocm_available = False
        try:
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            mps_available = bool(mps and mps.is_available())
        except (RuntimeError, AttributeError):
            mps_available = False
        try:
            xpu_backend = getattr(torch, "xpu", None)
            xpu_available = bool(xpu_backend and xpu_backend.is_available())
        except (RuntimeError, AttributeError):
            xpu_available = False

    structurally_compatible = not issues
    devices = []
    for name, available in (
        ("CPU", cpu_available),
        ("CUDA", cuda_available),
        ("ROCm", rocm_available),
        ("Metal/MPS", mps_available),
        ("Intel XPU", xpu_available),
    ):
        compatible = structurally_compatible and available
        if not available:
            reason = "PyTorch is not installed." if torch is None else f"{name} is not available in the current PyTorch environment."
        elif issues:
            reason = "Structural compatibility checks failed; see issues."
        else:
            reason = "Adapter metadata and tensor shapes match the native architecture. Real allocation and execution still need testing."
        devices.append({
            "name": name,
            "available": available,
            "compatible": compatible,
            "status": "compatible" if compatible else "incompatible" if available else "unavailable",
            "reason": reason,
        })

    return {
        "schema_version": 1,
        "base_model": {"name": model_dir.name, "model_type": model_type, "architecture": architecture},
        "adapter": {"name": adapter_dir.name, "rank": rank, "alpha": alpha, "target_modules": sorted(target_set)},
        "structurally_compatible": structurally_compatible,
        "compatible_on_any_device": any(item["compatible"] for item in devices),
        "adapter_tensor_count": adapter_tensor_count,
        "checked_layers": layer_count,
        "devices": devices,
        "issues": list(dict.fromkeys(issues)),
        "scope_note": "Metadata/tensor-shape preflight only. This does not load the full model, allocate VRAM, execute a forward pass, or guarantee device/driver execution.",
    }


class NativeLlamaRuntime:
    """AXIOM-owned implementation of Llama/Mistral/Qwen2/Qwen3 forward and decode steps."""

    def __init__(
        self,
        config: NativeLlamaConfig,
        weights: dict[str, Any],
        torch_module: Any,
        device: Any,
        *,
        adapter: dict[str, dict[str, Any]] | None = None,
        adapter_scaling: dict[str, float] | None = None,
        memory_mode: str = "full",
        layer_weight_sources: dict[str, Path] | None = None,
        weight_shapes: dict[str, tuple[int, ...]] | None = None,
    ) -> None:
        if memory_mode not in {"full", "layer_streaming", "disk_streaming"}:
            raise NativeEngineError("Native memory mode must be full, layer_streaming, or disk_streaming.")
        self.config, self.weights, self.torch, self.device = config, weights, torch_module, device
        self.adapter, self.adapter_scaling = adapter or {}, adapter_scaling or {}
        self.memory_mode = memory_mode
        self.layer_weight_sources = layer_weight_sources or {}
        self.layer_weight_names_by_index: dict[int, list[str]] = {}
        self.layer_weight_sources_by_index: dict[int, dict[str, Path]] = {}
        for tensor_name in set(weights) | set(self.layer_weight_sources):
            match = re.match(r"^model\.layers\.(\d+)\.", tensor_name)
            if match:
                self.layer_weight_names_by_index.setdefault(int(match.group(1)), []).append(tensor_name)
                source = self.layer_weight_sources.get(tensor_name)
                if source is not None:
                    self.layer_weight_sources_by_index.setdefault(int(match.group(1)), {})[tensor_name] = source
        for tensor_names in self.layer_weight_names_by_index.values():
            tensor_names.sort()
        self.weight_shapes = weight_shapes or {name: tuple(value.shape) for name, value in weights.items()}
        self.dtype = weights["model.embed_tokens.weight"].dtype
        self.name = "axiom-native-pytorch-causal-lm"
        self.last_generation: dict[str, Any] = {}
        self.live_generation: dict[str, Any] = {}
        self.last_attention_backend = "pytorch-scaled-dot-product-attention"
        self.memory_stats: dict[str, int] = {
            "layer_weight_tensors_staged": 0,
            "layer_weight_bytes_staged": 0,
            "layer_weight_bytes_returned_to_cpu": 0,
            "layer_weight_bytes_released": 0,
            "disk_layer_weight_loads": 0,
            "disk_layer_weight_bytes_read": 0,
            "kv_cache_layer_offloads": 0,
            "kv_cache_bytes_offloaded": 0,
        }

    @contextmanager
    def _layer_weights(self, layer_index: int):
        """Stage a CPU-resident or disk-resident transformer layer on the execution device."""
        if self.memory_mode not in {"layer_streaming", "disk_streaming"}:
            yield
            return
        if self.memory_mode == "disk_streaming":
            loaded: list[str] = []
            try:
                from safetensors import safe_open
                with ExitStack() as stack:
                    handles: dict[str, Any] = {}
                    for name, source in self.layer_weight_sources_by_index.get(layer_index, {}).items():
                        source_text = str(source)
                        if source_text not in handles:
                            handles[source_text] = stack.enter_context(
                                safe_open(source_text, framework="pt", device="cpu")
                            )
                        tensor = handles[source_text].get_tensor(name)
                        if tensor.dtype not in (self.torch.float16, self.torch.bfloat16, self.torch.float32):
                            raise NativeEngineError(f"Unsupported on-disk layer tensor dtype for {name!r}.")
                        byte_count = int(tensor.numel() * tensor.element_size())
                        active = tensor.to(device=self.device, dtype=self.dtype)
                        del tensor
                        self.weights[name] = active
                        loaded.append(name)
                        self.memory_stats["disk_layer_weight_loads"] += 1
                        self.memory_stats["disk_layer_weight_bytes_read"] += byte_count
                        self.memory_stats["layer_weight_bytes_staged"] += byte_count
                    yield
            finally:
                for name in loaded:
                    active_weight = self.weights.pop(name, None)
                    if active_weight is not None:
                        self.memory_stats["layer_weight_bytes_released"] += int(
                            active_weight.numel() * active_weight.element_size()
                        )
            return
        staged: dict[str, Any] = {}
        try:
            for name in self.layer_weight_names_by_index.get(layer_index, []):
                weight = self.weights.get(name)
                if weight is None or getattr(weight, "device", None) == self.device:
                    continue
                staged[name] = weight
                self.weights[name] = weight.to(device=self.device)
                self.memory_stats["layer_weight_tensors_staged"] += 1
                self.memory_stats["layer_weight_bytes_staged"] += int(weight.numel() * weight.element_size())
            yield
        finally:
            for name in staged:
                active_weight = self.weights[name]
                self.weights[name] = active_weight.to(device="cpu")
                self.memory_stats["layer_weight_bytes_returned_to_cpu"] += int(
                    self.weights[name].numel() * self.weights[name].element_size()
                )

    @staticmethod
    def _required_weight_names(c: NativeLlamaConfig) -> list[str]:
        names = ["model.embed_tokens.weight"]
        for i in range(c.num_hidden_layers):
            p = f"model.layers.{i}."
            names.extend(p + x for x in (
                "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
            ))
            if c.qk_norm:
                names.extend(p + x for x in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"))
        names.append("model.norm.weight")
        if not c.tie_word_embeddings:
            names.append("lm_head.weight")
        return names

    @classmethod
    def from_directory(
        cls,
        model_dir: Path,
        torch_module: Any,
        device: Any,
        *,
        adapter_dir: Path | None = None,
        trusted_roots: tuple[Path, ...] | None = None,
        memory_mode: str = "full",
        precision: str = "auto",
    ) -> "NativeLlamaRuntime":
        if memory_mode not in {"full", "layer_streaming", "disk_streaming"}:
            raise NativeEngineError("Native memory mode must be full, layer_streaming, or disk_streaming.")
        if precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise NativeEngineError("Precision must be auto, fp32, fp16, or bf16.")
        roots = trusted_roots or (model_dir.resolve(),)
        config = NativeLlamaConfig.from_mapping(_read_json_object(model_dir / "config.json", _MAX_CONFIG_BYTES, "Model config.json", roots))
        tokenizer_path = _resolve_trusted_file(model_dir / "tokenizer.json", roots, "Tokenizer file")
        if tokenizer_path.stat().st_size > _MAX_TOKENIZER_BYTES:
            raise NativeEngineError("AXIOM Native Engine requires a local tokenizer.json of at most 64 MiB.")
        files, indexed_map = _resolve_safetensor_files(model_dir, roots)
        discovered = _safetensor_keys(files, roots)
        weight_map = indexed_map or discovered
        if indexed_map and set(discovered) != set(indexed_map):
            raise NativeEngineError("Safetensors index and actual shard contents do not agree.")
        required = cls._required_weight_names(config)
        missing = [name for name in required if name not in weight_map]
        if missing:
            raise NativeEngineError("Checkpoint is missing native Llama weights: " + ", ".join(missing[:5]) + ".")
        try:
            from safetensors import safe_open
        except (ImportError, OSError):
            raise NativeDependencyError("AXIOM Native Engine requires safetensors; install the native-inference extra.") from None
        weights: dict[str, Any] = {}
        weight_shapes: dict[str, tuple[int, ...]] = {}
        layer_weight_sources: dict[str, Path] = {}
        file_paths = {p.name: _resolve_trusted_file(p, roots, "Safetensors shard") for p in files}
        try:
            with ExitStack() as stack:
                handles = {name: stack.enter_context(safe_open(str(path), framework="pt", device="cpu")) for name, path in file_paths.items()}
                embedding = handles[weight_map["model.embed_tokens.weight"]].get_tensor("model.embed_tokens.weight")
                if embedding.dtype not in (
                    torch_module.float16,
                    torch_module.bfloat16,
                    torch_module.float32,
                ):
                    raise NativeEngineError(
                        "Unsupported embedding weight dtype; native checkpoints require fp16, bf16, or fp32 weights."
                    )
                checkpoint_dtype = embedding.dtype
                dtype_map = {
                    "fp32": getattr(torch_module, "float32", None),
                    "fp16": getattr(torch_module, "float16", None),
                    "bf16": getattr(torch_module, "bfloat16", None),
                }
                dtype = checkpoint_dtype if precision == "auto" else dtype_map[precision]
                if dtype is None:
                    raise NativeEngineError(f"Precision {precision!r} is not available in the installed PyTorch runtime.")
                if precision != "auto" and precision == "bf16" and str(getattr(device, "type", device)).split(":", 1)[0] == "mps":
                    raise NativeEngineError("BF16 compute precision is not supported by the AXIOM MPS backend contract; choose auto, fp16, or fp32.")
                if precision == "bf16":
                    backend_name = str(getattr(device, "type", device)).split(":", 1)[0]
                    backend = getattr(torch_module, backend_name, None)
                    probe = getattr(backend, "is_bf16_supported", None)
                    if backend_name in {"cuda", "xpu"} and callable(probe):
                        try:
                            supported = probe()
                        except TypeError:
                            supported = probe(device=device)
                        except Exception:
                            supported = None
                        if supported is False:
                            raise NativeEngineError(
                                f"BF16 compute precision is not supported by the selected {backend_name.upper()} device; choose auto, fp16, or fp32."
                            )
                weight_shapes["model.embed_tokens.weight"] = tuple(int(dimension) for dimension in embedding.shape)
                weights["model.embed_tokens.weight"] = embedding.to(device=device, dtype=dtype)
                load_names = [name for name in required if name != "model.embed_tokens.weight"]
                if "lm_head.weight" in weight_map and "lm_head.weight" not in load_names:
                    load_names.append("lm_head.weight")
                if "lm_head.bias" in weight_map:
                    load_names.append("lm_head.bias")
                for i in range(config.num_hidden_layers):
                    p = f"model.layers.{i}."
                    for module in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
                        bias = p + module + ".bias"
                        if bias in weight_map:
                            load_names.append(bias)
                for name in load_names:
                    tensor = handles[weight_map[name]].get_tensor(name)
                    if tensor.dtype not in (torch_module.float16, torch_module.bfloat16, torch_module.float32):
                        raise NativeEngineError(f"Unsupported weight dtype for tensor {name!r}.")
                    weight_shapes[name] = tuple(int(dimension) for dimension in tensor.shape)
                    if memory_mode == "disk_streaming" and name.startswith("model.layers."):
                        layer_weight_sources[name] = file_paths[weight_map[name]]
                        del tensor
                        continue
                    load_device = "cpu" if memory_mode == "layer_streaming" and name.startswith("model.layers.") else device
                    weights[name] = tensor.to(device=load_device, dtype=dtype)
                    del tensor
        except NativeEngineError:
            raise
        except Exception as exc:
            message = str(exc).lower()
            if "out of memory" in message or "cannot allocate memory" in message or "alloc_failed" in message:
                raise NativeEngineError("Checkpoint loading ran out of memory on the selected device.") from None
            raise NativeEngineError("Checkpoint weights could not be loaded from local safetensors files.") from None
        hdim = config.hidden_size // config.num_attention_heads
        expected = {"model.embed_tokens.weight": (config.vocab_size, config.hidden_size), "model.norm.weight": (config.hidden_size,)}
        for i in range(config.num_hidden_layers):
            p = f"model.layers.{i}."
            expected.update({
                p + "self_attn.q_proj.weight": (config.num_attention_heads * hdim, config.hidden_size),
                p + "self_attn.k_proj.weight": (config.num_key_value_heads * hdim, config.hidden_size),
                p + "self_attn.v_proj.weight": (config.num_key_value_heads * hdim, config.hidden_size),
                p + "self_attn.o_proj.weight": (config.hidden_size, config.hidden_size),
                p + "mlp.gate_proj.weight": (config.intermediate_size, config.hidden_size),
                p + "mlp.up_proj.weight": (config.intermediate_size, config.hidden_size),
                p + "mlp.down_proj.weight": (config.hidden_size, config.intermediate_size),
                p + "input_layernorm.weight": (config.hidden_size,),
                p + "post_attention_layernorm.weight": (config.hidden_size,),
            })
            # Optional projection biases must match the output dimension. Some
            # supported checkpoints (for example Qwen2) legitimately use q/k/v
            # biases while others omit them, so validate only declared tensors.
            bias_output_dims = {
                "self_attn.q_proj": config.num_attention_heads * hdim,
                "self_attn.k_proj": config.num_key_value_heads * hdim,
                "self_attn.v_proj": config.num_key_value_heads * hdim,
                "self_attn.o_proj": config.hidden_size,
                "mlp.gate_proj": config.intermediate_size,
                "mlp.up_proj": config.intermediate_size,
                "mlp.down_proj": config.hidden_size,
            }
            for module, output_dim in bias_output_dims.items():
                bias_name = p + module + ".bias"
                if bias_name in weight_map:
                    expected[bias_name] = (output_dim,)
            if config.qk_norm:
                expected[p + "self_attn.q_norm.weight"] = (hdim,)
                expected[p + "self_attn.k_norm.weight"] = (hdim,)
        if "lm_head.weight" in weight_map:
            expected["lm_head.weight"] = (config.vocab_size, config.hidden_size)
        if "lm_head.bias" in weight_map:
            expected["lm_head.bias"] = (config.vocab_size,)
        for name, shape in expected.items():
            if name not in weight_shapes or weight_shapes[name] != shape:
                if name == "lm_head.weight" and config.tie_word_embeddings and name not in weight_shapes:
                    continue
                raise NativeEngineError(f"Checkpoint tensor {name!r} has a shape inconsistent with config.json.")
        if "lm_head.weight" not in weights:
            if not config.tie_word_embeddings:
                raise NativeEngineError("Checkpoint has no lm_head.weight and does not declare tied embeddings.")
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
            weight_shapes["lm_head.weight"] = tuple(weights["model.embed_tokens.weight"].shape)
        adapters: dict[str, dict[str, Any]] = {}
        scaling: dict[str, float] = {}
        if adapter_dir is not None:
            adapters, scaling = cls._load_lora_adapter(adapter_dir, config, torch_module, device, dtype, roots)
            for path, pair in adapters.items():
                base_shape = weight_shapes.get(path + ".weight")
                if base_shape is None or pair["A"].shape[1] != base_shape[1] or pair["B"].shape != (base_shape[0], pair["A"].shape[0]):
                    raise NativeEngineError(f"LoRA weight shapes do not match base module {path!r}.")
        return cls(
            config, weights, torch_module, device,
            adapter=adapters, adapter_scaling=scaling, memory_mode=memory_mode,
            layer_weight_sources=layer_weight_sources, weight_shapes=weight_shapes,
        )

    @classmethod
    def _load_lora_adapter(cls, adapter_dir: Path, config: NativeLlamaConfig, torch_module: Any, device: Any, dtype: Any, trusted_roots: tuple[Path, ...]) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
        c = _read_json_object(adapter_dir / "adapter_config.json", _MAX_CONFIG_BYTES, "LoRA adapter config", trusted_roots)
        for flag in ("use_dora", "use_qalora", "use_rslora"):
            if c.get(flag) is True:
                raise NativeEngineError(f"LoRA option {flag} is not supported by AXIOM Native Engine.")
        if c.get("bias", "none") != "none" or c.get("modules_to_save") or c.get("rank_pattern") or c.get("alpha_pattern") or c.get("fan_in_fan_out") is True:
            raise NativeEngineError("LoRA adapter uses bias, modules_to_save, per-module rank/alpha, or fan_in_fan_out which are not supported.")
        targets = c.get("target_modules")
        target_set = {targets} if isinstance(targets, str) else set(targets) if isinstance(targets, list) and all(isinstance(x, str) for x in targets) else set()
        rank, alpha = c.get("r"), c.get("lora_alpha")
        if not target_set or isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 4096:
            raise NativeEngineError("LoRA adapter target_modules or rank is invalid.")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or alpha <= 0:
            raise NativeEngineError("LoRA adapter lora_alpha is invalid.")
        if c.get("task_type") not in (None, "CAUSAL_LM"):
            raise NativeEngineError("Only CAUSAL_LM LoRA adapters are supported.")
        path = adapter_dir / "adapter_model.safetensors"
        if not path.is_file():
            raise NativeEngineError("LoRA adapter requires adapter_model.safetensors; pickle adapter files are unsupported.")
        safe_adapter_path = _resolve_trusted_file(path, trusted_roots, "LoRA safetensors checkpoint")
        try:
            from safetensors import safe_open
            with safe_open(str(safe_adapter_path), framework="pt", device="cpu") as handle:
                adapters: dict[str, dict[str, Any]] = {}
                for key in handle.keys():
                    match = re.fullmatch(r"(.+)\.lora_([AB])(?:\.default)?\.weight", key)
                    if not match:
                        raise NativeEngineError(f"Unsupported LoRA tensor key {key!r}.")
                    module_path, side = match.group(1), match.group(2)
                    if module_path.startswith("base_model.model."):
                        module_path = module_path[len("base_model.model."):]
                    elif module_path.startswith("base_model."):
                        module_path = module_path[len("base_model."):]
                    if module_path.split(".")[-1] not in target_set:
                        raise NativeEngineError("LoRA weight targets a module not listed in adapter_config.json.")
                    adapters.setdefault(module_path, {})[side] = handle.get_tensor(key).to(device=device, dtype=dtype)
        except NativeEngineError:
            raise
        except Exception as exc:
            message = str(exc).lower()
            if "out of memory" in message or "cannot allocate memory" in message or "alloc_failed" in message:
                raise NativeEngineError("Loading LoRA adapter ran out of memory on the selected device.") from None
            raise NativeEngineError("LoRA weights could not be read from local safetensors.") from None
        scaling: dict[str, float] = {}
        for module_path, pair in adapters.items():
            if "A" not in pair or "B" not in pair:
                raise NativeEngineError("Each LoRA module requires both lora_A and lora_B.")
            a, b = pair["A"], pair["B"]
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
                raise NativeEngineError(f"LoRA rank is inconsistent for {module_path!r}.")
            scaling[module_path] = float(alpha) / rank
        return adapters, scaling

    def _linear(self, hidden: Any, module: str) -> Any:
        functional = self.torch.nn.functional
        weight = self.weights.get(module + ".weight")
        if weight is None:
            raise NativeEngineError(f"Missing projection weights for {module!r}.")
        result = functional.linear(hidden, weight, self.weights.get(module + ".bias"))
        adapter = self.adapter.get(module)
        if adapter:
            result = result + functional.linear(functional.linear(hidden, adapter["A"]), adapter["B"]) * self.adapter_scaling[module]
        return result

    def _attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        blocked_mask: Any | None,
        *,
        is_causal: bool = False,
    ) -> Any:
        """Run explicit masks with native GQA when available and a compatible fallback."""
        torch = self.torch
        functional = torch.nn.functional
        scaled_dot_product = getattr(functional, "scaled_dot_product_attention", None)
        attention_mask = None if blocked_mask is None else ~blocked_mask
        query_heads, kv_heads = query.shape[1], key.shape[1]
        if kv_heads < 1 or query_heads % kv_heads:
            raise NativeEngineError("Grouped-query attention head counts are inconsistent.")
        grouped_query = query_heads != kv_heads

        if callable(scaled_dot_product):
            # A boolean SDPA mask uses True for visible positions. Cache-aware
            # sliding/chunked paths pass an explicit inverted mask; ordinary prefill
            # can use is_causal=True with no mask to avoid a dense quadratic allocation.
            if grouped_query:
                try:
                    # Native GQA avoids allocating repeated K/V heads for common
                    # Llama, Mistral and Qwen checkpoints.
                    result = scaled_dot_product(
                        query,
                        key,
                        value,
                        attn_mask=attention_mask,
                        dropout_p=0.0,
                        is_causal=is_causal,
                        enable_gqa=True,
                    )
                except (TypeError, NotImplementedError, RuntimeError) as exc:
                    # Some tensor backends expose enable_gqa but cannot execute it
                    # for the selected device/kernel. Fall back only for recognized
                    # GQA capability errors; preserve unrelated runtime failures.
                    message = str(exc).lower()
                    gqa_unavailable = isinstance(exc, (TypeError, NotImplementedError)) or any(
                        reason in message
                        for reason in (
                            "gqa",
                            "grouped query",
                            "grouped-query",
                            "not implemented",
                            "not supported",
                            "no viable backend",
                            "no available kernel",
                        )
                    )
                    if not gqa_unavailable:
                        raise
                    key = key.repeat_interleave(query_heads // kv_heads, dim=1)
                    value = value.repeat_interleave(query_heads // kv_heads, dim=1)
                    self.last_attention_backend = "pytorch-scaled-dot-product-attention-expanded-gqa"
                    return scaled_dot_product(
                        query,
                        key,
                        value,
                        attn_mask=attention_mask,
                        dropout_p=0.0,
                        is_causal=is_causal,
                    )
                self.last_attention_backend = "pytorch-scaled-dot-product-attention-gqa"
                return result

            self.last_attention_backend = "pytorch-scaled-dot-product-attention"
            return scaled_dot_product(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )

        # Older/minimal tensor backends use AXIOM's own masked attention math.
        # Their fallback expands grouped K/V heads only when the math requires it.
        if grouped_query:
            key = key.repeat_interleave(query_heads // kv_heads, dim=1)
            value = value.repeat_interleave(query_heads // kv_heads, dim=1)
        if blocked_mask is None and is_causal:
            query_length, key_length = query.shape[-2], key.shape[-2]
            query_positions = torch.arange(
                key_length - query_length,
                key_length,
                device=query.device,
            )
            key_positions = torch.arange(key_length, device=query.device)
            blocked_mask = key_positions[None, :] > query_positions[:, None]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        if blocked_mask is not None:
            scores = scores.masked_fill(blocked_mask[None, None, :, :], float("-inf"))
        probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        self.last_attention_backend = (
            "axiom-manual-matmul-attention-gqa" if grouped_query else "axiom-manual-matmul-attention"
        )
        return torch.matmul(probabilities, value)

    def _rms_norm(self, hidden: Any, weight: Any) -> Any:
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        return hidden * self.torch.rsqrt(variance + self.config.rms_norm_eps).to(dtype=hidden.dtype) * weight

    def _rope_embeddings(self, positions: Any, dtype: Any) -> tuple[Any, Any]:
        """Compute the rotary cosine/sine tables once for one forward sequence."""
        torch = self.torch
        dim = self.config.hidden_size // self.config.num_attention_heads
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, dim, 2, device=self.device, dtype=torch.float32) / dim)
        )
        scaled_positions = positions.to(dtype=torch.float32)
        if self.config.rope_scaling_type == "linear":
            scaled_positions = scaled_positions / self.config.rope_scaling_factor
        elif self.config.rope_scaling_type == "llama3":
            # Match Llama 3's wavelength-dependent frequency interpolation:
            # preserve high frequencies, scale low frequencies, and smooth the
            # transition between them instead of bluntly stretching all positions.
            wavelength = (2.0 * math.pi) / inv_freq
            original = float(self.config.rope_original_max_position_embeddings)
            low = float(self.config.rope_low_freq_factor)
            high = float(self.config.rope_high_freq_factor)
            low_wavelength = original / low
            high_wavelength = original / high
            scaled_inv_freq = inv_freq / self.config.rope_scaling_factor
            smooth = (original / wavelength - low) / (high - low)
            smooth_inv_freq = (1.0 - smooth) * scaled_inv_freq + smooth * inv_freq
            inv_freq = torch.where(
                wavelength < high_wavelength,
                inv_freq,
                torch.where(wavelength > low_wavelength, scaled_inv_freq, smooth_inv_freq),
            )
        frequencies = torch.einsum("i,j->ij", scaled_positions, inv_freq)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cos = embedding.cos().to(dtype=dtype)[None, None, :, :]
        sin = embedding.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin

    def _apply_rope(
        self,
        values: Any,
        positions: Any,
        embeddings: tuple[Any, Any] | None = None,
    ) -> Any:
        _, dim = self.torch, values.shape[-1]
        cos, sin = embeddings if embeddings is not None else self._rope_embeddings(positions, values.dtype)
        half = dim // 2
        rotated = self.torch.cat((-values[..., half:], values[..., :half]), dim=-1)
        return values * cos + rotated * sin

    def _cached_sequence_length(self, past_key_values: Any) -> int:
        """Validate a KV cache before any tensor operations and return its stored length."""
        if past_key_values is None:
            return 0
        if not isinstance(past_key_values, (tuple, list)) or len(past_key_values) != self.config.num_hidden_layers:
            raise NativeEngineError("Native KV cache must contain one key/value pair per transformer layer.")

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        expected_prefix = (1, self.config.num_key_value_heads)
        sequence_length: int | None = None
        for layer_index, pair in enumerate(past_key_values):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise NativeEngineError("Native KV cache layer entries must be key/value pairs.")
            key, value = pair
            if not self.torch.is_tensor(key) or not self.torch.is_tensor(value):
                raise NativeEngineError("Native KV cache entries must be tensors.")
            if key.ndim != 4 or value.ndim != 4 or tuple(key.shape) != tuple(value.shape):
                raise NativeEngineError("Native KV cache key/value tensor shapes are inconsistent.")
            if tuple(key.shape[:2]) != expected_prefix or key.shape[-1] != head_dim:
                raise NativeEngineError("Native KV cache tensor dimensions do not match the model configuration.")
            if key.dtype != value.dtype or key.device != value.device:
                raise NativeEngineError("Native KV cache key/value tensor types or devices are inconsistent.")
            if key.dtype != self.dtype:
                raise NativeEngineError("Native KV cache dtype does not match the loaded model weights.")
            expected_device = self.device if self.memory_mode == "full" else self.torch.device("cpu")
            if (
                key.device.type != expected_device.type
                or (
                    expected_device.index is not None
                    and key.device.index != expected_device.index
                )
            ):
                raise NativeEngineError("Native KV cache device does not match the selected memory mode.")
            layer_length = int(key.shape[-2])
            if layer_length < 1:
                raise NativeEngineError("Native KV cache sequence length must be positive.")
            if sequence_length is None:
                sequence_length = layer_length
            elif layer_length != sequence_length:
                raise NativeEngineError(
                    f"Native KV cache sequence length is inconsistent at layer {layer_index}."
                )
        return sequence_length or 0

    def forward(
        self,
        input_ids: Any,
        past_key_values: Any = None,
        *,
        past_sequence_length: int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[Any, tuple[Any, ...]]:
        torch, c = self.torch, self.config
        if not torch.is_tensor(input_ids):
            raise NativeEngineError("Native model input_ids must be a tensor.")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise NativeEngineError("AXIOM Native Engine currently accepts one prompt sequence per request.")
        if input_ids.shape[1] < 1:
            raise NativeEngineError("Native model input_ids must contain at least one token.")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise NativeEngineError("Native model input_ids must use an integer token-ID dtype.")
        if input_ids.device != self.device or input_ids.dtype != torch.long:
            input_ids = input_ids.to(device=self.device, dtype=torch.long)
        seq_len = input_ids.shape[1]
        stored_past_len = self._cached_sequence_length(past_key_values)
        past_len = stored_past_len if past_sequence_length is None else past_sequence_length
        if (
            isinstance(past_len, bool)
            or not isinstance(past_len, int)
            or past_len < stored_past_len
        ):
            raise NativeEngineError("Native KV cache position is inconsistent with the cached sequence.")
        if past_len + seq_len > self.config.max_position_embeddings:
            raise NativeEngineError("Generation would exceed the checkpoint's configured maximum context length.")
        hidden = torch.nn.functional.embedding(input_ids, self.weights["model.embed_tokens.weight"])
        positions = torch.arange(past_len, past_len + seq_len, device=self.device)
        rope_embeddings = self._rope_embeddings(positions, self.dtype)
        key_start_position = past_len - stored_past_len
        cache, head_dim = [], c.hidden_size // c.num_attention_heads
        for i in range(c.num_hidden_layers):
            if cancel_event is not None and cancel_event.is_set():
                raise NativeGenerationCancelled("Native generation was cancelled.")
            with self._layer_weights(i):
                p = f"model.layers.{i}."
                residual = hidden
                normed = self._rms_norm(hidden, self.weights[p + "input_layernorm.weight"])
                q, k, v = (self._linear(normed, p + "self_attn." + name) for name in ("q_proj", "k_proj", "v_proj"))
                q = q.view(1, seq_len, c.num_attention_heads, head_dim)
                k = k.view(1, seq_len, c.num_key_value_heads, head_dim)
                if c.qk_norm:
                    q = self._rms_norm(q, self.weights[p + "self_attn.q_norm.weight"])
                    k = self._rms_norm(k, self.weights[p + "self_attn.k_norm.weight"])
                q = self._apply_rope(q.transpose(1, 2), positions, rope_embeddings)
                k = self._apply_rope(k.transpose(1, 2), positions, rope_embeddings)
                v = v.view(1, seq_len, c.num_key_value_heads, head_dim).transpose(1, 2)
                if past_key_values is not None:
                    old_k, old_v = past_key_values[i]
                    if self.memory_mode in {"layer_streaming", "disk_streaming"}:
                        old_k, old_v = old_k.to(device=self.device), old_v.to(device=self.device)
                    k, v = torch.cat((old_k, k), dim=-2), torch.cat((old_v, v), dim=-2)
                # Keep only the useful KV suffix for sliding-window models. RoPE and
                # causal masks below still use absolute positions, even when this
                # bounded cache is shorter than the full token history.
                cache_k, cache_v = k, v
                if c.sliding_window and cache_k.shape[-2] > c.sliding_window:
                    # A slice view can retain the full prefill allocation as its
                    # backing storage; copy the suffix so old KV tensors are freed.
                    cache_k = cache_k[..., -c.sliding_window:, :].clone(
                        memory_format=torch.contiguous_format
                    )
                    cache_v = cache_v[..., -c.sliding_window:, :].clone(
                        memory_format=torch.contiguous_format
                    )
                if self.memory_mode in {"layer_streaming", "disk_streaming"}:
                    cache_k, cache_v = cache_k.detach().to(device="cpu"), cache_v.detach().to(device="cpu")
                    self.memory_stats["kv_cache_layer_offloads"] += 1
                    self.memory_stats["kv_cache_bytes_offloaded"] += int(
                        cache_k.numel() * cache_k.element_size() + cache_v.numel() * cache_v.element_size()
                    )
                    cache.append((cache_k, cache_v))
                else:
                    cache.append((cache_k, cache_v))
                # Let SDPA use its causal kernel for ordinary prefill and avoid
                # a dense sequence-by-sequence boolean mask. A one-token decode
                # with a populated non-sliding cache can attend to all stored keys
                # without a mask; sliding-window/chunked paths keep exact masks.
                is_causal = past_key_values is None and not c.sliding_window
                if not c.sliding_window and (
                    past_key_values is None or seq_len == 1
                ):
                    blocked = None
                else:
                    keys = torch.arange(key_start_position, past_len + seq_len, device=self.device)
                    blocked = keys[None, :] > positions[:, None]
                    if c.sliding_window:
                        blocked = blocked | (keys[None, :] <= positions[:, None] - c.sliding_window)
                attended = self._attention(
                    q, k, v, blocked, is_causal=is_causal
                ).transpose(1, 2).contiguous().view(1, seq_len, c.hidden_size)
                hidden = residual + self._linear(attended, p + "self_attn.o_proj")
                residual = hidden
                normed = self._rms_norm(hidden, self.weights[p + "post_attention_layernorm.weight"])
                gate, up = self._linear(normed, p + "mlp.gate_proj"), self._linear(normed, p + "mlp.up_proj")
                hidden = residual + self._linear(torch.nn.functional.silu(gate) * up, p + "mlp.down_proj")
        hidden = self._rms_norm(hidden, self.weights["model.norm.weight"])
        return self._linear(hidden[:, -1:, :], "lm_head")[:, -1, :], tuple(cache)

    def iter_generate(
        self,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        eos_token_ids: tuple[int, ...] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[int]:
        """Yield decoded tokens and stop cooperatively when cancellation is requested."""
        if not prompt_token_ids:
            raise NativeEngineError("Tokenizer returned no IDs for the prompt.")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            or token_id >= self.config.vocab_size
            for token_id in prompt_token_ids
        ):
            raise NativeEngineError("Prompt token IDs must be integers inside the checkpoint vocabulary.")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 2048:
            raise NativeEngineError("Native generation token limit must be between 1 and 2048.")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(float(temperature)) or not 0 <= float(temperature) <= 2:
            raise NativeEngineError("Native generation temperature must be a finite number between 0 and 2.")
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(float(top_p)) or not 0 < float(top_p) <= 1:
            raise NativeEngineError("Native generation top_p must be a finite number greater than 0 and at most 1.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 0 <= top_k <= 100_000:
            raise NativeEngineError("Native generation top_k must be an integer from 0 through 100000 (0 disables top-k).")
        if isinstance(min_p, bool) or not isinstance(min_p, (int, float)) or not math.isfinite(float(min_p)) or not 0 <= float(min_p) <= 1:
            raise NativeEngineError("Native generation min_p must be between 0 and 1 (0 disables min-p).")
        if isinstance(repetition_penalty, bool) or not isinstance(repetition_penalty, (int, float)) or not math.isfinite(float(repetition_penalty)) or not 0.1 <= float(repetition_penalty) <= 5:
            raise NativeEngineError("Native repetition_penalty must be a finite number between 0.1 and 5.")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (2**63 - 1)):
            raise NativeEngineError("Native sampling seed must be an integer from 0 through 2^63-1.")
        if top_k > self.config.vocab_size:
            raise NativeEngineError("top_k cannot exceed the checkpoint vocabulary size.")
        torch = self.torch
        sampling_generator = None
        if seed is not None:
            try:
                sampling_generator = torch.Generator(device=self.device)
                sampling_generator.manual_seed(seed)
            except (RuntimeError, TypeError, AttributeError):
                raise NativeEngineError("Seeded sampling is not supported by the selected tensor device.") from None
        eos = set(self.config.eos_token_ids if eos_token_ids is None else eos_token_ids)
        self.memory_stats = {
            "layer_weight_tensors_staged": 0,
            "layer_weight_bytes_staged": 0,
            "layer_weight_bytes_returned_to_cpu": 0,
            "layer_weight_bytes_released": 0,
            "disk_layer_weight_loads": 0,
            "disk_layer_weight_bytes_read": 0,
            "kv_cache_layer_offloads": 0,
            "kv_cache_bytes_offloaded": 0,
        }
        tokens = torch.tensor([prompt_token_ids], device=self.device, dtype=torch.long)
        generated: list[int] = []
        cache: tuple[Any, ...] = ()
        prefill_ms = 0.0
        decode_ms = 0.0
        total_started = time.perf_counter()
        stop_reason = "max_new_tokens"
        self.live_generation = {
            "stage": "prefill",
            "prompt_tokens": len(prompt_token_ids),
            "completion_tokens": 0,
            "prefill_ms": 0.0,
            "decode_ms": 0.0,
            "time_to_first_token_ms": 0.0,
            "generation_tokens_per_second": 0.0,
            "elapsed_ms": 0.0,
        }
        try:
            with torch.inference_mode():
                if cancel_event is not None and cancel_event.is_set():
                    stop_reason = "cancelled"
                    return
                prefill_started = time.perf_counter()
                try:
                    logits, cache = self.forward(tokens, cancel_event=cancel_event)
                    absolute_sequence_length = len(prompt_token_ids)
                    prefill_ms = (time.perf_counter() - prefill_started) * 1000
                    self.live_generation.update({
                        "stage": "decode",
                        "prefill_ms": round(prefill_ms, 3),
                        "time_to_first_token_ms": round(prefill_ms, 3),
                        "elapsed_ms": round((time.perf_counter() - total_started) * 1000, 3),
                    })
                except NativeGenerationCancelled:
                    prefill_ms = (time.perf_counter() - prefill_started) * 1000
                    stop_reason = "cancelled"
                    return
                for step in range(max_new_tokens):
                    if cancel_event is not None and cancel_event.is_set():
                        stop_reason = "cancelled"
                        break
                    scores = _sanitize_sampling_scores(logits.float(), torch)
                    if repetition_penalty != 1.0:
                        history = sorted(set(prompt_token_ids + generated))
                        if history:
                            history_ids = torch.tensor(history, device=self.device, dtype=torch.long)
                            prior_scores = scores.index_select(-1, history_ids)
                            adjusted_scores = torch.where(
                                prior_scores < 0,
                                prior_scores * float(repetition_penalty),
                                prior_scores / float(repetition_penalty),
                            )
                            scores.scatter_(-1, history_ids.view(1, -1), adjusted_scores)
                    if temperature == 0:
                        next_token = torch.argmax(scores, dim=-1)
                    else:
                        scores = _sanitize_sampling_scores(scores / float(temperature), torch)
                        scores = _apply_top_k_filter(scores, top_k, torch)
                        if top_p < 1:
                            sorted_scores, sorted_ids = torch.sort(scores, descending=True, dim=-1)
                            cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
                            remove = cumulative > top_p
                            remove[..., 1:] = remove[..., :-1].clone()
                            remove[..., 0] = False
                            sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
                            scores = torch.full_like(scores, float("-inf")).scatter(dim=-1, index=sorted_ids, src=sorted_scores)
                        scores = _apply_min_p_filter(scores, float(min_p))
                        if not bool(torch.isfinite(scores).any().item()):
                            raise NativeEngineError(
                                "Native sampling filters removed every usable token."
                            )
                        probabilities = torch.softmax(scores, dim=-1)
                        next_token = torch.multinomial(
                            probabilities, num_samples=1, generator=sampling_generator
                        ).squeeze(-1)
                    token_id = int(next_token.item())
                    if token_id < 0 or token_id >= self.config.vocab_size:
                        raise NativeEngineError("Native decoder produced a token ID outside vocab_size.")
                    if token_id in eos:
                        stop_reason = "eos_token"
                        break
                    generated.append(token_id)
                    live_elapsed = (time.perf_counter() - total_started) * 1000
                    live_compute_ms = prefill_ms + decode_ms
                    self.live_generation.update({
                        "stage": "decode",
                        "completion_tokens": len(generated),
                        "prefill_ms": round(prefill_ms, 3),
                        "decode_ms": round(decode_ms, 3),
                        "time_to_first_token_ms": round(prefill_ms, 3),
                        "generation_tokens_per_second": round(
                            len(generated) * 1000 / live_compute_ms, 3
                        ) if live_compute_ms > 0 else 0.0,
                        "elapsed_ms": round(live_elapsed, 3),
                    })
                    yield token_id
                    # Cancellation must not start another decode step.
                    if cancel_event is not None and cancel_event.is_set():
                        stop_reason = "cancelled"
                        break
                    if step + 1 < max_new_tokens:
                        decode_started = time.perf_counter()
                        try:
                            logits, cache = self.forward(
                                next_token.view(1, 1),
                                cache,
                                past_sequence_length=absolute_sequence_length,
                                cancel_event=cancel_event,
                            )
                            absolute_sequence_length += 1
                        except NativeGenerationCancelled:
                            decode_ms += (time.perf_counter() - decode_started) * 1000
                            stop_reason = "cancelled"
                            break
                        decode_ms += (time.perf_counter() - decode_started) * 1000
                        self.live_generation.update({
                            "decode_ms": round(decode_ms, 3),
                            "elapsed_ms": round((time.perf_counter() - total_started) * 1000, 3),
                            "generation_tokens_per_second": round(
                                len(generated) * 1000 / (prefill_ms + decode_ms), 3
                            ) if prefill_ms + decode_ms > 0 and generated else 0.0,
                        })
        finally:
            self.last_generation = {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(generated),
                "cache_layers": len(cache),
                "decoder": "axiom-autoregressive-kv-cache",
                "top_p": float(top_p),
                "top_k": top_k,
                "min_p": float(min_p),
                "repetition_penalty": float(repetition_penalty),
                "seed": seed,
                "stop_reason": stop_reason,
                "prefill_ms": round(prefill_ms, 3),
                "decode_ms": round(decode_ms, 3),
                "time_to_first_token_ms": round(prefill_ms, 3),
                "generation_tokens_per_second": round(
                    len(generated) * 1000 / (prefill_ms + decode_ms), 3
                ) if prefill_ms + decode_ms > 0 and generated else 0.0,
            }
            self.live_generation.update({
                "stage": "complete",
                "completion_tokens": len(generated),
                "prefill_ms": round(prefill_ms, 3),
                "decode_ms": round(decode_ms, 3),
                "time_to_first_token_ms": round(prefill_ms, 3),
                "generation_tokens_per_second": self.last_generation["generation_tokens_per_second"],
                "elapsed_ms": round((time.perf_counter() - total_started) * 1000, 3),
                "stop_reason": stop_reason,
            })

    def generate(
        self,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        eos_token_ids: tuple[int, ...] | None = None,
    ) -> list[int]:
        """Return the complete generated sequence; uses the same native streaming decoder."""
        return list(self.iter_generate(
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            eos_token_ids=eos_token_ids,
        ))


def load_native_runtime(
    model_dir: Path,
    torch_module: Any,
    device: Any,
    *,
    adapter_dir: Path | None = None,
    trusted_roots: tuple[Path, ...] | None = None,
    memory_mode: str = "full",
    precision: str = "auto",
) -> tuple[NativeLlamaRuntime, Any, dict[str, Any]]:
    """Load a supported local checkpoint/tokenizer without Transformers."""
    try:
        from tokenizers import Tokenizer
    except (ImportError, OSError):
        raise NativeDependencyError("AXIOM Native Engine requires tokenizers. Install the native-inference extra.") from None
    roots = trusted_roots or (model_dir.resolve(),)
    model = NativeLlamaRuntime.from_directory(
        model_dir, torch_module, device, adapter_dir=adapter_dir,
        trusted_roots=roots, memory_mode=memory_mode, precision=precision,
    )
    try:
        tokenizer = Tokenizer.from_file(str(_resolve_trusted_file(model_dir / "tokenizer.json", roots, "Tokenizer file")))
    except Exception:
        raise NativeEngineError("tokenizer.json is not compatible with AXIOM's native tokenizer adapter.") from None
    if tokenizer.get_vocab_size(with_added_tokens=True) > model.config.vocab_size:
        raise NativeEngineError("tokenizer.json vocabulary is larger than the model embedding table.")
    tokenizer_config = model_dir / "tokenizer_config.json"
    add_bos = model.config.bos_token_id is not None
    if tokenizer_config.exists():
        add_bos = _read_json_object(tokenizer_config, _MAX_CONFIG_BYTES, "Tokenizer config", roots).get("add_bos_token") is not False and add_bos
    details = {
        "model_type": model.config.model_type,
        "architecture": {
            "llama": "LlamaForCausalLM",
            "mistral": "MistralForCausalLM",
            "qwen2": "Qwen2ForCausalLM",
            "qwen3": "Qwen3ForCausalLM",
        }[model.config.model_type],
        "model_directory_name": model_dir.name,
        "model_directory_path": str(model_dir.resolve()),
        "adapter_directory_name": adapter_dir.name if adapter_dir else None,
        "add_bos_token": add_bos,
        "checkpoint_format": "safetensors",
        "execution_backend": "pytorch-native-axiom",
        "transformers_generation_used": False,
        "ollama_used": False,
        "dtype": str(model.dtype),
        "precision": {
            str(getattr(torch_module, "float32", "float32")): "fp32",
            str(getattr(torch_module, "float16", "float16")): "fp16",
            str(getattr(torch_module, "bfloat16", "bfloat16")): "bf16",
        }.get(str(model.dtype), str(model.dtype)),
        "precision_requested": precision,
        "device": str(device),
        "memory_mode": model.memory_mode,
    }
    return model, tokenizer, details
