from __future__ import annotations

import shutil
from dataclasses import dataclass


@dataclass
class ModelAnalysis:
    repo_id: str
    architecture: str
    model_type: str
    parameter_count: int | None
    parameter_source: str
    context_length: int | None
    precision: str
    weight_size_gb: float | None
    file_count: int
    files: list[str]


def _find_number(config: dict, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def estimate_parameters_from_config(config: dict) -> tuple[int | None, str]:
    direct_keys = (
        "num_parameters",
        "parameter_count",
        "n_parameters",
    )

    for key in direct_keys:
        value = config.get(key)

        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value, "model metadata"

    hidden = config.get("hidden_size")
    layers = config.get("num_hidden_layers")
    vocab = config.get("vocab_size")
    intermediate = config.get("intermediate_size")

    if all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (hidden, layers, vocab, intermediate)
    ):
        attention_heads = _find_number(config, ("num_attention_heads",))
        kv_heads = _find_number(config, ("num_key_value_heads",))
        if (
            attention_heads
            and kv_heads
            and kv_heads <= attention_heads
            and hidden % attention_heads == 0
        ):
            kv_width = (hidden // attention_heads) * kv_heads
        else:
            kv_width = hidden

        model_type = str(config.get("model_type") or "").lower()
        hidden_act = str(config.get("hidden_act") or "").lower()
        gated_mlp = hidden_act in {"silu", "swish"} or model_type in {
            "gemma", "gemma2", "llama", "mistral", "mixtral", "qwen2", "qwen3"
        }
        mlp_projections = 3 if gated_mlp else 2
        embedding_matrices = 1 if config.get("tie_word_embeddings") is True else 2

        attention_parameters = 2 * hidden * hidden + 2 * hidden * kv_width
        mlp_parameters = mlp_projections * hidden * intermediate
        norm_parameters = 2 * hidden
        estimate = int(
            layers * (attention_parameters + mlp_parameters + norm_parameters)
            + hidden  # final layer normalization
            + embedding_matrices * vocab * hidden
        )
        return estimate, "architecture estimate"

    return None, "unknown"


def analyze_config(
    repo_id: str,
    config: dict,
    files: list[str],
    weight_size_bytes: int | None = None,
) -> ModelAnalysis:
    if not isinstance(config, dict):
        raise TypeError("Model config must be a JSON object.")

    if weight_size_bytes is not None and (
        isinstance(weight_size_bytes, bool)
        or not isinstance(weight_size_bytes, int)
        or weight_size_bytes < 0
    ):
        raise ValueError("Weight size must be a non-negative integer.")

    architectures = config.get("architectures")

    architecture = (
        str(architectures[0])
        if isinstance(architectures, list) and architectures
        else "Unknown"
    )

    model_type = str(config.get("model_type") or "Unknown")

    parameters, parameter_source = estimate_parameters_from_config(config)

    context_length = _find_number(
        config,
        (
            "max_position_embeddings",
            "max_sequence_length",
            "seq_length",
            "model_max_length",
        ),
    )

    precision = str(config.get("torch_dtype") or "Unknown")

    quantization = config.get("quantization_config")

    if isinstance(quantization, dict):
        bits = quantization.get("bits")
        method = (
            quantization.get("quant_method")
            or quantization.get("method")
            or "quantized"
        )

        precision = f"{method} {bits}-bit" if bits else str(method)

    if any(name.lower().endswith(".gguf") for name in files):
        if precision == "Unknown":
            precision = "quantized / GGUF"

    elif precision == "Unknown" and any(
        name.lower().endswith(".safetensors") for name in files
    ):
        precision = "safetensors"

    weight_size_gb = (
        round(weight_size_bytes / (1024**3), 2)
        if weight_size_bytes is not None
        else None
    )

    return ModelAnalysis(
        repo_id=repo_id,
        architecture=architecture,
        model_type=model_type,
        parameter_count=parameters,
        parameter_source=parameter_source,
        context_length=context_length,
        precision=precision,
        weight_size_gb=weight_size_gb,
        file_count=len(files),
        files=files,
    )


def disk_info(path: str = ".") -> tuple[int, int]:
    usage = shutil.disk_usage(path)
    return usage.free, usage.total
