from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from axiom.core.native_engine import NativeEngineError, NativeLlamaConfig, _resolve_safetensor_files


def _config(**overrides):
    value = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 8,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "bos_token_id": 1,
        "tie_word_embeddings": False,
    }
    value.update(overrides)
    return value


def test_native_sampler_sanitizes_nonfinite_logits_and_rejects_empty_distribution() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import _sanitize_sampling_scores

    scores = _sanitize_sampling_scores(
        torch.tensor([[2.0, float("nan"), float("-inf"), -1.0]]),
        torch,
    )
    assert scores[0, 0].item() == 2.0
    assert scores[0, 1].item() == float("-inf")
    assert scores[0, 2].item() == float("-inf")
    assert scores[0, 3].item() == -1.0

    # Overflowed positive-infinity logits dominate normal finite logits, but
    # are converted to a valid equal-probability candidate set before softmax.
    overflowed = _sanitize_sampling_scores(
        torch.tensor([[float("inf"), 5.0, float("inf")]]),
        torch,
    )
    assert overflowed[0, 0].item() == 0.0
    assert overflowed[0, 1].item() == float("-inf")
    assert overflowed[0, 2].item() == 0.0
    assert torch.isfinite(torch.softmax(overflowed, dim=-1)).all().item()

    with pytest.raises(NativeEngineError, match="no usable token logits"):
        _sanitize_sampling_scores(
            torch.tensor([[float("nan"), float("-inf")]]),
            torch,
        )


def test_native_top_k_keeps_exact_candidate_count_when_logits_tie() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import _apply_top_k_filter

    logits = torch.tensor([[5.0, 5.0, 3.0, 1.0, -2.0]])
    one = _apply_top_k_filter(logits, 1, torch)
    assert torch.isfinite(one).sum().item() == 1
    assert one.max().item() == 5.0

    two = _apply_top_k_filter(logits, 2, torch)
    assert torch.isfinite(two).sum().item() == 2
    assert torch.isfinite(two[0, :2]).all().item()
    assert torch.isneginf(two[0, 2:]).all().item()

    # A disabled or full-vocabulary top-k must be a no-op.
    assert torch.equal(_apply_top_k_filter(logits, 0, torch), logits)
    assert torch.equal(_apply_top_k_filter(logits, logits.shape[-1], torch), logits)


def test_validate_min_p_rejects_nonfinite_and_out_of_range_values() -> None:
    from axiom.core.engine import validate_min_p

    assert validate_min_p(0) == 0.0
    assert validate_min_p(0.05) == 0.05
    assert validate_min_p(1) == 1.0
    for invalid in (-0.01, 1.01, float("nan"), float("inf"), True, "0.2"):
        with pytest.raises(ValueError, match="Min-p"):
            validate_min_p(invalid)


def test_native_min_p_filters_by_probability_relative_to_best_logit() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import _apply_min_p_filter

    logits = torch.tensor([[0.0, -0.5, -0.8, -2.0]])
    filtered = _apply_min_p_filter(logits, 0.5)
    # Softmax denominators cancel in the relative-probability comparison:
    # exp(score - max_score) >= min_p.
    assert torch.isfinite(filtered[0, :2]).all().item()
    assert torch.isneginf(filtered[0, 2:]).all().item()

    strict = _apply_min_p_filter(logits, 1.0)
    assert torch.isfinite(strict).sum().item() == 1
    assert strict[0, 0].item() == 0.0
    assert torch.isneginf(strict[0, 1:]).all().item()

    assert torch.equal(_apply_min_p_filter(logits, 0.0), logits)


def test_stop_sequences_are_bounded_and_deduplicated() -> None:
    from axiom.core.engine import validate_stop_sequences

    assert validate_stop_sequences(["<END>", "###", "<END>"]) == ("<END>", "###")
    assert validate_stop_sequences([]) == ()
    with pytest.raises(ValueError, match="non-whitespace"):
        validate_stop_sequences(["   "])
    with pytest.raises(ValueError, match="256 characters"):
        validate_stop_sequences(["S" * 257])
    with pytest.raises(ValueError, match="At most 16"):
        validate_stop_sequences([str(index) for index in range(17)])


def test_native_stream_stops_on_a_sequence_spanning_token_boundaries(monkeypatch) -> None:
    from types import SimpleNamespace
    from axiom.core import engine

    class TokenizerFixture:
        def encode(self, _prompt, add_special_tokens=False):
            return SimpleNamespace(ids=[1])

        def decode(self, token_ids, skip_special_tokens=True):
            lookup = {
                (): "",
                (101,): "answer<",
                (101, 102): "answer<END>discarded",
                (101, 102, 103): "answer<END>discarded extra",
            }
            return lookup.get(tuple(token_ids), "")

    class RuntimeFixture:
        config = SimpleNamespace(max_position_embeddings=64, bos_token_id=None, eos_token_ids=())
        last_attention_backend = "axiom-manual-matmul-attention"

        def __init__(self):
            self.requested = 0
            self.last_generation = {}

        def iter_generate(self, _prompt_ids, **_kwargs):
            try:
                for token_id in (101, 102, 103):
                    self.requested += 1
                    yield token_id
            finally:
                self.last_generation.update({
                    "decoder": "axiom-autoregressive-kv-cache",
                    "prompt_tokens": 1,
                    "completion_tokens": self.requested,
                    "cache_layers": 1,
                    "prefill_ms": 2.0,
                    "decode_ms": 1.0,
                    "time_to_first_token_ms": 2.0,
                    "generation_tokens_per_second": 666.667,
                    "stop_reason": "max_new_tokens",
                })

    runtime = RuntimeFixture()
    tokenizer = TokenizerFixture()
    monkeypatch.setattr(
        engine,
        "_load_local_model",
        lambda *_args: (
            runtime,
            tokenizer,
            None,
            "cpu",
            {
                "model_directory_name": "fixture",
                "adapter_directory_name": None,
                "architecture": "LlamaForCausalLM",
                "model_type": "llama",
                "add_bos_token": False,
            },
        ),
    )

    events = list(engine.generate_local_stream(
        "/trusted/models/fixture", "question",
        max_new_tokens=10, temperature=0,
        stop_sequences=["<END>"],
    ))
    token_events = [event for event in events if event["type"] == "token"]
    completion = events[-1]["result"]
    assert len(token_events) == 2
    assert all(event["response"] == "answer" for event in token_events)
    assert completion["response"] == "answer"
    assert completion["stop_reason"] == "stop_sequence"
    assert completion["stop_sequences"] == ["<END>"]
    assert runtime.requested == 2


def test_native_config_accepts_llama_mistral_qwen2_and_qwen3() -> None:
    assert NativeLlamaConfig.from_mapping(_config()).model_type == "llama"
    assert NativeLlamaConfig.from_mapping(
        _config(model_type="mistral", architectures=["MistralForCausalLM"], sliding_window=16)
    ).sliding_window == 16
    qwen = NativeLlamaConfig.from_mapping(_config(
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
        rope_theta=1_000_000.0,
        max_position_embeddings=32768,
        tie_word_embeddings=True,
    ))
    assert qwen.model_type == "qwen2"
    assert qwen.rope_theta == 1_000_000.0
    assert qwen.max_position_embeddings == 32768
    qwen3 = NativeLlamaConfig.from_mapping(_config(
        model_type="qwen3", architectures=["Qwen3ForCausalLM"],
        rope_theta=1_000_000.0, max_position_embeddings=32768,
        tie_word_embeddings=True,
    ))
    assert qwen3.model_type == "qwen3"
    assert qwen3.qk_norm is True


def test_native_config_supports_linear_rope_scaling() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    default_config = NativeLlamaConfig.from_mapping(_config())
    linear_config = NativeLlamaConfig.from_mapping(_config(
        rope_scaling={"rope_type": "linear", "factor": 2.0}
    ))
    assert linear_config.rope_scaling_type == "linear"
    assert linear_config.rope_scaling_factor == 2.0

    weights = _tiny_weights(torch)
    default_runtime = NativeLlamaRuntime(default_config, weights, torch, torch.device("cpu"))
    linear_runtime = NativeLlamaRuntime(linear_config, weights, torch, torch.device("cpu"))
    values = torch.randn((1, 2, 2, 4), dtype=torch.float32)
    scaled = linear_runtime._apply_rope(values, torch.tensor([0, 2]))
    reference = default_runtime._apply_rope(values, torch.tensor([0, 1]))
    assert torch.allclose(scaled, reference, atol=1e-6, rtol=1e-5)


def test_native_config_supports_llama3_rope_scaling_formula() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config(
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "original_max_position_embeddings": 16,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
        }
    ))
    runtime = NativeLlamaRuntime(config, _tiny_weights(torch), torch, torch.device("cpu"))
    positions = torch.arange(4)
    actual_cos, actual_sin = runtime._rope_embeddings(positions, torch.float32)

    dim = config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    wavelength = (2.0 * torch.pi) / inv_freq
    low_wavelength = 16.0 / 1.0
    high_wavelength = 16.0 / 4.0
    scaled_inv = inv_freq / 8.0
    smooth = (16.0 / wavelength - 1.0) / (4.0 - 1.0)
    smooth_inv = (1.0 - smooth) * scaled_inv + smooth * inv_freq
    expected_inv = torch.where(
        wavelength < high_wavelength,
        inv_freq,
        torch.where(wavelength > low_wavelength, scaled_inv, smooth_inv),
    )
    frequencies = torch.einsum("i,j->ij", positions.to(torch.float32), expected_inv)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    assert torch.allclose(actual_cos[0, 0], embedding.cos(), atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual_sin[0, 0], embedding.sin(), atol=1e-6, rtol=1e-5)


def test_dynamic_ntk_rope_scaling_extends_context_and_matches_frequency_formula() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config(
        max_position_embeddings=8,
        rope_theta=10_000.0,
        rope_scaling={"rope_type": "dynamic", "factor": 2.0},
    ))
    assert config.rope_scaling_type == "dynamic"
    assert config.max_position_embeddings == 8
    assert config.effective_max_position_embeddings == 16

    runtime = NativeLlamaRuntime(
        config, _tiny_weights(torch), torch, torch.device("cpu")
    )
    positions = torch.arange(12)
    actual_cos, actual_sin = runtime._rope_embeddings(positions, torch.float32)

    dim = config.hidden_size // config.num_attention_heads
    scale = (
        config.rope_scaling_factor * len(positions) / config.max_position_embeddings
        - (config.rope_scaling_factor - 1.0)
    )
    dynamic_theta = config.rope_theta * scale ** (dim / (dim - 2))
    inverse_frequencies = 1.0 / (
        dynamic_theta
        ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    frequencies = torch.einsum(
        "i,j->ij", positions.to(torch.float32), inverse_frequencies
    )
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    assert torch.allclose(actual_cos[0, 0], embedding.cos(), atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual_sin[0, 0], embedding.sin(), atol=1e-6, rtol=1e-5)

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5]], dtype=torch.long)
    with torch.inference_mode():
        logits, cache = runtime.forward(input_ids)
    assert tuple(logits.shape) == (1, config.vocab_size)
    assert tuple(cache[0][0].shape)[-2] == len(input_ids[0])
    with pytest.raises(NativeEngineError, match="maximum context length"):
        runtime.forward(torch.ones((1, 17), dtype=torch.long))


def test_native_config_supports_yarn_rope_scaling_formula() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config(
        rope_scaling={
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 16,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 1.25,
            "truncate": True,
        }
    ))
    assert config.rope_scaling_type == "yarn"
    assert config.rope_scaling_factor == 2.0
    assert config.rope_attention_factor == 1.25
    assert config.rope_beta_fast == 32.0
    assert config.rope_beta_slow == 1.0
    assert config.rope_truncate is True

    runtime = NativeLlamaRuntime(
        config, _tiny_weights(torch), torch, torch.device("cpu")
    )
    positions = torch.arange(4)
    actual_cos, actual_sin = runtime._rope_embeddings(positions, torch.float32)

    dim = config.hidden_size // config.num_attention_heads
    inverse_frequencies = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )

    def correction_dimension(rotations: float) -> float:
        return (
            dim * math.log(
                config.rope_original_max_position_embeddings
                / (rotations * 2.0 * math.pi)
            )
            / (2.0 * math.log(config.rope_theta))
        )

    low = math.floor(correction_dimension(config.rope_beta_fast))
    high = math.ceil(correction_dimension(config.rope_beta_slow))
    low, high = max(low, 0), min(high, dim - 1)
    if high <= low:
        high += 0.001
    ramp = (
        (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)
    ).clamp(0.0, 1.0)
    extrapolation_factor = 1.0 - ramp
    expected_inverse = (
        (inverse_frequencies / config.rope_scaling_factor) * (1.0 - extrapolation_factor)
        + inverse_frequencies * extrapolation_factor
    )
    frequencies = torch.einsum(
        "i,j->ij", positions.to(torch.float32), expected_inverse
    )
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    expected_cos = embedding.cos() * config.rope_attention_factor
    expected_sin = embedding.sin() * config.rope_attention_factor
    assert torch.allclose(actual_cos[0, 0], expected_cos, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual_sin[0, 0], expected_sin, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "overrides,detail",
    [
        ({"bos_token_id": True}, "BOS token ID"),
        ({"eos_token_id": [2, True]}, "EOS token IDs"),
        ({"tie_word_embeddings": "true"}, "tie_word_embeddings"),
        ({"rope_scaling": 0}, "RoPE scaling"),
    ],
)
def test_native_config_rejects_malformed_token_and_rope_metadata(overrides, detail) -> None:
    with pytest.raises(NativeEngineError, match=detail):
        NativeLlamaConfig.from_mapping(_config(**overrides))


@pytest.mark.parametrize(
    "overrides,detail",
    [
        ({"model_type": "gemma3"}, "supports llama, mistral, qwen2, and qwen3"),
        ({"hidden_size": 7}, "even-sized attention heads"),
        ({"rope_scaling": {"rope_type": "yarn", "factor": 2.0}}, "RoPE scaling"),
        ({"rope_scaling": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 16, "beta_fast": 1.0, "beta_slow": 32.0}}, "YaRN RoPE scaling requires"),
        ({"rope_scaling": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 16, "attention_factor": 0}}, "YaRN attention_factor"),
        ({"rope_scaling": {"rope_type": "yarn", "factor": 129.0, "original_max_position_embeddings": 16}}, "YaRN RoPE scaling factor"),
        ({"rope_scaling": {"rope_type": "default", "factor": True}}, "RoPE scaling"),
        ({"model_type": "llama", "architectures": ["Qwen3ForCausalLM"]}, "does not match"),
        ({"num_attention_heads": 3, "hidden_size": 12, "num_key_value_heads": 2}, "num_key_value_heads"),
        ({"hidden_size": True}, "missing valid native Llama model dimensions"),
        ({"num_hidden_layers": 1.5}, "missing valid native Llama model dimensions"),
        ({"rope_theta": True}, "missing valid native Llama model dimensions"),
    ],
)
def test_native_config_rejects_unsupported_or_inconsistent_architecture(overrides, detail) -> None:
    with pytest.raises(NativeEngineError, match=detail):
        NativeLlamaConfig.from_mapping(_config(**overrides))


def test_safetensor_index_rejects_relative_path_traversal(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "../outside.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(NativeEngineError, match="unsafe or unsupported shard path"):
        _resolve_safetensor_files(tmp_path)


def test_native_asset_symlink_must_resolve_inside_trusted_roots(tmp_path: Path) -> None:
    from axiom.core.native_engine import _resolve_trusted_file

    trusted = tmp_path / "trusted"
    blobs = trusted / "blobs"
    checkpoint = trusted / "snapshot"
    outside = tmp_path / "outside"
    blobs.mkdir(parents=True)
    checkpoint.mkdir()
    outside.mkdir()
    safe_blob = blobs / "config.json"
    safe_blob.write_text('{"model_type":"llama"}', encoding="utf-8")
    safe_link = checkpoint / "config.json"
    try:
        safe_link.symlink_to(safe_blob)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"File symlinks are unavailable on this platform: {exc}")
    assert _resolve_trusted_file(safe_link, (trusted.resolve(),), "Model config") == safe_blob.resolve()

    external = outside / "config.json"
    external.write_text('{"model_type":"llama"}', encoding="utf-8")
    unsafe_link = checkpoint / "external-config.json"
    unsafe_link.symlink_to(external)
    with pytest.raises(NativeEngineError, match="outside the configured trusted model roots"):
        _resolve_trusted_file(unsafe_link, (trusted.resolve(),), "Model config")


def _tiny_weights(torch, vocab_size=8, hidden=8, intermediate=12):
    weights = {
        "model.embed_tokens.weight": torch.randn(vocab_size, hidden) * 0.02,
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": torch.randn(vocab_size, hidden) * 0.02,
        "model.layers.0.input_layernorm.weight": torch.ones(hidden),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(hidden),
    }
    dimensions = {
        "self_attn.q_proj": (hidden, hidden),
        "self_attn.k_proj": (hidden // 2, hidden),
        "self_attn.v_proj": (hidden // 2, hidden),
        "self_attn.o_proj": (hidden, hidden),
        "mlp.gate_proj": (intermediate, hidden),
        "mlp.up_proj": (intermediate, hidden),
        "mlp.down_proj": (hidden, intermediate),
    }
    for name, shape in dimensions.items():
        weights["model.layers.0." + name + ".weight"] = torch.randn(*shape) * 0.02
    return weights


@pytest.mark.parametrize("force_manual", [False, True])
def test_native_grouped_query_attention_matches_expanded_reference(force_manual: bool, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config())
    runtime = NativeLlamaRuntime(config, _tiny_weights(torch), torch, torch.device("cpu"))
    query = torch.randn((1, 2, 3, 4), dtype=torch.float32)
    key = torch.randn((1, 1, 3, 4), dtype=torch.float32)
    value = torch.randn((1, 1, 3, 4), dtype=torch.float32)
    blocked = torch.triu(torch.ones((3, 3), dtype=torch.bool), diagonal=1)
    native_sdpa = torch.nn.functional.scaled_dot_product_attention
    calls = []

    if force_manual:
        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", None)
    else:
        def capture_sdpa(q, k, v, **kwargs):
            calls.append({
                "query_heads": q.shape[1],
                "key_heads": k.shape[1],
                "value_heads": v.shape[1],
                "enable_gqa": kwargs.get("enable_gqa"),
            })
            return native_sdpa(q, k, v, **kwargs)

        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", capture_sdpa)

    actual = runtime._attention(query, key, value, blocked)
    expanded_key = key.repeat_interleave(2, dim=1)
    expanded_value = value.repeat_interleave(2, dim=1)
    scores = torch.matmul(query, expanded_key.transpose(-2, -1)) / (4 ** 0.5)
    probabilities = torch.softmax(
        scores.masked_fill(blocked[None, None, :, :], float("-inf")), dim=-1
    )
    expected = torch.matmul(probabilities, expanded_value)

    assert torch.allclose(actual, expected, atol=2e-5, rtol=1e-4)
    if force_manual:
        assert runtime.last_attention_backend == "axiom-manual-matmul-attention-gqa"
    else:
        assert runtime.last_attention_backend in {
            "pytorch-scaled-dot-product-attention-gqa",
            "pytorch-scaled-dot-product-attention-expanded-gqa",
        }
        assert calls[0]["query_heads"] == 2
        if runtime.last_attention_backend == "pytorch-scaled-dot-product-attention-gqa":
            assert calls[0]["key_heads"] == calls[0]["value_heads"] == 1
            assert calls[0]["enable_gqa"] is True
        else:
            # Older SDPA builds reject enable_gqa; the compatibility path expands K/V.
            assert calls[-1]["key_heads"] == calls[-1]["value_heads"] == 2


def test_native_gqa_falls_back_when_backend_reports_unsupported_kernel(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    runtime = NativeLlamaRuntime(
        NativeLlamaConfig.from_mapping(_config()),
        _tiny_weights(torch),
        torch,
        torch.device("cpu"),
    )
    native_sdpa = torch.nn.functional.scaled_dot_product_attention
    calls = []

    def reject_native_gqa(query, key, value, **kwargs):
        calls.append((key.shape[1], kwargs.get("enable_gqa", False)))
        if kwargs.get("enable_gqa"):
            raise RuntimeError("grouped-query attention is not supported by this test backend")
        return native_sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", reject_native_gqa)
    query = torch.randn((1, 2, 3, 4), dtype=torch.float32)
    key = torch.randn((1, 1, 3, 4), dtype=torch.float32)
    value = torch.randn((1, 1, 3, 4), dtype=torch.float32)
    blocked = torch.triu(torch.ones((3, 3), dtype=torch.bool), diagonal=1)

    actual = runtime._attention(query, key, value, blocked)
    expanded_key = key.repeat_interleave(2, dim=1)
    expanded_value = value.repeat_interleave(2, dim=1)
    scores = torch.matmul(query, expanded_key.transpose(-2, -1)) / (4 ** 0.5)
    probabilities = torch.softmax(
        scores.masked_fill(blocked[None, None], float("-inf")), dim=-1
    )
    expected = torch.matmul(probabilities, expanded_value)

    assert torch.allclose(actual, expected, atol=2e-5, rtol=1e-4)
    assert runtime.last_attention_backend == "pytorch-scaled-dot-product-attention-expanded-gqa"
    assert calls == [(1, True), (2, False)]


@pytest.mark.parametrize("force_manual", [False, True])
def test_native_attention_sdpa_and_manual_paths_match_reference(force_manual: bool, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config())
    runtime = NativeLlamaRuntime(config, _tiny_weights(torch), torch, torch.device("cpu"))
    if force_manual:
        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", None)

    query = torch.randn((1, 2, 3, 4), dtype=torch.float32)
    key = torch.randn((1, 2, 3, 4), dtype=torch.float32)
    value = torch.randn((1, 2, 3, 4), dtype=torch.float32)
    blocked = torch.triu(torch.ones((3, 3), dtype=torch.bool), diagonal=1)
    actual = runtime._attention(query, key, value, blocked)

    scores = torch.matmul(query, key.transpose(-2, -1)) / (4 ** 0.5)
    expected_probs = torch.softmax(scores.masked_fill(blocked[None, None], float("-inf")), dim=-1)
    expected = torch.matmul(expected_probs, value)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=1e-4)
    assert runtime.last_attention_backend == (
        "axiom-manual-matmul-attention" if force_manual else "pytorch-scaled-dot-product-attention"
    )


@pytest.mark.parametrize(
    "cache_kind, detail",
    [
        ("wrong_layer_count", "one key/value pair per transformer layer"),
        ("mismatched_key_value_shape", "key/value tensor shapes are inconsistent"),
        ("wrong_head_count", "tensor dimensions do not match the model configuration"),
        ("wrong_dtype", "dtype does not match the loaded model weights"),
        ("wrong_device", "device does not match the selected memory mode"),
        ("empty_sequence", "sequence length must be positive"),
        ("non_tensor", "entries must be tensors"),
    ],
)
def test_native_forward_rejects_malformed_kv_cache(cache_kind: str, detail: str) -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config())
    runtime = NativeLlamaRuntime(config, _tiny_weights(torch), torch, torch.device("cpu"))
    with torch.inference_mode():
        _, cache = runtime.forward(torch.tensor([[1, 3, 4]], dtype=torch.long))
        if cache_kind == "wrong_layer_count":
            malformed = ()
        elif cache_kind == "mismatched_key_value_shape":
            malformed = ((cache[0][0][..., :-1, :], cache[0][1]),)
        elif cache_kind == "wrong_head_count":
            malformed = ((
                cache[0][0].repeat_interleave(2, dim=1),
                cache[0][1].repeat_interleave(2, dim=1),
            ),)
        elif cache_kind == "wrong_dtype":
            malformed = ((cache[0][0].double(), cache[0][1].double()),)
        elif cache_kind == "wrong_device":
            malformed = ((cache[0][0].to(device="meta"), cache[0][1].to(device="meta")),)
        elif cache_kind == "empty_sequence":
            malformed = ((cache[0][0][..., :0, :], cache[0][1][..., :0, :]),)
        else:
            malformed = ((None, None),)

        with pytest.raises(NativeEngineError, match=detail):
            runtime.forward(torch.tensor([[5]], dtype=torch.long), malformed)


def test_native_forward_computes_rope_embeddings_once_for_all_layers(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config(num_hidden_layers=2))
    weights = _tiny_weights(torch)
    for name, tensor in list(weights.items()):
        if name.startswith("model.layers.0."):
            weights[name.replace("model.layers.0.", "model.layers.1.", 1)] = tensor.clone()
    runtime = NativeLlamaRuntime(config, weights, torch, torch.device("cpu"))
    original = runtime._rope_embeddings
    calls = []

    def capture(positions, dtype):
        calls.append((positions.detach().cpu().tolist(), dtype))
        return original(positions, dtype)

    monkeypatch.setattr(runtime, "_rope_embeddings", capture)
    with torch.inference_mode():
        logits, _ = runtime.forward(torch.tensor([[1, 2, 3]], dtype=torch.long))

    assert tuple(logits.shape) == (1, config.vocab_size)
    assert calls == [([0, 1, 2], torch.float32)]


def test_native_forward_uses_implicit_causal_prefill_and_unmasked_decode(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config())
    runtime = NativeLlamaRuntime(config, _tiny_weights(torch), torch, torch.device("cpu"))
    calls = []
    original = runtime._attention

    def capture(query, key, value, blocked_mask, *, is_causal=False):
        calls.append((None if blocked_mask is None else tuple(blocked_mask.shape), is_causal))
        return original(query, key, value, blocked_mask, is_causal=is_causal)

    monkeypatch.setattr(runtime, "_attention", capture)
    with torch.inference_mode():
        _, cache = runtime.forward(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
        runtime.forward(
            torch.tensor([[5]], dtype=torch.long),
            cache,
            past_sequence_length=4,
        )

    assert calls == [(None, True), (None, False)]


def test_native_attention_implicit_causal_mask_matches_explicit_reference() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    runtime = NativeLlamaRuntime(
        NativeLlamaConfig.from_mapping(_config()),
        _tiny_weights(torch),
        torch,
        torch.device("cpu"),
    )
    query = torch.randn((1, 2, 4, 4), dtype=torch.float32)
    key = torch.randn((1, 1, 4, 4), dtype=torch.float32)
    value = torch.randn((1, 1, 4, 4), dtype=torch.float32)
    actual = runtime._attention(query, key, value, None, is_causal=True)

    expanded_key = key.repeat_interleave(2, dim=1)
    expanded_value = value.repeat_interleave(2, dim=1)
    scores = torch.matmul(query, expanded_key.transpose(-2, -1)) / (4 ** 0.5)
    causal = torch.triu(torch.ones((4, 4), dtype=torch.bool), diagonal=1)
    probabilities = torch.softmax(
        scores.masked_fill(causal[None, None], float("-inf")), dim=-1
    )
    expected = torch.matmul(probabilities, expanded_value)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=1e-4)


def test_native_generate_batch_matches_serial_greedy_generation() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config())
    runtime = NativeLlamaRuntime(config, _tiny_weights(torch), torch, torch.device("cpu"))
    prompts = [[1, 2, 3], [3, 2, 1]]

    with torch.inference_mode():
        actual = runtime.generate_batch(
            prompts, max_new_tokens=3, temperature=0, eos_token_ids=()
        )
        expected = [
            runtime.generate(prompt, max_new_tokens=3, temperature=0, eos_token_ids=())
            for prompt in prompts
        ]

    assert actual == expected
    assert len(actual) == 2
    assert all(len(output) == 3 for output in actual)
    assert runtime.last_batch_generation["batch_size"] == 2


def test_native_generate_batch_buckets_different_prompt_lengths() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config())
    runtime = NativeLlamaRuntime(config, _tiny_weights(torch), torch, torch.device("cpu"))
    prompts = [[1, 2, 3], [3, 2], [2, 3, 1, 1], [3, 2, 1]]

    with torch.inference_mode():
        actual = runtime.generate_batch(
            prompts, max_new_tokens=2, temperature=0, eos_token_ids=()
        )
        expected = [
            runtime.generate(prompt, max_new_tokens=2, temperature=0, eos_token_ids=())
            for prompt in prompts
        ]

    assert actual == expected
    assert runtime.last_batch_generation["batch_size"] == len(prompts)
    assert runtime.last_batch_generation["batch_groups"] == 3
    assert all(len(output) == 2 for output in actual)


def test_native_generate_batch_bounds_same_length_microbatches_and_preserves_order(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config())
    runtime = NativeLlamaRuntime(config, _tiny_weights(torch), torch, torch.device("cpu"))
    prompts = [[(index % (config.vocab_size - 1)) + 1, 2, 3] for index in range(10)]
    observed_batch_sizes = []
    original_forward = runtime.forward

    def record_forward(input_ids, *args, **kwargs):
        observed_batch_sizes.append(int(input_ids.shape[0]))
        return original_forward(input_ids, *args, **kwargs)

    monkeypatch.setattr(runtime, "forward", record_forward)
    with torch.inference_mode():
        actual = runtime.generate_batch(
            prompts, max_new_tokens=2, temperature=0, eos_token_ids=()
        )
    batched_sizes = list(observed_batch_sizes)
    assert actual == [
        runtime.generate(prompt, max_new_tokens=2, temperature=0, eos_token_ids=())
        for prompt in prompts
    ]
    assert len(actual) == 10
    assert batched_sizes
    assert max(batched_sizes) <= 8
    assert runtime.last_batch_generation["batch_groups"] == 1
    assert runtime.last_batch_generation["microbatch_count"] == 2
    assert runtime.last_batch_generation["microbatch_size_limit"] == 8
    assert runtime.last_batch_generation["max_microbatch_size"] == 8


def test_native_generate_batch_handles_eos_and_validates_prompts(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeEngineError, NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config())
    runtime = NativeLlamaRuntime(config, _tiny_weights(torch), torch, torch.device("cpu"))
    variable_length_outputs = runtime.generate_batch(
        [[1, 2], [3]], max_new_tokens=1, temperature=0, eos_token_ids=()
    )
    assert len(variable_length_outputs) == 2
    assert runtime.last_batch_generation["batch_groups"] == 2
    with pytest.raises(NativeEngineError, match="inside the checkpoint vocabulary"):
        runtime.generate_batch([[1], [config.vocab_size]], max_new_tokens=1, temperature=0)

    eos_id = 7
    def eos_forward(input_ids, past_key_values=None, *, past_sequence_length=None, cancel_event=None):
        batch = int(input_ids.shape[0])
        logits = torch.full((batch, config.vocab_size), -100.0)
        logits[:, eos_id] = 100.0
        cache_length = int(input_ids.shape[1]) if past_key_values is None else int(past_key_values[0][0].shape[-2]) + int(input_ids.shape[1])
        cache = tuple(
            (
                torch.zeros((batch, config.num_key_value_heads, cache_length, config.hidden_size // config.num_attention_heads)),
                torch.zeros((batch, config.num_key_value_heads, cache_length, config.hidden_size // config.num_attention_heads)),
            )
            for _ in range(config.num_hidden_layers)
        )
        return logits, cache

    monkeypatch.setattr(runtime, "forward", eos_forward)
    outputs = runtime.generate_batch(
        [[1, 2], [2, 1]], max_new_tokens=2, temperature=0, eos_token_ids=(eos_id,)
    )
    assert outputs == [[], []]
    assert runtime.last_batch_generation["finished_sequences"] == 2


def test_native_forward_supports_batched_prefill_and_incremental_decode() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config())
    runtime = NativeLlamaRuntime(config, _tiny_weights(torch), torch, torch.device("cpu"))
    input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long)

    with torch.inference_mode():
        batch_logits, batch_cache = runtime.forward(input_ids)
        single_runs = [runtime.forward(input_ids[row:row + 1]) for row in range(input_ids.shape[0])]

    assert tuple(batch_logits.shape) == (2, config.vocab_size)
    assert torch.allclose(batch_logits, torch.cat([item[0] for item in single_runs]), atol=2e-5, rtol=1e-4)
    for layer_index, (batch_key, batch_value) in enumerate(batch_cache):
        assert tuple(batch_key.shape[:3]) == (2, config.num_key_value_heads, input_ids.shape[1])
        assert torch.allclose(batch_key[0:1], single_runs[0][1][layer_index][0], atol=1e-6, rtol=1e-5)
        assert torch.allclose(batch_key[1:2], single_runs[1][1][layer_index][0], atol=1e-6, rtol=1e-5)
        assert torch.allclose(batch_value[0:1], single_runs[0][1][layer_index][1], atol=1e-6, rtol=1e-5)
        assert torch.allclose(batch_value[1:2], single_runs[1][1][layer_index][1], atol=1e-6, rtol=1e-5)

    next_ids = torch.tensor([[4], [5]], dtype=torch.long)
    with torch.inference_mode():
        batch_next_logits, batch_next_cache = runtime.forward(
            next_ids, batch_cache, past_sequence_length=input_ids.shape[1]
        )
        single_next = [
            runtime.forward(
                next_ids[row:row + 1],
                single_runs[row][1],
                past_sequence_length=input_ids.shape[1],
            )
            for row in range(next_ids.shape[0])
        ]
    assert tuple(batch_next_logits.shape) == (2, config.vocab_size)
    assert torch.allclose(
        batch_next_logits, torch.cat([item[0] for item in single_next]), atol=2e-5, rtol=1e-4
    )
    with pytest.raises(NativeEngineError, match="batch size must match"):
        runtime.forward(torch.tensor([[6], [7], [1]], dtype=torch.long), batch_next_cache)


def test_native_forward_rejects_zero_sized_batches() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeEngineError, NativeLlamaRuntime

    runtime = NativeLlamaRuntime(
        NativeLlamaConfig.from_mapping(_config()), _tiny_weights(torch), torch, torch.device("cpu")
    )
    with pytest.raises(NativeEngineError, match="batch must contain at least one"):
        runtime.forward(torch.empty((0, 1), dtype=torch.long))


def test_native_forward_and_incremental_kv_cache_run_without_transformers() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config())
    runtime = NativeLlamaRuntime(config, _tiny_weights(torch), torch, torch.device("cpu"))
    with torch.inference_mode():
        logits, cache = runtime.forward(torch.tensor([[1, 3, 4]], dtype=torch.long))
        assert tuple(logits.shape) == (1, config.vocab_size)
        assert len(cache) == config.num_hidden_layers
        assert tuple(cache[0][0].shape) == (1, config.num_key_value_heads, 3, 4)
        next_logits, next_cache = runtime.forward(torch.tensor([[5]], dtype=torch.long), cache)
        full_logits, full_cache = runtime.forward(torch.tensor([[1, 3, 4, 5]], dtype=torch.long))
    assert tuple(next_logits.shape) == (1, config.vocab_size)
    assert tuple(next_cache[0][0].shape) == (1, config.num_key_value_heads, 4, 4)
    assert torch.allclose(next_logits, full_logits, atol=1e-5, rtol=1e-4)
    assert tuple(full_cache[0][0].shape) == (1, config.num_key_value_heads, 4, 4)


def test_mistral_sliding_window_bounds_cache_and_preserves_absolute_rope_positions(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config(
        model_type="mistral",
        architectures=["MistralForCausalLM"],
        sliding_window=3,
    ))
    runtime = NativeLlamaRuntime(config, _tiny_weights(torch), torch, torch.device("cpu"))
    observed_positions = []
    original_rope = runtime._apply_rope

    def capture_rope(values, positions, embeddings=None):
        observed_positions.append(positions.detach().cpu().tolist())
        return original_rope(values, positions, embeddings)

    monkeypatch.setattr(runtime, "_apply_rope", capture_rope)
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    with torch.inference_mode():
        logits, cache = runtime.forward(prompt)
        assert tuple(logits.shape) == (1, config.vocab_size)
        assert tuple(cache[0][0].shape) == (1, config.num_key_value_heads, 3, 4)
        assert tuple(cache[0][1].shape) == (1, config.num_key_value_heads, 3, 4)
        # Compact storage must not retain the original seven-token prefill
        # allocation behind a three-token slice view.
        assert cache[0][0].untyped_storage().nbytes() == cache[0][0].numel() * cache[0][0].element_size()
        assert cache[0][1].untyped_storage().nbytes() == cache[0][1].numel() * cache[0][1].element_size()

        next_logits, next_cache = runtime.forward(
            torch.tensor([[2]], dtype=torch.long),
            cache,
            past_sequence_length=7,
        )
        assert tuple(next_logits.shape) == (1, config.vocab_size)
        assert tuple(next_cache[0][0].shape) == (1, config.num_key_value_heads, 3, 4)
        assert next_cache[0][0].untyped_storage().nbytes() == next_cache[0][0].numel() * next_cache[0][0].element_size()

    # The one-token decode must use absolute position 7, not the cache's
    # truncated length (3). _apply_rope is called once for Q and once for K.
    assert observed_positions[-2:] == [[7], [7]]
    with pytest.raises(NativeEngineError, match="maximum context length"):
        runtime.forward(
            torch.tensor([[3]], dtype=torch.long),
            next_cache,
            past_sequence_length=config.max_position_embeddings,
        )


def test_native_forward_validates_input_tensor_shape_dtype_and_empty_sequences() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config())
    runtime = NativeLlamaRuntime(config, _tiny_weights(torch), torch, torch.device("cpu"))

    with pytest.raises(NativeEngineError, match="must be a tensor"):
        runtime.forward([[1]])
    with pytest.raises(NativeEngineError, match="one prompt sequence"):
        runtime.forward(torch.tensor([1], dtype=torch.long))
    with pytest.raises(NativeEngineError, match="at least one token"):
        runtime.forward(torch.empty((1, 0), dtype=torch.long))
    with pytest.raises(NativeEngineError, match="integer token-ID dtype"):
        runtime.forward(torch.tensor([[1.0]], dtype=torch.float32))

    with torch.inference_mode():
        logits, _ = runtime.forward(torch.tensor([[1, 2]], dtype=torch.int32))
    assert tuple(logits.shape) == (1, config.vocab_size)


@pytest.mark.parametrize("token_ids", [[-1], [8], [True], ["1"]])
def test_native_decoder_rejects_prompt_token_ids_outside_checkpoint_vocabulary(token_ids) -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config())
    runtime = NativeLlamaRuntime(config, _tiny_weights(torch), torch, torch.device("cpu"))
    with pytest.raises(NativeEngineError, match="Prompt token IDs must be integers inside"):
        list(runtime.iter_generate(
            token_ids,
            max_new_tokens=1,
            temperature=0,
        ))


def test_native_decoder_stops_before_emitting_eos_and_does_not_step_past_output_limit() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config(max_position_embeddings=2))
    runtime = NativeLlamaRuntime(config, _tiny_weights(torch), torch, torch.device("cpu"))
    with torch.inference_mode():
        initial_logits, _ = runtime.forward(torch.tensor([[1]], dtype=torch.long))
        eos_id = int(torch.argmax(initial_logits, dim=-1).item())
        assert runtime.generate([1], max_new_tokens=1, temperature=0, eos_token_ids=(eos_id,)) == []
        # With no EOS override, the final token can fill the last context slot;
        # the decoder must not run an extra, out-of-window forward pass.
        output = runtime.generate([1], max_new_tokens=1, temperature=0, eos_token_ids=())
    assert len(output) == 1
    assert runtime.last_generation["completion_tokens"] == 1


def test_complete_native_generate_forwards_min_p_to_shared_decoder() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    runtime = NativeLlamaRuntime(
        NativeLlamaConfig.from_mapping(_config()),
        _tiny_weights(torch),
        torch,
        torch.device("cpu"),
    )
    runtime.generate(
        [1, 3],
        max_new_tokens=1,
        temperature=0.7,
        top_p=0.9,
        min_p=0.2,
        eos_token_ids=(),
    )
    assert runtime.last_generation["min_p"] == 0.2


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("temperature", True),
        ("temperature", float("nan")),
        ("top_p", True),
        ("top_p", float("inf")),
        ("repetition_penalty", True),
        ("repetition_penalty", float("nan")),
    ],
)
def test_native_runtime_rejects_boolean_or_nonfinite_sampling_values(parameter, value) -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    runtime = NativeLlamaRuntime(
        NativeLlamaConfig.from_mapping(_config()),
        _tiny_weights(torch),
        torch,
        torch.device("cpu"),
    )
    options = {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.0}
    options[parameter] = value
    with pytest.raises(NativeEngineError, match="finite number"):
        list(runtime.iter_generate(
            [1, 3],
            max_new_tokens=1,
            eos_token_ids=(),
            **options,
        ))


def test_native_iterator_honors_cancellation_before_prompt_prefill(monkeypatch) -> None:
    from threading import Event

    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config())
    runtime = NativeLlamaRuntime(config, _tiny_weights(torch), torch, torch.device("cpu"))
    cancel_event = Event()
    cancel_event.set()

    def unexpected_forward(input_ids, past_key_values=None, *, past_sequence_length=None, cancel_event=None):
        pytest.fail("A pre-cancelled request must not start model forward execution.")

    monkeypatch.setattr(runtime, "forward", unexpected_forward)
    assert list(runtime.iter_generate(
        [1, 3], max_new_tokens=4, temperature=0, eos_token_ids=(), cancel_event=cancel_event
    )) == []
    assert runtime.last_generation["completion_tokens"] == 0


def test_native_iterator_stops_before_next_token_forward_when_cancelled(monkeypatch) -> None:
    from threading import Event

    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config())
    runtime = NativeLlamaRuntime(config, _tiny_weights(torch), torch, torch.device("cpu"))
    cancel_event = Event()
    original_forward = runtime.forward
    forward_calls = 0

    def counted_forward(input_ids, past_key_values=None, *, past_sequence_length=None, cancel_event=None):
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(
            input_ids, past_key_values, past_sequence_length=past_sequence_length,
            cancel_event=cancel_event,
        )

    monkeypatch.setattr(runtime, "forward", counted_forward)
    iterator = runtime.iter_generate(
        [1, 3], max_new_tokens=4, temperature=0, eos_token_ids=(), cancel_event=cancel_event
    )
    first_token = next(iterator)
    assert 0 <= first_token < config.vocab_size
    assert forward_calls == 1
    cancel_event.set()
    with pytest.raises(StopIteration):
        next(iterator)
    # counted_forward fails if any model forward pass occurs after cancellation.
    assert runtime.last_generation["completion_tokens"] == 1


def _tiny_adapter_file(safetensors_torch, torch, root: Path, *, bad_a_shape: bool = False) -> Path:
    adapter = root / "tiny-lora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "task_type": "CAUSAL_LM", "bias": "none", "r": 1,
        "lora_alpha": 2, "target_modules": ["q_proj"],
    }), encoding="utf-8")
    a_shape = (1, 7) if bad_a_shape else (1, 8)
    safetensors_torch.save_file({
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(a_shape),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones((8, 1)),
    }, str(adapter / "adapter_model.safetensors"))
    return adapter


def test_native_adapter_compatibility_checks_tensor_shapes_and_device_presence(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from axiom.core.native_engine import inspect_native_adapter_compatibility

    root = tmp_path / "trusted"
    model = root / "base"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    safetensors_torch.save_file(_tiny_weights(torch), str(model / "model.safetensors"))
    adapter = _tiny_adapter_file(safetensors_torch, torch, root)

    report = inspect_native_adapter_compatibility(model, adapter, trusted_roots=(root.resolve(),))

    assert report["schema_version"] == 1
    assert report["structurally_compatible"] is True
    assert report["issues"] == []
    assert report["adapter"]["rank"] == 1
    assert report["adapter"]["target_modules"] == ["q_proj"]
    assert report["adapter_tensor_count"] == 2
    assert {device["name"] for device in report["devices"]} == {"CPU", "CUDA", "ROCm", "Metal/MPS", "Intel XPU"}
    cpu = next(device for device in report["devices"] if device["name"] == "CPU")
    assert cpu["available"] is True
    assert cpu["compatible"] is True


def test_native_adapter_compatibility_rejects_projection_shape_mismatch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from axiom.core.native_engine import inspect_native_adapter_compatibility

    root = tmp_path / "trusted"
    model = root / "base"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    safetensors_torch.save_file(_tiny_weights(torch), str(model / "model.safetensors"))
    adapter = _tiny_adapter_file(safetensors_torch, torch, root, bad_a_shape=True)

    report = inspect_native_adapter_compatibility(model, adapter, trusted_roots=(root.resolve(),))

    assert report["structurally_compatible"] is False
    assert any("LoRA A tensor shape" in issue for issue in report["issues"])
    assert next(device for device in report["devices"] if device["name"] == "CPU")["compatible"] is False


def test_native_adapter_delta_is_applied_in_axiom_linear_path() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config())
    weights = _tiny_weights(torch)
    module = "model.layers.0.self_attn.q_proj"
    a = torch.ones((1, 8), dtype=torch.float32)
    b = torch.ones((8, 1), dtype=torch.float32)
    runtime = NativeLlamaRuntime(
        config,
        weights,
        torch,
        torch.device("cpu"),
        adapter={module: {"A": a, "B": b}},
        adapter_scaling={module: 2.0},
    )
    sample = torch.ones((1, 1, 8), dtype=torch.float32)
    actual = runtime._linear(sample, module)
    base = torch.nn.functional.linear(sample, weights[module + ".weight"])
    expected = base + torch.nn.functional.linear(torch.nn.functional.linear(sample, a), b) * 2.0
    assert torch.allclose(actual, expected)


def test_tiny_safetensors_checkpoint_generates_through_axiom_native_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("tokenizers")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
    from axiom.core.engine import generate_local, generate_local_stream

    model_root = tmp_path / "models"
    model_dir = model_root / "tiny-llama"
    model_dir.mkdir(parents=True)
    monkeypatch.setenv("AXIOM_MODEL_ROOTS", str(model_root))
    (model_dir / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text(json.dumps({"add_bos_token": True}), encoding="utf-8")
    tokenizer = Tokenizer(models.WordLevel(
        {"[UNK]": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4, "token5": 5, "token6": 6, "token7": 7},
        unk_token="[UNK]",
    ))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens([
        AddedToken("<s>", special=True),
        AddedToken("</s>", special=True),
        AddedToken("[UNK]", special=True),
    ])
    tokenizer.save(str(model_dir / "tokenizer.json"))
    safetensors_torch.save_file(_tiny_weights(torch), str(model_dir / "model.safetensors"))

    result = generate_local(str(model_dir), "hello world", max_new_tokens=2, temperature=0)

    assert result["engine"] == "axiom-native-engine"
    assert result["execution_backend"] == "pytorch-native-axiom"
    assert result["transformers_generation_used"] is False
    assert result["ollama_used"] is False
    assert result["checkpoint_format"] == "safetensors"
    assert result["prompt_tokens"] == 3
    assert result["completion_tokens"] == 2
    assert result["decoder"] == "axiom-autoregressive-kv-cache"
    assert result["attention_backend"] in {
        "pytorch-scaled-dot-product-attention",
        "pytorch-scaled-dot-product-attention-gqa",
        "pytorch-scaled-dot-product-attention-expanded-gqa",
        "axiom-manual-matmul-attention",
        "axiom-manual-matmul-attention-gqa",
    }
    assert result["prefill_ms"] >= 0
    assert result["decode_ms"] >= 0
    assert result["time_to_first_token_ms"] == result["prefill_ms"]
    assert result["generation_tokens_per_second"] >= 0
    assert result["sampling"] == {
        "temperature": 0,
        "top_p": 0.9,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "seed": None,
    }

    events = list(generate_local_stream(str(model_dir), "hello world", max_new_tokens=2, temperature=0))
    assert events[-1]["type"] == "complete"
    assert events[-1]["result"]["response"] == result["response"]
    assert all(event["type"] in {"token", "telemetry"} for event in events[:-1])
    telemetry_events = [event for event in events if event["type"] == "telemetry"]
    assert telemetry_events
    assert telemetry_events[-1]["metrics"]["completion_tokens"] >= 1
    assert telemetry_events[-1]["metrics"]["prefill_ms"] >= 0
    token_events = [event for event in events[:-1] if event["type"] == "token"]
    assert len(token_events) == events[-1]["result"]["completion_tokens"]
    assert isinstance(result["response"], str)



def test_device_selection_prefers_available_accelerator_and_reports_choice(monkeypatch) -> None:
    from types import SimpleNamespace
    from axiom.core import engine

    rows = [
        {"id": "cuda", "name": "NVIDIA CUDA", "available": False, "reason": "not present", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "rocm", "name": "AMD ROCm", "available": False, "reason": "not present", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "mps", "name": "Apple Metal (MPS)", "available": True, "reason": "available", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "xpu", "name": "Intel XPU", "available": True, "reason": "available", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "cpu", "name": "CPU", "available": True, "reason": "available", "memory_free_gib": None, "memory_total_gib": None},
    ]
    monkeypatch.setattr(engine, "_device_capability_rows", lambda _torch=None: rows)
    fake_torch = SimpleNamespace(device=lambda name: name)

    device, selection = engine._resolve_execution_device(fake_torch, "auto")

    assert device == "mps"
    assert selection["requested"] == "auto"
    assert selection["selected"] == "mps"
    assert "highest-priority available backend" in selection["reason"]


def test_device_selection_rejects_unavailable_or_invalid_explicit_backend(monkeypatch) -> None:
    from types import SimpleNamespace
    from axiom.core.engine import AxiomEngineError, _resolve_execution_device

    rows = [
        {"id": "cuda", "name": "NVIDIA CUDA", "available": False, "reason": "CUDA is not available", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "rocm", "name": "AMD ROCm", "available": False, "reason": "ROCm is not available", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "mps", "name": "Apple Metal (MPS)", "available": False, "reason": "MPS is not available", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "xpu", "name": "Intel XPU", "available": False, "reason": "XPU is not available", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "cpu", "name": "CPU", "available": True, "reason": "available", "memory_free_gib": None, "memory_total_gib": None},
    ]
    monkeypatch.setattr("axiom.core.engine._device_capability_rows", lambda _torch=None: rows)
    fake_torch = SimpleNamespace(device=lambda name: name)

    with pytest.raises(AxiomEngineError, match="unavailable"):
        _resolve_execution_device(fake_torch, "cuda")
    with pytest.raises(ValueError, match="Execution device must be one of"):
        _resolve_execution_device(fake_torch, "network")


def test_process_peak_rss_is_reported_in_device_and_system_telemetry(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace
    from axiom.core import engine

    peaks = iter((8_192, 12_288, 16_384))
    monkeypatch.setattr(engine, "_process_peak_rss_bytes", lambda: next(peaks))
    fake_psutil = SimpleNamespace(
        Process=lambda: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=4_096),
            num_threads=lambda: 1,
            cpu_percent=lambda interval=None: 0.0,
        ),
        cpu_percent=lambda interval=None: 0.0,
        virtual_memory=lambda: SimpleNamespace(
            total=64 * 1024**3,
            available=48 * 1024**3,
            used=16 * 1024**3,
            percent=25.0,
        ),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(engine, "_device_capability_rows", lambda _torch=None: [])
    monkeypatch.setattr(engine, "_LOADED_RUNTIME", None)
    monkeypatch.setattr(engine, "_LOADED_ARCHITECTURE", None)

    before = engine._device_memory_snapshot(None, None)
    after = engine._device_memory_snapshot(None, None, baseline=before)
    assert before["process_peak_rss_bytes"] == 8_192
    assert after["process_peak_rss_bytes"] == 12_288
    assert after["process_peak_rss_delta_bytes"] == 4_096

    telemetry = engine.runtime_telemetry()
    assert telemetry["process"]["peak_rss_bytes"] == 16_384
    assert telemetry["privacy_scope"]["model_paths_included"] is False


def test_auto_device_falls_back_when_accelerator_memory_plan_cannot_fit(monkeypatch, tmp_path: Path) -> None:
    import sys
    from types import SimpleNamespace
    from axiom.core import engine
    from axiom.core.engine import AxiomEngineError

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    fake_torch = SimpleNamespace(device=lambda name: name)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(engine, "_LOADED_RUNTIME", None)
    monkeypatch.setattr(engine, "_LOADED_ARCHITECTURE", None)
    monkeypatch.setattr(engine, "_resolve_model_directory", lambda *_args: model_dir)
    monkeypatch.setattr(engine, "_inspect_model_config", lambda _path: {
        "config": _config(), "model_type": "llama", "architecture": "LlamaForCausalLM",
    })
    monkeypatch.setattr(engine, "_allowed_model_roots", lambda: (tmp_path.resolve(),))
    monkeypatch.setattr(engine, "_device_capability_rows", lambda _torch=None: [
        {"id": "cuda", "name": "NVIDIA CUDA", "available": True, "reason": "available", "memory_free_gib": 0.1, "memory_total_gib": 8.0},
        {"id": "rocm", "name": "AMD ROCm", "available": False, "reason": "unavailable", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "mps", "name": "Apple Metal (MPS)", "available": False, "reason": "unavailable", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "xpu", "name": "Intel XPU", "available": False, "reason": "unavailable", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "cpu", "name": "CPU", "available": True, "reason": "available", "memory_free_gib": None, "memory_total_gib": None},
    ])

    class FakeModel:
        config = SimpleNamespace(max_position_embeddings=128)
        memory_mode = "disk_streaming"

    planner_devices: list[str] = []
    load_devices: list[str] = []

    def fake_plan(_path, _inspected, _torch, device, device_selection, _adapter=None, _precision="auto"):
        selected = str(device)
        planner_devices.append(selected)
        if selected == "cuda":
            raise AxiomEngineError(
                "Automatic memory planning could not find an estimated strategy that fits this device."
            )
        assert device_selection["selected"] == "cpu"
        return "disk_streaming", {
            "requested": "auto",
            "selected": "disk_streaming",
            "reason": "The CPU memory estimate fits.",
            "estimates_are_advisory": True,
        }

    def fake_load(_path, _torch, device, **_kwargs):
        load_devices.append(str(device))
        return FakeModel(), object(), {
            "architecture": "LlamaForCausalLM",
            "model_type": "llama",
            "adapter_directory_name": None,
            "precision": "fp32",
        }

    monkeypatch.setattr(engine, "_choose_memory_mode", fake_plan)
    monkeypatch.setattr("axiom.core.native_engine.load_native_runtime", fake_load)

    _model, _tokenizer, _torch, device, details = engine._load_local_model(
        str(model_dir), None, device_preference="auto", memory_mode="auto"
    )

    assert planner_devices == ["cuda", "cpu"]
    assert load_devices == ["cpu"]
    assert device == "cpu"
    assert details["device_selection"]["selected"] == "cpu"
    assert "memory planner" in details["device_selection"]["reason"]
    assert details["memory_selection"]["selected"] == "disk_streaming"
    assert details["memory_selection"]["fallback_from"] == "cuda"


def test_auto_device_retries_cpu_after_accelerator_oom(monkeypatch, tmp_path: Path) -> None:
    import sys
    from types import SimpleNamespace
    from axiom.core import engine
    from axiom.core.native_engine import NativeEngineError

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    fake_torch = SimpleNamespace(device=lambda name: name)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(engine, "_LOADED_RUNTIME", None)
    monkeypatch.setattr(engine, "_LOADED_ARCHITECTURE", None)
    monkeypatch.setattr(engine, "_resolve_model_directory", lambda *_args: model_dir)
    config = _config()
    shard = model_dir / "model.safetensors"
    shard.write_bytes(b"x" * 8192)
    monkeypatch.setattr(engine, "_inspect_model_config", lambda _path: {
        "config": config, "model_type": "llama", "architecture": "LlamaForCausalLM",
    })
    monkeypatch.setattr("axiom.core.native_engine._resolve_safetensor_files", lambda *_args, **_kwargs: (
        [shard], {"model.embed_tokens.weight": "model.safetensors"}
    ))
    monkeypatch.setattr(engine, "_available_system_memory_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(engine, "_allowed_model_roots", lambda: (tmp_path.resolve(),))
    monkeypatch.setattr(engine, "_device_capability_rows", lambda _torch=None: [
        {"id": "cuda", "name": "NVIDIA CUDA", "available": True, "reason": "available", "memory_free_gib": 0.1, "memory_total_gib": 8.0},
        {"id": "rocm", "name": "AMD ROCm", "available": False, "reason": "unavailable", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "mps", "name": "Apple Metal (MPS)", "available": False, "reason": "unavailable", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "xpu", "name": "Intel XPU", "available": False, "reason": "unavailable", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "cpu", "name": "CPU", "available": True, "reason": "available", "memory_free_gib": None, "memory_total_gib": None},
    ])

    class FakeModel:
        config = SimpleNamespace(max_position_embeddings=128)

    calls: list[str] = []

    def fake_load(_path, _torch, device, **_kwargs):
        calls.append(str(device))
        if str(device) == "cuda":
            raise NativeEngineError("Checkpoint loading ran out of memory on the selected device.")
        return FakeModel(), object(), {
            "architecture": "LlamaForCausalLM",
            "model_type": "llama",
            "adapter_directory_name": None,
        }

    monkeypatch.setattr("axiom.core.native_engine.load_native_runtime", fake_load)
    _model, _tokenizer, _torch, device, details = engine._load_local_model(str(model_dir), None, "auto")

    assert calls == ["cuda", "cpu"]
    assert device == "cpu"
    assert details["device_selection"]["requested"] == "auto"
    assert details["device_selection"]["selected"] == "cpu"
    assert "retried on CPU" in details["device_selection"]["reason"]


def test_explicit_accelerator_does_not_silently_fall_back_after_oom(monkeypatch, tmp_path: Path) -> None:
    import sys
    from types import SimpleNamespace
    from axiom.core import engine
    from axiom.core.native_engine import NativeEngineError

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    fake_torch = SimpleNamespace(device=lambda name: name)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(engine, "_LOADED_RUNTIME", None)
    monkeypatch.setattr(engine, "_LOADED_ARCHITECTURE", None)
    monkeypatch.setattr(engine, "_resolve_model_directory", lambda *_args: model_dir)
    config = _config()
    shard = model_dir / "model.safetensors"
    shard.write_bytes(b"x" * 8192)
    monkeypatch.setattr(engine, "_inspect_model_config", lambda _path: {
        "config": config, "model_type": "llama", "architecture": "LlamaForCausalLM",
    })
    monkeypatch.setattr("axiom.core.native_engine._resolve_safetensor_files", lambda *_args, **_kwargs: (
        [shard], {"model.embed_tokens.weight": "model.safetensors"}
    ))
    monkeypatch.setattr(engine, "_available_system_memory_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(engine, "_allowed_model_roots", lambda: (tmp_path.resolve(),))
    monkeypatch.setattr(engine, "_device_capability_rows", lambda _torch=None: [
        {"id": "cuda", "name": "NVIDIA CUDA", "available": True, "reason": "available", "memory_free_gib": 0.1, "memory_total_gib": 8.0},
        {"id": "rocm", "name": "AMD ROCm", "available": False, "reason": "unavailable", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "mps", "name": "Apple Metal (MPS)", "available": False, "reason": "unavailable", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "xpu", "name": "Intel XPU", "available": False, "reason": "unavailable", "memory_free_gib": None, "memory_total_gib": None},
        {"id": "cpu", "name": "CPU", "available": True, "reason": "available", "memory_free_gib": None, "memory_total_gib": None},
    ])
    calls: list[str] = []

    def fake_load(_path, _torch, device, **_kwargs):
        calls.append(str(device))
        raise NativeEngineError("Checkpoint loading ran out of memory on the selected device.")

    monkeypatch.setattr("axiom.core.native_engine.load_native_runtime", fake_load)
    with pytest.raises(ValueError, match="out of memory"):
        engine._load_local_model(str(model_dir), None, "cuda")
    assert calls == ["cuda"]


def test_layer_streaming_mode_matches_full_cpu_forward_and_offloads_kv_cache() -> None:
    torch = pytest.importorskip("torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    config = NativeLlamaConfig.from_mapping(_config())
    base_weights = _tiny_weights(torch)
    full = NativeLlamaRuntime(
        config, {name: tensor.clone() for name, tensor in base_weights.items()},
        torch, torch.device("cpu"), memory_mode="full",
    )
    streamed = NativeLlamaRuntime(
        config, {name: tensor.clone() for name, tensor in base_weights.items()},
        torch, torch.device("cpu"), memory_mode="layer_streaming",
    )
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)

    full_logits, _full_cache = full.forward(tokens)
    streamed_logits, streamed_cache = streamed.forward(tokens)

    assert torch.allclose(streamed_logits, full_logits, atol=1e-6, rtol=1e-5)
    assert streamed.memory_mode == "layer_streaming"
    assert len(streamed_cache) == config.num_hidden_layers
    assert all(key.device.type == "cpu" and value.device.type == "cpu" for key, value in streamed_cache)
    assert streamed.memory_stats["kv_cache_layer_offloads"] == config.num_hidden_layers
    assert streamed.memory_stats["kv_cache_bytes_offloaded"] > 0
    assert streamed.memory_stats["layer_weight_bytes_staged"] == 0


def test_qwen3_native_loader_applies_query_key_rms_norm(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    root = tmp_path / "trusted"
    model_dir = root / "tiny-qwen3"
    model_dir.mkdir(parents=True)
    config = _config(model_type="qwen3", architectures=["Qwen3ForCausalLM"], rope_theta=1_000_000.0)
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    weights = _tiny_weights(torch)
    weights["model.layers.0.self_attn.q_norm.weight"] = torch.ones(4)
    weights["model.layers.0.self_attn.k_norm.weight"] = torch.ones(4)
    safetensors_torch.save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in weights.items()},
        str(model_dir / "model.safetensors"),
    )
    runtime = NativeLlamaRuntime.from_directory(
        model_dir, torch, torch.device("cpu"), trusted_roots=(root.resolve(),), memory_mode="full",
    )
    logits, cache = runtime.forward(torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert logits.shape == (1, config["vocab_size"])
    assert len(cache) == config["num_hidden_layers"]
    assert runtime.config.qk_norm is True


@pytest.mark.parametrize("dtype_name", ["int64", "float64"])
def test_native_loader_rejects_unsupported_embedding_weight_dtype(
    tmp_path: Path, dtype_name: str
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from axiom.core.native_engine import NativeEngineError, NativeLlamaRuntime

    root = tmp_path / "trusted"
    model_dir = root / "invalid-embedding-dtype"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    weights = _tiny_weights(torch)
    weights["model.embed_tokens.weight"] = weights["model.embed_tokens.weight"].to(
        dtype=getattr(torch, dtype_name)
    )
    safetensors_torch.save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in weights.items()},
        str(model_dir / "model.safetensors"),
    )

    with pytest.raises(NativeEngineError, match="Unsupported embedding weight dtype"):
        NativeLlamaRuntime.from_directory(
            model_dir, torch, torch.device("cpu"), trusted_roots=(root.resolve(),)
        )


@pytest.mark.parametrize(
    "bias_name,shape",
    [
        ("model.layers.0.self_attn.q_proj.bias", (7,)),
        ("model.layers.0.self_attn.k_proj.bias", (5,)),
        ("model.layers.0.mlp.gate_proj.bias", (11,)),
        ("lm_head.bias", (7,)),
    ],
)
def test_native_loader_rejects_projection_bias_shape_mismatches(
    tmp_path: Path, bias_name: str, shape: tuple[int, ...]
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from axiom.core.native_engine import NativeEngineError, NativeLlamaRuntime

    root = tmp_path / "trusted"
    model_dir = root / "bad-bias"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    weights = _tiny_weights(torch)
    weights[bias_name] = torch.zeros(shape, dtype=torch.float32)
    safetensors_torch.save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in weights.items()},
        str(model_dir / "model.safetensors"),
    )

    with pytest.raises(NativeEngineError, match="shape inconsistent with config.json"):
        NativeLlamaRuntime.from_directory(
            model_dir, torch, torch.device("cpu"), trusted_roots=(root.resolve(),)
        )


def test_native_loader_accepts_qwen2_query_key_value_biases(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    root = tmp_path / "trusted"
    model_dir = root / "qwen2-with-biases"
    model_dir.mkdir(parents=True)
    config = _config(model_type="qwen2", architectures=["Qwen2ForCausalLM"])
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    weights = _tiny_weights(torch)
    weights["model.layers.0.self_attn.q_proj.bias"] = torch.zeros(8)
    weights["model.layers.0.self_attn.k_proj.bias"] = torch.zeros(4)
    weights["model.layers.0.self_attn.v_proj.bias"] = torch.zeros(4)
    safetensors_torch.save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in weights.items()},
        str(model_dir / "model.safetensors"),
    )

    runtime = NativeLlamaRuntime.from_directory(
        model_dir, torch, torch.device("cpu"), trusted_roots=(root.resolve(),)
    )
    with torch.inference_mode():
        logits, _ = runtime.forward(torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert tuple(logits.shape) == (1, config["vocab_size"])
    assert runtime.weights["model.layers.0.self_attn.q_proj.bias"].shape == (8,)
    assert runtime.weights["model.layers.0.self_attn.k_proj.bias"].shape == (4,)
    assert runtime.weights["model.layers.0.self_attn.v_proj.bias"].shape == (4,)


def test_native_loader_explicit_fp16_precision_casts_model_weights(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    root = tmp_path / "trusted"
    model_dir = root / "tiny-fp32"
    model_dir.mkdir(parents=True)
    config = _config()
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    safetensors_torch.save_file(
        {name: tensor.float().detach().cpu().contiguous() for name, tensor in _tiny_weights(torch).items()},
        str(model_dir / "model.safetensors"),
    )
    runtime = NativeLlamaRuntime.from_directory(
        model_dir, torch, torch.device("cpu"), trusted_roots=(root.resolve(),),
        memory_mode="full", precision="fp16",
    )
    assert runtime.dtype == torch.float16
    assert all(tensor.dtype == torch.float16 for tensor in runtime.weights.values())


def test_native_loader_rejects_unsupported_precision(tmp_path: Path) -> None:
    from axiom.core.native_engine import NativeEngineError

    with pytest.raises(NativeEngineError, match="Precision must be auto, fp32, fp16, or bf16"):
        from axiom.core.native_engine import NativeLlamaRuntime
        NativeLlamaRuntime.from_directory(
            tmp_path, object(), object(), trusted_roots=(tmp_path.resolve(),), precision="int4",
        )


def test_disk_streaming_reads_layer_weights_on_demand_and_matches_full_forward(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from axiom.core.native_engine import NativeLlamaRuntime

    root = tmp_path / "trusted"
    model_dir = root / "tiny-llama"
    model_dir.mkdir(parents=True)
    config = _config()
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    safetensors_torch.save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in _tiny_weights(torch).items()},
        str(model_dir / "model.safetensors"),
    )

    full = NativeLlamaRuntime.from_directory(
        model_dir, torch, torch.device("cpu"), trusted_roots=(root.resolve(),), memory_mode="full",
    )
    streamed = NativeLlamaRuntime.from_directory(
        model_dir, torch, torch.device("cpu"), trusted_roots=(root.resolve(),), memory_mode="disk_streaming",
    )
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)

    expected, _ = full.forward(tokens)
    actual, cache = streamed.forward(tokens)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert streamed.memory_mode == "disk_streaming"
    assert streamed.layer_weight_sources
    assert not any(name.startswith("model.layers.") for name in streamed.weights)
    assert streamed.memory_stats["disk_layer_weight_loads"] > 0
    assert streamed.memory_stats["disk_layer_weight_bytes_read"] > 0
    assert streamed.memory_stats["layer_weight_bytes_released"] > 0
    assert all(key.device.type == "cpu" and value.device.type == "cpu" for key, value in cache)
    assert streamed.memory_stats["kv_cache_layer_offloads"] == config["num_hidden_layers"]


def test_native_benchmark_excludes_warmups_and_never_returns_prompt_or_response(monkeypatch) -> None:
    from axiom.core import engine as engine_module

    calls = []

    def fake_generate(model_path, prompt, **kwargs):
        calls.append({"model_path": model_path, "prompt": prompt, **kwargs})
        measured_run = len(calls) - 1
        elapsed = [999.0, 10.0, 20.0, 30.0][measured_run]
        return {
            "model": "tiny-model",
            "architecture": "LlamaForCausalLM",
            "model_type": "llama",
            "device": "cpu",
            "precision": "fp32",
            "memory_mode": "full",
            "checkpoint_format": "safetensors",
            "elapsed_ms": elapsed,
            "prefill_ms": elapsed / 4,
            "decode_ms": elapsed / 2,
            "time_to_first_token_ms": elapsed / 10,
            "generation_tokens_per_second": [1.0, 200.0, 250.0, 300.0][measured_run],
            "prompt_tokens": 5,
            "completion_tokens": [99, 2, 4, 6][measured_run],
            "stop_reason": "max_new_tokens",
            "response": "PRIVATE_GENERATED_RESPONSE",
            "device_memory": {"peak_allocated_bytes": 1234},
            "memory_selection": {"selected": "full"},
        }

    monkeypatch.setattr(engine_module, "generate_local", fake_generate)
    report = engine_module.benchmark_local(
        "/trusted/tiny-model",
        "PRIVATE_BENCHMARK_PROMPT",
        warmup_runs=1,
        iterations=3,
        max_new_tokens=8,
    )

    assert len(calls) == 4
    assert all(call["prompt"] == "PRIVATE_BENCHMARK_PROMPT" for call in calls)
    assert all(call["_lock_held"] is True for call in calls)
    assert report["warmup_runs_requested"] == report["warmup_runs_completed"] == 1
    assert report["warmups_excluded"] is True
    assert report["iterations_requested"] == report["iterations_completed"] == 3
    assert report["metrics"]["median_elapsed_ms"] == 20.0
    assert report["metrics"]["p95_elapsed_ms"] == 30.0
    assert report["metrics"]["completion_tokens"] == 12
    assert report["metrics"]["prompt_tokens"] == 15
    assert report["metrics"]["aggregate_tokens_per_second"] == 200.0
    assert [item["completion_tokens"] for item in report["runs"]] == [2, 4, 6]
    serialized = json.dumps(report)
    assert "PRIVATE_BENCHMARK_PROMPT" not in serialized
    assert "PRIVATE_GENERATED_RESPONSE" not in serialized
    assert "/trusted/tiny-model" not in serialized
    assert report["privacy"] == {
        "prompt_returned": False,
        "response_returned": False,
        "results_persisted": False,
        "cloud_provider_used": False,
    }


def test_native_benchmark_rejects_unbounded_work_before_generation(monkeypatch) -> None:
    from axiom.core import engine as engine_module

    monkeypatch.setattr(
        engine_module,
        "generate_local",
        lambda *_args, **_kwargs: pytest.fail("generation must not start for invalid benchmark bounds"),
    )
    with pytest.raises(ValueError, match="Measured iterations"):
        engine_module.benchmark_local("model", "hello", iterations=6)
    with pytest.raises(ValueError, match="output length"):
        engine_module.benchmark_local("model", "hello", max_new_tokens=129)


def _planner_fixture(tmp_path: Path, monkeypatch, *, system_memory: int | None):
    from types import SimpleNamespace
    import shutil

    from axiom.core import engine as engine_module

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard = model_dir / "model.safetensors"
    shard.write_bytes(b"x" * 8192)
    monkeypatch.setattr(
        "axiom.core.native_engine._resolve_safetensor_files",
        lambda *_args, **_kwargs: ([shard], {"model.embed_tokens.weight": "model.safetensors"}),
    )
    monkeypatch.setattr(engine_module, "_available_system_memory_bytes", lambda: system_memory)
    monkeypatch.setattr(engine_module, "_allowed_model_roots", lambda: (tmp_path.resolve(),))
    monkeypatch.setattr(
        engine_module,
        "_device_capability_rows",
        lambda _torch: [{"id": "cpu", "memory_free_gib": None, "memory_total_gib": None}],
    )
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: SimpleNamespace(free=10 * 1024**3))
    return engine_module, model_dir, {"config": _config()}


def test_mistral_sliding_window_memory_estimate_bounds_retained_cache_and_accounts_for_prefill(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from axiom.core import engine as engine_module
    from axiom.core.native_engine import NativeLlamaConfig

    module, model_dir, inspected = _planner_fixture(
        tmp_path, monkeypatch, system_memory=8 * 1024**3
    )
    inspected["config"] = _config(
        model_type="mistral",
        architectures=["MistralForCausalLM"],
        sliding_window=3,
        max_position_embeddings=128,
    )
    mode, selection = module._choose_memory_mode(
        model_dir,
        inspected,
        object(),
        SimpleNamespace(type="cpu"),
        {"selected": "cpu"},
    )

    config = NativeLlamaConfig.from_mapping(inspected["config"])
    context = selection["estimated_context_tokens"]
    cache_context = min(context, config.sliding_window)
    dtype_bytes = selection["estimated_dtype_bytes"]
    head_dim = config.hidden_size // config.num_attention_heads
    expected_cache_bytes = (
        2
        * config.num_hidden_layers
        * config.num_key_value_heads
        * head_dim
        * cache_context
        * dtype_bytes
    )
    expected_prefill_bytes = (
        2 * config.num_key_value_heads * head_dim * context * dtype_bytes
    )
    unbounded_cache_bytes = (
        2
        * config.num_hidden_layers
        * config.num_key_value_heads
        * head_dim
        * context
        * dtype_bytes
    )

    assert mode == "full"
    assert context > cache_context
    assert selection["estimated_kv_cache_context_tokens"] == cache_context
    assert selection["estimated_kv_cache_bytes"] == expected_cache_bytes
    assert selection["estimated_kv_prefill_temporary_bytes"] == expected_prefill_bytes
    assert selection["estimated_kv_cache_bytes"] < unbounded_cache_bytes


def test_mistral_estimator_bounds_retained_kv_cache_and_reports_prefill_temporary(
    tmp_path: Path, monkeypatch
) -> None:
    from axiom.core import engine as engine_module

    model_dir = tmp_path / "mistral-model"
    model_dir.mkdir()
    config = _config(
        model_type="mistral",
        architectures=["MistralForCausalLM"],
        sliding_window=3,
        max_position_embeddings=32,
    )
    monkeypatch.setattr(engine_module, "_resolve_model_directory", lambda *_args: model_dir)
    monkeypatch.setattr(engine_module, "_inspect_model_config", lambda *_args: {
        "config": config,
        "architecture": "MistralForCausalLM",
    })

    report = engine_module.estimate_native_quantization(
        str(model_dir), context_length=24, batch_size=2
    )
    candidate = next(item for item in report["candidates"] if item["format"] == "bf16")
    config_obj = NativeLlamaConfig.from_mapping(config)
    head_dim = config_obj.hidden_size // config_obj.num_attention_heads
    gib = 1024**3
    expected_cache = (
        2
        * config_obj.num_hidden_layers
        * config_obj.num_key_value_heads
        * head_dim
        * 3
        * 2
        * 2
    )
    expected_prefill = 2 * config_obj.num_key_value_heads * head_dim * 24 * 2 * 2

    assert report["assumptions"]["kv_cache_context_length"] == 3
    assert report["assumptions"]["kv_prefill_temporary_context_length"] == 24
    assert candidate["estimated_kv_cache_gib"] == round(expected_cache / gib, 9)
    assert candidate["estimated_kv_prefill_temporary_gib"] == round(expected_prefill / gib, 9)


def test_non_sliding_estimator_does_not_double_count_prefill_kv(
    tmp_path: Path, monkeypatch
) -> None:
    from axiom.core import engine as engine_module

    model_dir = tmp_path / "llama-model"
    model_dir.mkdir()
    config = _config(max_position_embeddings=32)
    monkeypatch.setattr(engine_module, "_resolve_model_directory", lambda *_args: model_dir)
    monkeypatch.setattr(engine_module, "_inspect_model_config", lambda *_args: {
        "config": config,
        "architecture": "LlamaForCausalLM",
    })

    report = engine_module.estimate_native_quantization(
        str(model_dir), context_length=24, batch_size=2
    )
    candidate = next(item for item in report["candidates"] if item["format"] == "bf16")

    assert report["assumptions"]["kv_cache_context_length"] == 24
    assert report["assumptions"]["kv_prefill_temporary_context_length"] == 0
    assert candidate["estimated_kv_prefill_temporary_gib"] == 0.0


def test_auto_memory_planner_selects_full_mode_when_cpu_ram_is_sufficient(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from axiom.core import engine as engine_module

    memory = 8 * 1024**3
    module, model_dir, inspected = _planner_fixture(tmp_path, monkeypatch, system_memory=memory)
    mode, selection = module._choose_memory_mode(
        model_dir, inspected, object(), SimpleNamespace(type="cpu"), {"selected": "cpu"}
    )
    assert mode == "full"
    assert selection["selected"] == "full"
    assert selection["available_system_ram_bytes"] == memory
    assert selection["estimates_are_advisory"] is True


def test_auto_memory_planner_counts_adapter_weights(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from axiom.core import engine as engine_module

    module, model_dir, inspected = _planner_fixture(tmp_path, monkeypatch, system_memory=8 * 1024**3)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"x" * 1024)
    mode, selection = module._choose_memory_mode(
        model_dir, inspected, object(), SimpleNamespace(type="cpu"), {"selected": "cpu"}, adapter_dir,
    )
    assert mode == "full"
    assert selection["estimated_adapter_weight_bytes"] == 1024
    assert selection["estimated_total_weight_bytes"] == selection["estimated_checkpoint_weight_bytes"] + 1024


def test_auto_memory_planner_fails_closed_when_cpu_ram_is_unknown(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from axiom.core import engine as engine_module

    module, model_dir, inspected = _planner_fixture(tmp_path, monkeypatch, system_memory=None)
    with pytest.raises(engine_module.AxiomEngineError, match="available system RAM could not be measured"):
        module._choose_memory_mode(
            model_dir, inspected, object(), SimpleNamespace(type="cpu"), {"selected": "cpu"}
        )


def test_auto_memory_planner_does_not_choose_streaming_without_memory_measurements(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from axiom.core import engine as engine_module

    module, model_dir, inspected = _planner_fixture(tmp_path, monkeypatch, system_memory=None)
    monkeypatch.setattr(
        module,
        "_device_capability_rows",
        lambda _torch: [{"id": "cuda", "memory_free_gib": None, "memory_total_gib": None}],
    )
    with pytest.raises(engine_module.AxiomEngineError, match="neither free accelerator memory nor available system RAM"):
        module._choose_memory_mode(
            model_dir, inspected, object(), SimpleNamespace(type="cuda"), {"selected": "cuda"}
        )


def test_qwen2_prompt_uses_chatml_when_template_metadata_is_missing(tmp_path: Path) -> None:
    from axiom.core.engine import _format_prompt_for_native_model

    class TinyTokenizer:
        def token_to_id(self, token: str):
            return None

    formatted, source = _format_prompt_for_native_model(
        "What is 2+2?", tmp_path, "qwen2", TinyTokenizer()
    )
    assert formatted == "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    assert source == "qwen2-chatml-fallback"


def test_native_prompt_renders_checkpoint_chat_template_in_sandbox(tmp_path: Path) -> None:
    pytest.importorskip("jinja2")
    from axiom.core.engine import _format_prompt_for_native_model

    template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": template,
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
    }), encoding="utf-8")

    class TinyTokenizer:
        def token_to_id(self, token: str):
            return 1 if token in {"<|bos|>", "<|eos|>"} else None

    formatted, source = _format_prompt_for_native_model(
        "Hello", tmp_path, "llama", TinyTokenizer(),
        chat_messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ],
    )
    assert formatted == "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    assert source == "tokenizer-config-template"


def test_qwen2_chat_fallback_preserves_all_turns(tmp_path: Path) -> None:
    from axiom.core.engine import _format_prompt_for_native_model

    class TinyTokenizer:
        def token_to_id(self, token: str):
            return None

    formatted, source = _format_prompt_for_native_model(
        "last question", tmp_path, "qwen2", TinyTokenizer(),
        chat_messages=[
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "last question"},
        ],
    )
    assert formatted == (
        "<|im_start|>system\nYou are concise.<|im_end|>\n"
        "<|im_start|>user\nFirst question<|im_end|>\n"
        "<|im_start|>assistant\nFirst answer<|im_end|>\n"
        "<|im_start|>user\nlast question<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert source == "qwen2-chatml-fallback"


def test_native_prompt_rejects_invalid_chat_template_json(tmp_path: Path) -> None:
    from axiom.core.engine import _format_prompt_for_native_model, AxiomEngineError

    (tmp_path / "tokenizer_config.json").write_text("{bad", encoding="utf-8")

    class TinyTokenizer:
        def token_to_id(self, token: str):
            return None

    with pytest.raises(AxiomEngineError, match="chat template"):
        _format_prompt_for_native_model("Hello", tmp_path, "llama", TinyTokenizer())



def test_native_context_trims_old_history_before_token_truncation(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace
    from axiom.core import engine

    class TokenizerFixture:
        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(ids=[1] * len(text))

        def decode(self, token_ids, skip_special_tokens=True):
            return "answer" if token_ids else ""

    class RuntimeFixture:
        config = SimpleNamespace(max_position_embeddings=64, bos_token_id=None, eos_token_ids=())
        last_attention_backend = "axiom-manual-matmul-attention"
        memory_mode = "full"
        memory_stats = {}

        def __init__(self):
            self.last_generation = {}
            self.prompt_ids = []

        def iter_generate(self, prompt_token_ids, **_kwargs):
            self.prompt_ids = list(prompt_token_ids)
            try:
                yield 101
            finally:
                self.last_generation.update({
                    "prompt_tokens": len(prompt_token_ids),
                    "completion_tokens": 1,
                    "decoder": "axiom-autoregressive-kv-cache",
                    "prefill_ms": 1.0,
                    "decode_ms": 1.0,
                    "time_to_first_token_ms": 1.0,
                    "generation_tokens_per_second": 500.0,
                    "stop_reason": "max_new_tokens",
                })

    runtime = RuntimeFixture()
    tokenizer = TokenizerFixture()
    monkeypatch.setattr(engine, "_load_local_model", lambda *_args, **_kwargs: (
        runtime, tokenizer, None, "cpu", {
            "model_directory_name": "fixture",
            "model_directory_path": str(tmp_path),
            "adapter_directory_name": None,
            "architecture": "LlamaForCausalLM",
            "model_type": "llama",
            "add_bos_token": False,
            "device_selection": {"requested": "cpu", "selected": "cpu", "reason": "test"},
        }
    ))

    events = list(engine.generate_local_stream(
        str(tmp_path), "new ask", max_new_tokens=2, temperature=0,
        chat_messages=[
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "old user history " * 5},
            {"role": "assistant", "content": "old assistant history " * 5},
            {"role": "user", "content": "new ask"},
        ],
        include_token_events=False,
    ))
    result = events[-1]["result"]
    context = result["context_management"]
    assert context["messages_dropped_to_fit_context"] == 2
    assert context["token_truncation_applied"] is False
    assert context["prompt_tokens_used"] <= 62
    assert context["original_message_count"] == 4
    assert result["prompt_tokens"] == context["prompt_tokens_used"]


def test_engine_admission_wait_is_bounded(monkeypatch) -> None:
    from axiom.core import engine as engine_module

    monkeypatch.setattr(engine_module, "_MAX_ENGINE_QUEUE_WAIT_SECONDS", 0.01)
    assert engine_module._ENGINE_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(engine_module.AxiomEngineError, match="busy with another request"):
            with engine_module._engine_request_lock():
                pytest.fail("Should not enter while another request owns the engine")
    finally:
        engine_module._ENGINE_LOCK.release()


def test_engine_admission_wait_honors_cancellation(monkeypatch) -> None:
    from threading import Event
    from axiom.core import engine as engine_module

    cancelled = Event()
    cancelled.set()
    with pytest.raises(engine_module.AxiomEngineError, match="cancelled while waiting"):
        with engine_module._engine_request_lock(cancelled):
            pytest.fail("Cancelled request must not acquire the engine lock")
