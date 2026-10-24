from __future__ import annotations

import json
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


@pytest.mark.parametrize(
    "overrides,detail",
    [
        ({"model_type": "gemma3"}, "supports llama, mistral, qwen2, and qwen3"),
        ({"hidden_size": 7}, "even-sized attention heads"),
        ({"rope_scaling": {"rope_type": "yarn", "factor": 2.0}}, "RoPE scaling"),
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
