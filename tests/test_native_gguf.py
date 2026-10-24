from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _weights(torch, vocab_size=8, hidden=8, intermediate=12):
    return {
        "model.embed_tokens.weight": torch.randn(vocab_size, hidden) * 0.02,
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": torch.randn(vocab_size, hidden) * 0.02,
        "model.layers.0.input_layernorm.weight": torch.ones(hidden),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(hidden),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden, hidden) * 0.02,
        "model.layers.0.self_attn.k_proj.weight": torch.randn(hidden // 2, hidden) * 0.02,
        "model.layers.0.self_attn.v_proj.weight": torch.randn(hidden // 2, hidden) * 0.02,
        "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden, hidden) * 0.02,
        "model.layers.0.mlp.gate_proj.weight": torch.randn(intermediate, hidden) * 0.02,
        "model.layers.0.mlp.up_proj.weight": torch.randn(intermediate, hidden) * 0.02,
        "model.layers.0.mlp.down_proj.weight": torch.randn(hidden, intermediate) * 0.02,
    }


def _name_map():
    return {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
        "model.layers.0.input_layernorm.weight": "blk.0.attn_norm.weight",
        "model.layers.0.post_attention_layernorm.weight": "blk.0.ffn_norm.weight",
        "model.layers.0.self_attn.q_proj.weight": "blk.0.attn_q.weight",
        "model.layers.0.self_attn.k_proj.weight": "blk.0.attn_k.weight",
        "model.layers.0.self_attn.v_proj.weight": "blk.0.attn_v.weight",
        "model.layers.0.self_attn.o_proj.weight": "blk.0.attn_output.weight",
        "model.layers.0.mlp.gate_proj.weight": "blk.0.ffn_gate.weight",
        "model.layers.0.mlp.up_proj.weight": "blk.0.ffn_up.weight",
        "model.layers.0.mlp.down_proj.weight": "blk.0.ffn_down.weight",
    }


def _write_gguf(model_dir: Path, torch, *, quantized=False, embedded_hf_json=True, hidden=8, intermediate=12):
    import gguf
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import quantize
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    model_dir.mkdir(parents=True, exist_ok=True)
    vocab_size = 8
    weights = _weights(torch, vocab_size, hidden, intermediate)
    if hidden != 8:
        weights = {
            "model.embed_tokens.weight": torch.randn(vocab_size, hidden) * 0.02,
            "model.norm.weight": torch.ones(hidden),
            "lm_head.weight": torch.randn(vocab_size, hidden) * 0.02,
            "model.layers.0.input_layernorm.weight": torch.ones(hidden),
            "model.layers.0.post_attention_layernorm.weight": torch.ones(hidden),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden, hidden) * 0.02,
            "model.layers.0.self_attn.k_proj.weight": torch.randn(hidden // 2, hidden) * 0.02,
            "model.layers.0.self_attn.v_proj.weight": torch.randn(hidden // 2, hidden) * 0.02,
            "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden, hidden) * 0.02,
            "model.layers.0.mlp.gate_proj.weight": torch.randn(intermediate, hidden) * 0.02,
            "model.layers.0.mlp.up_proj.weight": torch.randn(intermediate, hidden) * 0.02,
            "model.layers.0.mlp.down_proj.weight": torch.randn(hidden, intermediate) * 0.02,
        }
    writer = gguf.GGUFWriter(model_dir / "tiny.gguf", "llama")
    writer.add_context_length(32)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_block_count(1)
    writer.add_head_count(2 if hidden == 8 else 4)
    writer.add_head_count_kv(1 if hidden == 8 else 2)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_freq_base(10000.0)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")
    tokens = ["<unk>", "a", "b", "c", "d", "e", "f", "<eos>"]
    writer.add_token_list(tokens)
    writer.add_token_types([2, 1, 1, 1, 1, 1, 1, 3])
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(7)
    tokenizer = Tokenizer(models.WordLevel({token: index for index, token in enumerate(tokens)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    if embedded_hf_json:
        writer.add_string("tokenizer.huggingface.json", tokenizer.to_str())
    for native_name, value in weights.items():
        array = value.detach().cpu().numpy().astype(np.float32)
        if quantized and native_name == "model.layers.0.self_attn.q_proj.weight":
            encoded = quantize(array, GGMLQuantizationType.Q4_0)
            writer.add_tensor(_name_map()[native_name], encoded, raw_shape=array.shape, raw_dtype=GGMLQuantizationType.Q4_0)
        else:
            writer.add_tensor(_name_map()[native_name], array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_native_gguf_loads_and_generates_with_axiom_runtime(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("gguf")
    pytest.importorskip("tokenizers")
    from axiom.core.native_engine import load_native_runtime

    root = tmp_path / "trusted"
    model_dir = root / "tiny-gguf"
    _write_gguf(model_dir, torch)
    runtime, tokenizer, details = load_native_runtime(
        model_dir, torch, torch.device("cpu"), trusted_roots=(root.resolve(),), precision="fp32"
    )
    assert details["checkpoint_format"] == "gguf"
    assert details["execution_backend"] == "pytorch-native-axiom"
    assert details["ollama_used"] is False
    assert details["transformers_generation_used"] is False
    assert tokenizer.encode("a").ids == [1]
    output = runtime.generate([0, 1], max_new_tokens=2, temperature=0, eos_token_ids=())
    assert len(output) == 2
    assert all(0 <= token_id < runtime.config.vocab_size for token_id in output)


def test_native_gguf_embedded_bpe_tokenizer_preserves_token_ids(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("gguf")
    from axiom.core.native_gguf import load_native_gguf_runtime

    root = tmp_path / "trusted"
    model_dir = root / "embedded-bpe"
    _write_gguf(model_dir, torch, embedded_hf_json=False)
    runtime, tokenizer, details = load_native_gguf_runtime(
        model_dir, torch, torch.device("cpu"), trusted_roots=(root.resolve(),), precision="fp32"
    )
    assert details["tokenizer_source"] == "tokenizer.ggml.tokens"
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("a") == 1
    assert runtime.config.vocab_size == 8


def test_native_gguf_quantized_source_dequantizes_layers_on_demand(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("gguf")
    from axiom.core.native_engine import load_native_runtime

    root = tmp_path / "trusted"
    model_dir = root / "quantized-gguf"
    _write_gguf(model_dir, torch, quantized=True, hidden=64, intermediate=64)
    runtime, _tokenizer, details = load_native_runtime(
        model_dir, torch, torch.device("cpu"), trusted_roots=(root.resolve()),
        memory_mode="disk_streaming", precision="fp32",
    )
    layer_weight = "model.layers.0.self_attn.q_proj.weight"
    assert layer_weight not in runtime.weights
    assert layer_weight in runtime.layer_weight_loaders
    assert details["gguf_low_bit_kernels_used"] is False
    with torch.inference_mode():
        logits, _ = runtime.forward(torch.tensor([[0, 1]], dtype=torch.long))
    assert tuple(logits.shape) == (1, runtime.config.vocab_size)
    assert layer_weight not in runtime.weights
    assert runtime.memory_stats["disk_layer_weight_loads"] > 0


def test_native_gguf_rejects_unsupported_architecture(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("gguf")
    import gguf
    from axiom.core.native_gguf import inspect_gguf_checkpoint

    model_dir = tmp_path / "unsupported"
    model_dir.mkdir()
    writer = gguf.GGUFWriter(model_dir / "unsupported.gguf", "gemma3")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with pytest.raises(ValueError, match="limited to dense llama, mistral, qwen2 and qwen3"):
        inspect_gguf_checkpoint(model_dir, trusted_roots=(tmp_path.resolve(),))
