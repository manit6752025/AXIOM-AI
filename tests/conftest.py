import pytest
import re


@pytest.fixture(autouse=True)
def _plain_cli_output(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("TERMINAL_WIDTH", "200")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("PY_COLORS", raising=False)
    try:
        import typer.rich_utils as rich_utils

        monkeypatch.setattr(rich_utils, "MAX_WIDTH", 200, raising=False)
        monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", False, raising=False)
    except ImportError:
        pass


@pytest.fixture
def plain():
    def _plain(text: str) -> str:
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)
        text = re.sub(r"[│╭╮╰╯─┃┏┓┗┛━]", " ", text)
        return " ".join(text.split())

    return _plain


@pytest.fixture
def tiny_local_transformer_checkpoints(tmp_path, monkeypatch):
    """Create tiny CPU-only BERT safetensors checkpoints under a trusted local model root."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    required = ("BertConfig", "BertModel", "BertForSequenceClassification", "BertTokenizerFast")
    if any(not hasattr(transformers, name) for name in required):
        pytest.skip("The installed Transformers build does not expose the BERT CPU smoke-test classes.")

    monkeypatch.chdir(tmp_path)
    model_root = tmp_path / "models"
    model_root.mkdir()
    vocab = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "axiom", "local", "runtime", "embedding", "retrieval",
        "query", "document", "ranking", "support", "backup",
        "plugin", "model", "text", "semantic", "safe",
    ]
    vocab_path = model_root / "vocab.txt"
    vocab_path.write_text("\\n".join(vocab) + "\\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)

    embedding_dir = model_root / "tiny-embedding"
    embedding_dir.mkdir()
    embedding_config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        pad_token_id=0,
    )
    embedding_model = transformers.BertModel(embedding_config)
    embedding_model.save_pretrained(embedding_dir, safe_serialization=True)
    tokenizer.save_pretrained(embedding_dir)

    reranker_dir = model_root / "tiny-reranker"
    reranker_dir.mkdir()
    reranker_config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        pad_token_id=0,
        num_labels=2,
    )
    reranker_model = transformers.BertForSequenceClassification(reranker_config)
    reranker_model.save_pretrained(reranker_dir, safe_serialization=True)
    tokenizer.save_pretrained(reranker_dir)

    assert (embedding_dir / "model.safetensors").is_file()
    assert (reranker_dir / "model.safetensors").is_file()
    assert not (embedding_dir / "pytorch_model.bin").exists()
    assert not (reranker_dir / "pytorch_model.bin").exists()
    return {"embedding": embedding_dir, "reranker": reranker_dir}
