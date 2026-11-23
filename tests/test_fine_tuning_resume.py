from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from axiom.core import fine_tuning


class FakeTensor:
    def to(self, _device):
        return self

    def clone(self):
        return FakeTensor()

    def __eq__(self, _other):
        return FakeTensor()

    def __ne__(self, _other):
        return FakeTensor()

    def __setitem__(self, _key, _value):
        return None

    def any(self):
        return self

    def item(self):
        return True

    def cpu(self):
        return self


class FakeLoss:
    def detach(self):
        return self

    def item(self):
        return 0.25

    def backward(self):
        return None


class FakeParameter:
    requires_grad = True

    @staticmethod
    def numel():
        return 16


class FakeLinear:
    pass


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(use_cache=True)
        self.parameter = FakeParameter()

    def named_modules(self):
        return [("model.q_proj", FakeLinear())]

    def parameters(self):
        return [self.parameter]

    def to(self, _device):
        return self

    def train(self):
        return None

    def __call__(self, **_kwargs):
        return SimpleNamespace(loss=FakeLoss())

    def save_pretrained(self, destination, *, safe_serialization):
        assert safe_serialization is True
        folder = Path(destination)
        (folder / "adapter_config.json").write_text('{"r":4}', encoding="utf-8")
        (folder / "adapter_model.safetensors").write_bytes(b"adapter-weights")


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = None

    def __call__(self, *_args, **_kwargs):
        return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

    def save_pretrained(self, destination):
        (Path(destination) / "tokenizer.json").write_text("{}", encoding="utf-8")


class FakeOptimizer:
    def __init__(self, _parameters, *, lr):
        self.learning_rate = lr

    def zero_grad(self, **_kwargs):
        return None

    def step(self):
        return None

    def state_dict(self):
        return {"state": {}, "param_groups": []}

    def load_state_dict(self, _state):
        return None


class FakeDevice:
    def __init__(self, name):
        self.type = name

    def __str__(self):
        return self.type


class FakeCuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0

    @staticmethod
    def manual_seed_all(_seed):
        return None


class FakeTorch(ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.nn = SimpleNamespace(
            Linear=FakeLinear,
            utils=SimpleNamespace(clip_grad_norm_=lambda *_args, **_kwargs: None),
        )
        self.optim = SimpleNamespace(AdamW=FakeOptimizer)
        self.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
        self.cuda = FakeCuda()

    @staticmethod
    def device(name):
        return FakeDevice(name)

    @staticmethod
    def manual_seed(_seed):
        return None

    @staticmethod
    def ones_like(_tensor):
        return FakeTensor()

    @staticmethod
    def get_rng_state():
        return FakeTensor()

    @staticmethod
    def set_rng_state(_state):
        return None

    @staticmethod
    def save(payload, destination):
        with Path(destination).open("wb") as stream:
            pickle.dump(payload, stream)

    @staticmethod
    def load(stream, *, map_location, weights_only):
        assert map_location == "cpu"
        assert weights_only is True
        return pickle.load(stream)


class FakePeftModel:
    @staticmethod
    def from_pretrained(model, checkpoint, *, is_trainable, local_files_only):
        assert Path(checkpoint).is_dir()
        assert is_trainable is True
        assert local_files_only is True
        return model


class FakeAutoModel:
    @staticmethod
    def from_pretrained(_model_path, *, local_files_only, trust_remote_code):
        assert local_files_only is True
        assert trust_remote_code is False
        return FakeModel()


class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(_model_path, *, local_files_only, trust_remote_code):
        assert local_files_only is True
        assert trust_remote_code is False
        return FakeTokenizer()


def _install_training_stubs(monkeypatch):
    torch = FakeTorch()
    peft = ModuleType("peft")
    peft.LoraConfig = lambda **kwargs: kwargs
    peft.PeftModel = FakePeftModel
    peft.TaskType = SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
    peft.get_peft_model = lambda model, _config: model
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = FakeAutoModel
    transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return torch


def test_local_training_resumes_verified_adapter_optimizer_and_sampling_state(tmp_path, monkeypatch):
    _install_training_stubs(monkeypatch)
    model_dir = tmp_path / "base"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")
    output = tmp_path / "adapter"
    config = {
        "model_path": str(model_dir),
        "dataset_path": str(dataset),
        "output_dir": str(output),
        "text_field": "text",
        "max_steps": 4,
        "sequence_length": 64,
        "learning_rate": 0.0001,
        "lora_rank": 4,
        "checkpoint_interval": 2,
        "output_policy": "new",
    }
    validated = fine_tuning.validate_config(config)

    def stop_after_first_checkpoint(step, _loss):
        if step == 2:
            raise fine_tuning.FineTuningCancelledError("test interruption")

    with pytest.raises(fine_tuning.FineTuningCancelledError, match="test interruption"):
        fine_tuning._run_transformers(validated, stop_after_first_checkpoint)

    from axiom.core.fine_tuning_checkpoints import latest_checkpoint

    checkpoint, state = latest_checkpoint(output)
    assert checkpoint.name == "checkpoint-000002"
    assert state["completed_steps"] == 2

    resume_config = {**config, "output_policy": "resume"}
    resumed = fine_tuning.validate_config(resume_config)
    observed_steps = []
    result = fine_tuning._run_transformers(
        resumed,
        lambda step, _loss: observed_steps.append(step),
        begin_finalizing=lambda: None,
    )

    assert observed_steps == [3, 4]
    assert result["resumed_from_step"] == 2
    assert result["checkpoint"]["latest_step"] == 4
    assert (output / "adapter_config.json").is_file()
    assert (output / "adapter_model.safetensors").is_file()
    assert (output / "tokenizer.json").is_file()
