from __future__ import annotations

from axiom.core import engine_fabric


def test_model_inventory_parses_ollama_and_openai_shapes():
    assert engine_fabric._model_ids("ollama", {"models": [{"name": "qwen2.5:0.5b"}, {"name": "../bad"}]}) == ["qwen2.5:0.5b"]
    assert engine_fabric._model_ids("llama-cpp", {"data": [{"id": "local-model"}, {"id": "bad model"}]}) == ["local-model"]


def test_dispatch_to_openai_compatible_backend_uses_fixed_chat_route(monkeypatch):
    observed = {}

    def fake_request(backend_id, path, payload=None, *, timeout):
        observed.update(backend_id=backend_id, path=path, payload=payload, timeout=timeout)
        return {
            "choices": [{"message": {"role": "assistant", "content": "Hello from llama.cpp"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 4},
        }

    monkeypatch.setattr(engine_fabric, "_request_json", fake_request)
    result = engine_fabric.run_engine_backend_chat(
        "llama-cpp", "local-model", [{"role": "user", "content": "Say hello."}],
        max_new_tokens=24, temperature=0.2, top_p=0.8, top_k=40,
        repetition_penalty=1.15, seed=123, stop_sequences=["<END>"],
    )
    assert observed["backend_id"] == "llama-cpp"
    assert observed["path"] == "/v1/chat/completions"
    assert observed["payload"]["messages"] == [{"role": "user", "content": "Say hello."}]
    assert observed["payload"]["max_tokens"] == 24
    assert observed["payload"]["top_p"] == 0.8
    assert observed["payload"]["top_k"] == 40
    assert observed["payload"]["repeat_penalty"] == 1.15
    assert observed["payload"]["seed"] == 123
    assert observed["payload"]["stop"] == ["<END>"]
    assert observed["timeout"] == engine_fabric._REQUEST_TIMEOUT
    assert result["response"] == "Hello from llama.cpp"
    assert result["eval_count"] == 4
    assert result["privacy"]["cloud_provider_used"] is False


def test_dispatch_to_ollama_uses_native_chat_options(monkeypatch):
    observed = {}

    def fake_request(backend_id, path, payload=None, *, timeout):
        observed.update(backend_id=backend_id, path=path, payload=payload)
        return {
            "message": {"role": "assistant", "content": "Local answer"},
            "done": True, "prompt_eval_count": 9, "eval_count": 6, "total_duration": 3_000_000,
        }

    monkeypatch.setattr(engine_fabric, "_request_json", fake_request)
    result = engine_fabric.run_engine_backend_chat(
        "ollama", "qwen2.5:0.5b", [{"role": "user", "content": "Answer locally."}],
        top_p=0.85, top_k=32, min_p=0.2, repetition_penalty=1.2, seed=7, stop_sequences=["DONE"],
    )
    assert observed["path"] == "/api/chat"
    assert observed["payload"]["options"] == {
        "num_predict": 256, "temperature": 0.7, "top_p": 0.85,
        "top_k": 32, "min_p": 0.2, "repeat_penalty": 1.2, "seed": 7, "stop": ["DONE"],
    }
    assert result["response"] == "Local answer"
    assert result["total_duration_ms"] == 3
    assert result["execution"] == "local_loopback"


def test_native_engine_dispatches_via_trusted_alias_and_never_external_http(monkeypatch):
    from axiom.core import llm_client

    captured = {}
    monkeypatch.setattr(
        engine_fabric, "_request_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("native dispatch must not use HTTP")),
    )

    def fake_chat(provider, model, messages, **kwargs):
        captured.update(provider=provider, model=model, messages=messages, **kwargs)
        return {
            "response": "native response", "prompt_tokens": 4, "completion_tokens": 3,
            "device": "cpu", "precision": "fp32", "generated_at": "2026-10-24T00:00:00+00:00",
        }

    monkeypatch.setattr(llm_client, "chat", fake_chat)
    result = engine_fabric.run_engine_backend_chat(
        "axiom-native", "local-model", [{"role": "user", "content": "Hi"}],
        max_new_tokens=32, top_p=0.8, top_k=12, min_p=0.24, repetition_penalty=1.1,
        seed=10, stop_sequences=["END"],
    )
    assert captured["provider"] == "axiom-local"
    assert captured["model"] == "local-model"
    assert captured["top_p"] == 0.8 and captured["top_k"] == 12
    assert captured["min_p"] == 0.24
    assert captured["repetition_penalty"] == 1.1 and captured["seed"] == 10
    assert captured["stop_sequences"] == ("END",)
    assert result["backend_id"] == "axiom-native"
    assert result["execution"] == "in_process_native"
    assert result["response"] == "native response"
    assert result["eval_count"] == 3
    assert result["privacy"]["cloud_provider_used"] is False


def test_native_backend_advertises_supported_model_aliases(monkeypatch):
    from axiom.core import engine, llm_client

    monkeypatch.setattr(engine, "runtime_status", lambda: {"ready": True, "setup_note": None})
    monkeypatch.setattr(llm_client, "_native_models", lambda: [
        {"name": "qwen-tiny", "path": "/trusted/qwen-tiny", "model_type": "qwen2"},
        {"name": "qwen-tiny", "path": "/trusted/qwen-tiny-copy", "model_type": "qwen2"},
        {"name": "llama-small", "path": "/trusted/llama-small", "model_type": "llama"},
    ])
    backend = engine_fabric._native_backend()
    assert backend["status"] == "ready"
    assert backend["supports_chat"] is True
    assert backend["models"] == ["llama-small"]
    assert backend["model_count"] == 1


def test_chat_validates_before_any_local_request(monkeypatch):
    monkeypatch.setattr(engine_fabric, "_request_json", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not dispatch")))
    invalid = [
        ("unknown", "model", [{"role": "user", "content": "Hi"}]),
        ("ollama", "http://remote.invalid", [{"role": "user", "content": "Hi"}]),
        ("ollama", "model", [{"role": "system", "content": "no user turn"}]),
    ]
    for backend, model, messages in invalid:
        try:
            engine_fabric.run_engine_backend_chat(backend, model, messages)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid route, model, or conversation must be rejected")


def test_benchmark_returns_measured_metadata_without_model_text(monkeypatch):
    calls = []

    def fake_chat(backend_id, model, messages, **kwargs):
        calls.append((backend_id, model, messages, kwargs))
        return {
            "response": "DO_NOT_RETURN_THIS_MODEL_TEXT",
            "eval_count": 8,
            "total_duration_ms": 12.0,
            "done": True,
        }

    monkeypatch.setattr(engine_fabric, "run_engine_backend_chat", fake_chat)
    result = engine_fabric.run_engine_backend_benchmark(
        "llama-cpp", "tiny-local", "A representative prompt.",
        warmup_runs=1, iterations=3, max_new_tokens=16,
    )
    assert len(calls) == 4
    assert result["configuration"]["measured_requests"] == 3
    assert result["measurements"]["completion_tokens_known"] == 3
    assert result["measurements"]["total_completion_tokens"] == 24
    assert result["measurements"]["latency_ms"]["median"] >= 0
    assert "DO_NOT_RETURN_THIS_MODEL_TEXT" not in str(result)
    assert "A representative prompt." not in str(result)
    assert result["privacy"]["responses_returned"] is False
    assert result["privacy"]["results_persisted"] is False


def test_benchmark_rejects_unbounded_iterations_before_dispatch(monkeypatch):
    monkeypatch.setattr(engine_fabric, "run_engine_backend_chat", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")))
    try:
        engine_fabric.run_engine_backend_benchmark(
            "llama-cpp", "tiny-local", "Prompt", warmup_runs=1, iterations=6,
        )
    except ValueError as exc:
        assert "between 1 and 5" in str(exc)
    else:
        raise AssertionError("benchmark must enforce iteration cap")


def test_auto_router_fails_over_only_between_ready_exact_model_matches(monkeypatch):
    report = {
        "profile": "balanced",
        "backends": [
            {"id": "axiom-native", "name": "AXIOM Native Engine", "status": "ready", "supports_chat": True, "models": []},
            {"id": "llama-cpp", "name": "llama.cpp server", "status": "ready", "supports_chat": True, "models": ["tiny-local"]},
            {"id": "ollama", "name": "Ollama", "status": "ready", "supports_chat": True, "models": ["tiny-local"]},
            {"id": "vllm", "name": "vLLM", "status": "ready", "supports_chat": True, "models": ["other-model"]},
        ],
    }
    monkeypatch.setattr(engine_fabric, "discover_engine_backends", lambda profile: report)
    observed = []

    def fake_chat(backend_id, model, messages, **kwargs):
        observed.append((backend_id, model, messages, kwargs))
        if backend_id == "llama-cpp":
            raise engine_fabric.EngineFabricError("simulated local timeout")
        return {
            "schema_version": 1, "backend_id": "ollama", "backend_name": "Ollama",
            "model": model, "response": "local fallback response", "done": True,
            "prompt_eval_count": 3, "eval_count": 5, "total_duration_ms": 18.0,
            "generated_at": "2026-10-24T00:00:00+00:00", "execution": "local_loopback",
            "privacy": {"prompt_returned": False, "response_persisted": False, "cloud_provider_used": False},
        }

    monkeypatch.setattr(engine_fabric, "run_engine_backend_chat", fake_chat)
    result = engine_fabric.run_engine_fabric_chat(
        "balanced", "tiny-local", [{"role": "user", "content": "Hello"}],
        max_new_tokens=64, temperature=0.2,
    )
    assert [item[0] for item in observed] == ["llama-cpp", "ollama"]
    assert all(item[1] == "tiny-local" for item in observed)
    assert observed[0][3]["request_timeout"] == 45.0
    assert result["response"] == "local fallback response"
    assert result["routing"]["fallback_used"] is True
    assert result["routing"]["max_attempts"] == 3
    assert [item["outcome"] for item in result["routing"]["attempts"]] == ["failed", "selected"]
    assert "do not prove identical checkpoint weights" in result["routing"]["scope_note"]


def test_auto_router_rejects_model_ids_not_advertised_by_any_ready_local_service(monkeypatch):
    monkeypatch.setattr(engine_fabric, "discover_engine_backends", lambda profile: {
        "profile": profile,
        "backends": [
            {"id": "ollama", "name": "Ollama", "status": "ready", "supports_chat": True, "models": ["qwen-local"]},
            {"id": "llama-cpp", "name": "llama.cpp", "status": "offline", "supports_chat": True, "models": ["not-ready"]},
        ],
    })
    monkeypatch.setattr(
        engine_fabric, "run_engine_backend_chat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not dispatch")),
    )
    try:
        engine_fabric.run_engine_fabric_chat(
            "balanced", "missing-model", [{"role": "user", "content": "Hello"}],
        )
    except engine_fabric.EngineFabricError as exc:
        assert "exact model ID" in str(exc)
    else:
        raise AssertionError("auto-routing must not guess a model alias")


def test_vllm_uses_its_repetition_penalty_extension_name(monkeypatch):
    observed = {}

    def fake_request(backend_id, path, payload=None, *, timeout):
        observed.update(backend_id=backend_id, path=path, payload=payload)
        return {
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }

    monkeypatch.setattr(engine_fabric, "_request_json", fake_request)
    engine_fabric.run_engine_backend_chat(
        "vllm", "local-model", [{"role": "user", "content": "Say ok"}],
        repetition_penalty=1.25, top_k=16,
    )
    assert observed["payload"]["repetition_penalty"] == 1.25
    assert observed["payload"]["top_k"] == 16


def test_sampling_validation_happens_before_any_backend_request(monkeypatch):
    monkeypatch.setattr(
        engine_fabric, "_request_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not dispatch")),
    )
    invalid_options = [
        {"top_p": 0}, {"top_p": float("nan")}, {"top_k": -1}, {"top_k": True},
        {"min_p": -0.01}, {"min_p": 1.01}, {"min_p": float("nan")}, {"min_p": True},
        {"repetition_penalty": 0}, {"seed": -1}, {"seed": True},
        {"stop_sequences": [""]}, {"stop_sequences": ["x" * 257]},
        {"stop_sequences": ["one"] * 17},
    ]
    for options in invalid_options:
        try:
            engine_fabric.run_engine_backend_chat(
                "ollama", "tiny-local", [{"role": "user", "content": "hello"}], **options,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid sampling settings to be rejected: {options!r}")


def test_min_p_auto_route_filters_compatibility_before_candidate_limit(monkeypatch):
    monkeypatch.setattr(engine_fabric, "discover_engine_backends", lambda profile: {
        "profile": profile,
        "backends": [
            {"id": "vllm", "name": "vLLM", "status": "ready", "supports_chat": True, "models": ["tiny"]},
            {"id": "llama-cpp", "name": "llama.cpp", "status": "ready", "supports_chat": True, "models": ["tiny"]},
            {"id": "lm-studio", "name": "LM Studio", "status": "ready", "supports_chat": True, "models": ["tiny"]},
            {"id": "ollama", "name": "Ollama", "status": "ready", "supports_chat": True, "models": ["tiny"]},
        ],
    })
    observed = []

    def fake_chat(backend_id, model, messages, **kwargs):
        observed.append((backend_id, kwargs))
        return {
            "schema_version": 1, "backend_id": backend_id, "backend_name": "Ollama",
            "model": model, "response": "filtered correctly", "done": True,
            "prompt_eval_count": 2, "eval_count": 3, "total_duration_ms": 10.0,
            "generated_at": "2026-10-24T00:00:00+00:00", "execution": "local_loopback",
            "privacy": {"prompt_returned": False, "response_persisted": False, "cloud_provider_used": False},
        }

    monkeypatch.setattr(engine_fabric, "run_engine_backend_chat", fake_chat)
    result = engine_fabric.run_engine_fabric_chat(
        "throughput", "tiny", [{"role": "user", "content": "hello"}], min_p=0.2,
    )
    assert [backend_id for backend_id, _kwargs in observed] == ["ollama"]
    assert observed[0][1]["min_p"] == 0.2
    assert result["backend_id"] == "ollama"


def test_min_p_is_rejected_for_backends_that_do_not_support_it(monkeypatch):
    monkeypatch.setattr(
        engine_fabric, "_request_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unsupported sampling must not dispatch")),
    )
    try:
        engine_fabric.run_engine_backend_chat(
            "llama-cpp", "tiny-local", [{"role": "user", "content": "hello"}], min_p=0.2,
        )
    except ValueError as exc:
        assert "AXIOM Native Engine and Ollama" in str(exc)
    else:
        raise AssertionError("nonzero min-p must not be silently dropped for external backends")


def test_benchmark_forwards_sampling_options_without_disclosing_stop_text(monkeypatch):
    calls = []

    def fake_chat(backend_id, model, messages, **kwargs):
        calls.append(kwargs)
        return {"response": "secret generated text", "eval_count": 2, "total_duration_ms": 1}

    monkeypatch.setattr(engine_fabric, "run_engine_backend_chat", fake_chat)
    result = engine_fabric.run_engine_backend_benchmark(
        "ollama", "tiny-local", "private representative prompt",
        warmup_runs=0, iterations=1, top_p=0.75, top_k=12, min_p=0.18,
        repetition_penalty=1.2, seed=99, stop_sequences=["SECRET_STOP"],
    )
    assert calls[0]["top_p"] == 0.75
    assert calls[0]["top_k"] == 12
    assert calls[0]["min_p"] == 0.18
    assert calls[0]["repetition_penalty"] == 1.2
    assert calls[0]["seed"] == 99
    assert calls[0]["stop_sequences"] == ("SECRET_STOP",)
    assert result["configuration"]["stop_sequence_count"] == 1
    assert result["configuration"]["min_p"] == 0.18
    assert "SECRET_STOP" not in str(result)
    assert "private representative prompt" not in str(result)


def test_auto_router_can_select_native_engine_before_external_services(monkeypatch):
    report = {
        "profile": "balanced",
        "backends": [
            {"id": "axiom-native", "name": "AXIOM Native Engine", "status": "ready", "supports_chat": True, "models": ["qwen-tiny"]},
            {"id": "llama-cpp", "name": "llama.cpp server", "status": "ready", "supports_chat": True, "models": ["qwen-tiny"]},
        ],
    }
    monkeypatch.setattr(engine_fabric, "discover_engine_backends", lambda profile: report)
    calls = []

    def fake_chat(backend_id, model, messages, **kwargs):
        calls.append((backend_id, kwargs))
        return {
            "schema_version": 1, "backend_id": backend_id,
            "backend_name": "AXIOM Native Engine" if backend_id == "axiom-native" else "llama.cpp server",
            "model": model, "response": "native wins by profile", "done": True,
            "prompt_eval_count": 2, "eval_count": 4, "total_duration_ms": 9.5,
            "generated_at": "2026-10-24T00:00:00+00:00",
            "execution": "in_process_native" if backend_id == "axiom-native" else "local_loopback",
            "privacy": {"prompt_returned": False, "response_persisted": False, "cloud_provider_used": False},
        }

    monkeypatch.setattr(engine_fabric, "run_engine_backend_chat", fake_chat)
    result = engine_fabric.run_engine_fabric_chat(
        "balanced", "qwen-tiny", [{"role": "user", "content": "Hello"}],
    )
    assert [item[0] for item in calls] == ["axiom-native"]
    assert "request_timeout" not in calls[0][1]
    assert result["backend_id"] == "axiom-native"
    assert result["routing"]["attempts"][0]["outcome"] == "selected"


def test_native_benchmark_uses_the_common_bounded_benchmark_contract(monkeypatch):
    calls = []

    def fake_chat(backend_id, model, messages, **kwargs):
        calls.append((backend_id, kwargs))
        return {"response": "do not return", "eval_count": 5, "total_duration_ms": 2}

    monkeypatch.setattr(engine_fabric, "run_engine_backend_chat", fake_chat)
    result = engine_fabric.run_engine_backend_benchmark(
        "axiom-native", "qwen-tiny", "representative prompt",
        warmup_runs=0, iterations=2, max_new_tokens=8,
    )
    assert len(calls) == 2 and all(item[0] == "axiom-native" for item in calls)
    assert result["backend_name"] == "AXIOM Native Engine"
    assert result["measurements"]["completion_tokens_known"] == 2
    assert "do not return" not in str(result)
    assert "representative prompt" not in str(result)


def test_generation_capabilities_describe_runtime_dialects_without_overclaiming():
    native = engine_fabric._generation_capabilities("axiom-native")
    assert native["transport"] == "in_process"
    assert native["extension_sampling_controls"] == []
    assert native["parameter_mapping"]["repetition_penalty"] == "repetition_penalty"
    assert native["parameter_mapping"]["min_p"] == "min_p"
    assert "min_p" in native["native_sampling_controls"]

    ollama = engine_fabric._generation_capabilities("ollama")
    assert ollama["transport"] == "ollama_options"
    assert ollama["parameter_mapping"]["max_new_tokens"] == "options.num_predict"
    assert ollama["parameter_mapping"]["repetition_penalty"] == "options.repeat_penalty"
    assert ollama["parameter_mapping"]["min_p"] == "options.min_p"
    assert "min_p" in ollama["native_sampling_controls"]
    assert ollama["extension_sampling_controls"] == []

    vllm = engine_fabric._generation_capabilities("vllm")
    llama_cpp = engine_fabric._generation_capabilities("llama-cpp")
    assert vllm["parameter_mapping"]["repetition_penalty"] == "repetition_penalty"
    assert llama_cpp["parameter_mapping"]["repetition_penalty"] == "repeat_penalty"
    assert "top_k" in vllm["extension_sampling_controls"]
    assert "API-version support" in vllm["note"]


def test_discovered_backends_publish_sampling_contract(monkeypatch):
    def fake_probe(definition):
        return {
            "id": definition["id"], "name": definition["name"], "kind": definition["kind"],
            "endpoint": definition["endpoint"], "supports_chat": True, "formats": definition["formats"],
            "strengths": definition["strengths"], "tradeoffs": definition["tradeoffs"], "best_for": definition["best_for"],
            "status": "ready", "installed": True, "reachable": True, "models": ["tiny"], "model_count": 1,
            "error": None, "scope": "fixed_loopback_only",
            "generation": engine_fabric._generation_capabilities(definition["id"]),
        }
    monkeypatch.setattr(engine_fabric, "_probe_backend", fake_probe)
    monkeypatch.setattr(engine_fabric, "_native_backend", lambda: {
        "id": "axiom-native", "name": "AXIOM Native Engine", "kind": "native", "endpoint": None,
        "supports_chat": True, "formats": ["safetensors"], "strengths": [], "tradeoffs": [],
        "best_for": ["balanced"], "status": "ready", "installed": True, "reachable": True,
        "models": ["tiny"], "model_count": 1, "error": None, "scope": "in_process_native_engine",
        "generation": engine_fabric._generation_capabilities("axiom-native"),
    })
    report = engine_fabric.discover_engine_backends()
    assert all("generation" in backend for backend in report["backends"])
    assert report["backends"][0]["generation"]["transport"] == "in_process"
