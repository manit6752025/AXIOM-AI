# Model Studio: model search, local installs, and GGUF

Model Studio provides explicit access to the public Hugging Face model API for **searching GGUF repositories** and downloading a selected single-file GGUF. It can optionally register that file as a local Ollama model. It can also call `ollama pull` for an Ollama library tag, or convert a local Transformers model directory to GGUF using a local llama.cpp checkout.

## Interactive questionnaire

```bash
axiom --no-banner model studio wizard
```

The wizard asks what the model is for, asks for an approximate RAM budget, selects a source (Ollama library or Hugging Face GGUF), presents search results/files, and confirms before starting a download. The RAM answer is only a hint, not a fit guarantee.

## Search and inspect

```bash
axiom --no-banner model studio search "Qwen instruct GGUF"
axiom --no-banner model studio files author/repository-GGUF
axiom --no-banner model studio status
```

The search reads public catalogue metadata when explicitly requested. Before downloading, inspect the model card and license. Downloads are limited to 20 GiB per file and require enough available disk space. Multi-file shards are not automatically assembled by this initial feature.

## Install from Ollama or Hugging Face

```bash
axiom --no-banner model studio pull qwen3.5:0.8b
axiom --no-banner model studio install-gguf author/repository-GGUF model-Q4_K_M.gguf --name axiom-local-model
axiom --no-banner model studio install-gguf author/repository-GGUF model-Q4_K_M.gguf --name axiom-local-model --download-only
```

The first command uses the locally installed Ollama CLI. The second downloads a public GGUF, writes a Modelfile, and registers the model with the local Ollama service. The third saves the GGUF without registering it. GGUFs are stored under `~/.local/share/axiom/model-studio/gguf/` by default (or `AXIOM_MODEL_STUDIO_HOME` if set). It refuses to overwrite an existing file.

## Create/quantize a GGUF from local Transformers weights

This operation **converts an existing model; it does not train a new base model**. The source must be a compatible local Transformers directory containing `config.json`. Set `--llama-cpp-dir` to a local llama.cpp checkout containing `convert_hf_to_gguf.py`; quantization also needs a built `llama-quantize` executable under that checkout.

```bash
axiom --no-banner model studio convert-gguf "$HOME/models/my-transformers-model" \
  --llama-cpp-dir "$HOME/src/llama.cpp" \
  --output "$HOME/models/my-model-Q4_K_M.gguf" \
  --quantization Q4_K_M
```

Conversion can consume substantial RAM and disk space. The output path must be new and its parent directory must already exist. GGUF compatibility depends on the source architecture and llama.cpp support. Review conversion output and run a real inference smoke test before treating a model as usable.

## Online provider connections

Model Studio can optionally test an OpenAI-compatible public HTTPS endpoint and send a prompt through its `/chat/completions` route. The UI asks you to consent before sending a prompt. The API key is held only in the current page/request and is not saved by Model Studio. AXIOM screens out non-HTTPS/private/loopback targets and blocks redirects to avoid forwarding an API key to a different host. The remote provider will receive your prompts and may process them under its own terms; do not send secrets/private data. Provider compatibility varies.

```bash
axiom --no-banner model studio remote-models https://provider.example/api/v1
axiom --no-banner model studio remote-run https://provider.example/api/v1 MODEL_ID "Explain GGUF in one sentence"
```

The CLI prompts for the API key with hidden input and `remote-run` asks for confirmation before sending the prompt. This differs from local Ollama inference, which remains loopback-only.

## Privacy and networking

Search/download requests go to Hugging Face only after an explicit user action. Ollama pulls contact the configured Ollama registry through the Ollama tool. Remote model requests send prompts to the provider only after explicit action and consent; credentials and remote responses are not written to Model Studio job history.
