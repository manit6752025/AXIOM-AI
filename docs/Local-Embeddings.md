# Local embeddings

AXIOM creates dense text embeddings locally from a trusted Hugging Face-style
checkpoint. It uses the Transformers encoder forward pass, attention-mask mean
pooling, and optional L2 normalization. It does not call hosted embedding APIs,
download model files, execute repository code, or persist input text.

## Install and prepare a model

Install the optional dependencies:

~~~bash
python -m pip install -e '.[embeddings]'
~~~

Place tokenizer/configuration files and model.safetensors (or a safetensors
shard index plus its shards) under an AXIOM trusted model root. Configure extra
directories with AXIOM_MODEL_ROOTS. Remote code, pickle weights, and network
downloads are disabled.

## Embed text

~~~bash
axiom engine embed --model-path ./models/my-encoder --text "AXIOM runs locally."
axiom engine embed --model-path ./models/my-encoder --text "First document" --text "Second document" --json
axiom engine embed --model-path ./models/my-encoder --input-file ./data/document.txt --device cpu --json
axiom engine embed-unload
~~~

The JSON response includes vectors and chunk offsets, never source text.
Character chunking uses bounded overlap to preserve context across boundaries;
tokenizer truncation is independently bounded by max_length.

## Bounds and privacy

A request accepts at most 32 inputs, 65,536 characters per input, 262,144
characters total, 128 chunks, 16 chunks per inference batch, and 512 tokens per
chunk. Output is capped at two million vector values. The API and CLI share the
same core validation.

The selected model must be under a trusted model root. Loading is
local_files_only, trust_remote_code=False, and use_safetensors=True. CPU, CUDA,
ROCm, Metal/MPS, and Intel XPU are supported when available. Input text is not
logged or retained; embeddings are returned to the caller only, not persisted
to a vector database. The Runtime page, CLI, and /api/engine/embeddings endpoint
use the same adapter. Unload clears AXIOM's cached model without changing files.

This is an encoder adapter, not yet a complete knowledge-base/vector-store
feature. Ingestion, persistence, hybrid retrieval, and reranking are separate
roadmap deliverables.
