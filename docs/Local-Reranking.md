# Local cross-encoder reranking

AXIOM reranks query/document pairs with a local Transformers sequence
classification checkpoint. The model scores each pair jointly, then AXIOM
sorts candidates by descending score. This differs from embedding retrieval:
a cross-encoder is slower but can compare query and candidate content together.

## Prepare and run

Install the optional local embeddings dependencies:

~~~bash
python -m pip install -e '.[embeddings]'
~~~

Use a compatible tokenizer and a sequence-classification model with
model.safetensors (or a safetensors shard index and shards) in a trusted AXIOM
model root. Remote model code, pickle weights, hosted inference, and downloads
are disabled.

~~~bash
axiom engine rerank --model-path ./models/my-reranker --query "local AI" --document "Candidate one" --document "Candidate two" --json
axiom engine rerank --model-path ./models/my-reranker --query "local AI" --documents-file ./data/candidates.json
axiom engine rerank-unload
~~~

The JSON-file option must contain an array of strings. The API equivalent is
POST /api/engine/rerank. The Runtime page includes the same local reranking
workflow and reports rank, input index, score, and character count without
displaying candidate content.

## Score contract and limits

Requests accept one nonblank query (up to 8,192 characters) and 1–64 nonblank
documents (up to 8,192 characters each), with a combined 262,144-character
limit. Batch size is 1–16 and token length is 32–512. Scores are finite and
sorted descending; equal scores retain their original input order.

One-logit regression heads return sigmoid(logit); two-logit classifiers return
softmax(logits)[1]. These are model ranking signals, not calibrated relevance
probabilities. AXIOM does not reinterpret labels or claim a universal score
meaning across different models. The response contains input indices and
scores, not query/document text. No source text, model files, or results are
persisted.
