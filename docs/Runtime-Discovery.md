# Runtime connector discovery

The Runtime page includes a passive local connector inventory for Ollama, llama.cpp server, LM Studio CLI, vLLM CLI, llamafile, vLLM Python, llama-cpp-python, and MLX-LM.

Discovery only calls `shutil.which` for known executable names and Python's `importlib.util.find_spec` for known package names. It does not launch a process, import a candidate runtime package, open a local or remote socket, download a model, or inspect credentials.

Statuses mean:

- **Discovered:** a known executable is on `PATH` or a known Python package is discoverable in the current environment.
- **Not found:** the executable/package could not be discovered by the passive check.
- **Not connected:** discovery does not claim a server is running, an endpoint is compatible, a model is loaded, or inference is ready.

The field `potential_capabilities` is a catalogue hint, not negotiated capabilities. Actual runtime connection, readiness, latency, throughput, and compatibility checks remain separate roadmap work. No executable absolute paths or secrets are returned.
