"""CLI commands and interactive questionnaire for Model Studio."""
from __future__ import annotations
import json
import typer
from rich.console import Console
from rich.table import Table
from axiom.core import model_studio as studio

console = Console()
studio_app = typer.Typer(help="Search, download, convert, and quantize local AI models.", no_args_is_help=True)

def _fail(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code=1)

@studio_app.command("search")
def search(query: str = typer.Argument(..., help="Search public Hugging Face GGUF repositories."),
           limit: int = typer.Option(12, min=1, max=20), json_output: bool = typer.Option(False, "--json")) -> None:
    """Search public model catalogue metadata; no model is downloaded."""
    try: result = studio.search_gguf_catalog(query, limit)
    except (ValueError, studio.ModelStudioError) as exc: _fail(str(exc))
    if json_output:
        console.print_json(json.dumps(result, ensure_ascii=False)); return
    table = Table(title=f"GGUF catalogue · {query}")
    table.add_column("#", style="cyan"); table.add_column("Repository"); table.add_column("Downloads", justify="right"); table.add_column("Likes", justify="right"); table.add_column("Updated")
    for index, item in enumerate(result["results"], 1): table.add_row(str(index), item["id"], f'{item["downloads"]:,}', f'{item["likes"]:,}', item["last_modified"][:10])
    console.print(table)
    console.print(result["note"])

@studio_app.command("files")
def files(repo_id: str, json_output: bool = typer.Option(False, "--json")) -> None:
    """List downloadable single-file GGUF files in a public repository."""
    try: result = studio.list_gguf_files(repo_id)
    except (ValueError, studio.ModelStudioError) as exc: _fail(str(exc))
    if json_output:
        console.print_json(json.dumps(result, ensure_ascii=False)); return
    table = Table(title=f"GGUF files · {repo_id}")
    table.add_column("#", style="cyan"); table.add_column("File"); table.add_column("Quantization"); table.add_column("Bytes", justify="right")
    for index, item in enumerate(result["files"], 1): table.add_row(str(index), item["filename"], item["quantization"], f'{item["size_bytes"]:,}' if item["size_bytes"] is not None else "unknown")
    console.print(table)

@studio_app.command("status")
def status(json_output: bool = typer.Option(False, "--json")) -> None:
    """Show local model-tool availability and recent Studio jobs."""
    result = {"runtime": studio.model_studio_status(), "jobs": studio.list_jobs(10)}
    if json_output: console.print_json(json.dumps(result, ensure_ascii=False)); return
    console.print_json(json.dumps(result, ensure_ascii=False))

@studio_app.command("pull")
def pull(model_name: str = typer.Argument(..., help="Ollama library model tag, e.g. qwen3.5:0.8b")) -> None:
    """Download a model from the Ollama library to the local Ollama store."""
    try: result = studio.pull_ollama_model(model_name, on_log=lambda msg: console.print(msg))
    except (ValueError, studio.ModelStudioError) as exc: _fail(str(exc))
    console.print(f"[green]Installed locally:[/green] {result['model_name']}")

@studio_app.command("pull-native")
def pull_native(repo_id: str = typer.Argument(..., help="Public Hugging Face repository ID for AXIOM-native safetensors."), json_output: bool = typer.Option(False, "--json")) -> None:
    """Pull compatible Llama/Mistral/Qwen2 safetensors into AXIOM's native model store."""
    try:
        result = studio.pull_native_model(
            repo_id,
            on_log=(lambda message: console.print(message)) if not json_output else None,
            on_progress=(lambda current, total: console.print(
                "\rDownloaded " + f"{current:,}" + (f" / {total:,} bytes" if total else " bytes"),
                end="",
            )) if not json_output else None,
        )
    except (ValueError, studio.ModelStudioError) as exc:
        _fail(str(exc))
    if json_output:
        console.print()
        console.print_json(json.dumps(result, ensure_ascii=False))
        return
    console.print("\n[green]Installed for AXIOM Native Engine:[/green] " + result["path"])
    console.print("Run: axiom engine run --model-path " + repr(result["path"]) + ' --prompt "Hello."')

@studio_app.command("install-gguf")
def install_gguf(
    repo_id: str,
    filename: str,
    model_name: str = typer.Option(..., "--name", help="Local name used for storage or runtime registration."),
    no_ollama: bool = typer.Option(False, "--download-only", help="Download to Model Studio storage without registering with Ollama."),
    native: bool = typer.Option(False, "--native", help="Install directly into AXIOM Native Engine's trusted model root."),
    json_output: bool = typer.Option(False, "--json", help="Print the final result as machine-readable JSON."),
) -> None:
    """Download a public Hub GGUF to Model Studio, Ollama, or AXIOM Native Engine."""
    if native and no_ollama:
        _fail("Choose either --native or --download-only, not both.")
    register_ollama = not (no_ollama or native)
    try:
        result = studio.install_hf_gguf(
            repo_id,
            filename,
            model_name,
            register_ollama,
            on_log=(lambda msg: console.print(msg)) if not json_output else None,
            on_progress=(lambda current, total: console.print(
                f"\\rDownloaded {current:,}" + (f" / {total:,} bytes" if total else " bytes"),
                end="",
            )) if not json_output else None,
            install_native=native,
        )
    except (ValueError, studio.ModelStudioError) as exc:
        _fail(str(exc))
    if json_output:
        console.print_json(json.dumps(result, ensure_ascii=False))
        return
    if native:
        console.print("\\n[green]Installed for AXIOM Native Engine:[/green] " + result["path"])
        console.print("Run: axiom engine run --model-path " + repr(result["path"]) + ' --prompt "Hello."')
    else:
        console.print("\\n[green]GGUF saved:[/green] " + result["path"])
        if result.get("registered_with_ollama"):
            console.print(f"[green]Ready for local inference:[/green] {result['model_name']}")

@studio_app.command("convert-gguf")
def convert_gguf(source_dir: str = typer.Argument(..., help="Existing local Transformers model directory."),
                 llama_cpp_dir: str = typer.Option(..., "--llama-cpp-dir", help="Local llama.cpp source/build directory."),
                 output_path: str = typer.Option(..., "--output", help="New .gguf file in an existing directory."),
                 outtype: str = typer.Option("f16", help="auto, f16, f32, or bf16."),
                 quantization: str = typer.Option("Q4_K_M", help="none, Q2_K, Q3_K_M, Q4_0, Q4_K_M, Q5_K_M, Q6_K, or Q8_0.")) -> None:
    """Convert local Transformers weights to GGUF, optionally quantizing them."""
    try: result = studio.convert_local_model_to_gguf(source_dir, llama_cpp_dir, output_path, outtype, quantization, on_log=lambda msg: console.print(msg))
    except (ValueError, studio.ModelStudioError) as exc: _fail(str(exc))
    console.print_json(json.dumps(result, ensure_ascii=False))

@studio_app.command("wizard")
def wizard() -> None:
    """Interactive model install questionnaire: task, source, catalogue, and local setup."""
    console.print("[bold]AXIOM Model Studio · install questionnaire[/bold]")
    task = typer.prompt("What do you need? (general chat / coding / reasoning / multilingual / low memory)", default="general chat")
    size = typer.prompt("Approximate RAM available (e.g. 8 GB, 16 GB, 32 GB)", default="16 GB")
    source = typer.prompt("Source: ollama or huggingface GGUF", default="huggingface", show_default=True).strip().lower()
    if source in {"ollama", "o"}:
        typer.echo("Browse model tags at https://ollama.com/search, then install a selected local tag.")
        model = typer.prompt("Ollama model tag (for example qwen3.5:0.8b)")
        if not typer.confirm(f"Pull {model} into your local Ollama store?", default=False):
            raise typer.Exit(code=0)
        try: result = studio.pull_ollama_model(model, on_log=lambda message: console.print(message))
        except (ValueError, studio.ModelStudioError) as exc: _fail(str(exc))
        console.print(f"[green]Installed:[/green] {result['model_name']}")
        return
    if source not in {"huggingface", "hf", "gguf", "huggingface gguf"}:
        _fail("Choose 'ollama' or 'huggingface'.")
    size_lower = size.strip().lower()
    low_ram = size_lower.startswith(("4", "8")) or "low" in size_lower or "small" in size_lower
    suffix = "GGUF Q4_K_M" if low_ram else "GGUF"
    query_default = {"coding": "coding", "reasoning": "reasoning", "multilingual": "multilingual", "low memory": "1B"}.get(task.lower().replace("-", " "), "instruct")
    query = typer.prompt("Search Hugging Face GGUF models", default=f"{query_default} {suffix}")
    try: catalog = studio.search_gguf_catalog(query, 12)
    except (ValueError, studio.ModelStudioError) as exc: _fail(str(exc))
    if not catalog["results"]: _fail("No GGUF repositories found. Try a broader query, such as 'Qwen GGUF' or 'Llama GGUF'.")
    for index, item in enumerate(catalog["results"], 1):
        console.print(f"{index:>2}. {item['id']} · {item['downloads']:,} downloads · {item['likes']:,} likes")
    selected = typer.prompt("Select repository number", type=int)
    if selected < 1 or selected > len(catalog["results"]): _fail("Selection is outside the result list.")
    repo = catalog["results"][selected - 1]["id"]
    try: inventory = studio.list_gguf_files(repo)
    except (ValueError, studio.ModelStudioError) as exc: _fail(str(exc))
    if not inventory["files"]: _fail("No single-file GGUF files were found in the selected repository.")
    for index, item in enumerate(inventory["files"][:80], 1):
        size_value = f"{item['size_bytes'] / (1024**3):.2f} GiB" if item["size_bytes"] else "size unknown"
        console.print(f"{index:>2}. {item['filename']} · {item['quantization']} · {size_value}")
    file_index = typer.prompt("Select GGUF file number", type=int)
    if file_index < 1 or file_index > min(len(inventory["files"]), 80): _fail("Selection is outside the displayed file list.")
    selected_file = inventory["files"][file_index - 1]["filename"]
    default_name = "axiom-" + repo.split("/")[-1].lower().replace("_", "-")[:48]
    model_name = typer.prompt("Local model name", default=default_name)
    register = typer.confirm("Register this GGUF with local Ollama after downloading?", default=True)
    console.print(f"Task: {task} · available RAM: {size} · repository: {repo} · file: {selected_file}")
    if not typer.confirm("Start download and installation?", default=False): raise typer.Exit(code=0)
    try:
        result = studio.install_hf_gguf(repo, selected_file, model_name, register,
            on_log=lambda message: console.print(message),
            on_progress=lambda current, total: console.print(f"\rDownloaded {current:,}" + (f" / {total:,} bytes" if total else " bytes"), end=""))
    except (ValueError, studio.ModelStudioError) as exc: _fail(str(exc))
    console.print("\n[green]Saved:[/green] " + result["path"])
    if result["registered_with_ollama"]: console.print(f"[green]Ready for inference:[/green] axiom infer run {result['model_name']} --prompt \"Hello\"")

@studio_app.command("remote-models")
def remote_models(base_url: str = typer.Argument(..., help="Public OpenAI-compatible HTTPS base URL ending in /v1")) -> None:
    """Test a public HTTPS endpoint and list its models without saving the API key."""
    api_key = typer.prompt("Provider API key (not saved by AXIOM)", hide_input=True)
    try: result = studio.list_remote_models(base_url, api_key)
    except (ValueError, studio.ModelStudioError) as exc: _fail(str(exc))
    console.print_json(json.dumps(result, ensure_ascii=False))

@studio_app.command("remote-run")
def remote_run(base_url: str = typer.Argument(..., help="Public OpenAI-compatible HTTPS base URL ending in /v1"),
               model: str = typer.Argument(..., help="Model ID from remote-models"),
               prompt: str = typer.Argument(..., help="Prompt; the provider will receive this text"),
               max_tokens: int = typer.Option(256, min=1, max=2048)) -> None:
    """Send one prompt to an explicitly configured online model provider."""
    if not typer.confirm("This sends the prompt to the remote provider. Continue?", default=False):
        raise typer.Exit(code=0)
    api_key = typer.prompt("Provider API key (not saved by AXIOM)", hide_input=True)
    try: result = studio.run_remote_chat(base_url, api_key, model, prompt, max_tokens=max_tokens)
    except (ValueError, studio.ModelStudioError) as exc: _fail(str(exc))
    console.print(result["response"])
    console.print(f"\n[dim]Provider: {result['base_url']} · model: {result['model']} · tokens: {result['completion_tokens']}[/dim]")

def register_model_studio(model_app: typer.Typer) -> None:
    """Attach `axiom model studio ...` commands to the existing model family."""
    model_app.add_typer(studio_app, name="studio")
