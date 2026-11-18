"""Start a narrowly scoped, token-protected OpenAI-compatible bridge for mobile agents."""
from __future__ import annotations

import secrets
import typer

from axiom.cli.errors import emit_error
from rich.console import Console

console = Console()


def register_mobile_bridge(app: typer.Typer) -> None:
    @app.command("mobile-bridge")
    def mobile_bridge(
        port: int = typer.Option(40509, "--port", min=1024, max=65535, help="TCP port for the OpenAI-compatible mobile endpoint."),
        listen_lan: bool = typer.Option(False, "--listen-lan", help="Allow devices on your LAN to connect. Requires the generated bearer token."),
    ) -> None:
        """Serve only /v1/models and /v1/chat/completions for a mobile AI client."""
        import os

        try:
            import uvicorn
            from fastapi import FastAPI
        except ImportError:
            emit_error("Install the API extra with: python -m pip install 'axiom-all'.", code="api_extra_missing")

        from axiom.api.openai_compat import router as openai_compat_router

        token = secrets.token_urlsafe(32)
        os.environ["AXIOM_OPENAI_API_TOKEN"] = token
        host = "0.0.0.0" if listen_lan else "127.0.0.1"

        bridge = FastAPI(
            title="AXIOM Mobile Agent Bridge",
            version="1",
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
        )
        # Do not mount the full AXIOM workspace API on a LAN-facing listener.
        bridge.include_router(openai_compat_router)

        console.print(f"[bold]AXIOM Mobile Agent Bridge[/bold] · {host}:{port}")
        console.print(f"Base URL: http://<AXIOM-computer-LAN-IP>:{port}/v1" if listen_lan else f"Base URL: http://127.0.0.1:{port}/v1")
        console.print(f"Model inventory: run 'axiom engine discover --json' and use a compatible model ID.")
        console.print(f"API key (generated for this process; copy into the mobile client): [bold]{token}[/bold]")
        console.print("Only OpenAI-compatible model listing and chat completion routes are exposed.")
        if listen_lan:
            console.print("[yellow]LAN safety:[/yellow] Keep this on a trusted private network, do not port-forward it, and stop it when not in use. HTTP does not encrypt traffic on the LAN; use a VPN or encrypted tunnel on untrusted networks.")
        console.print("Press Ctrl-C to stop the bridge.")
        try:
            uvicorn.run(bridge, host=host, port=port, access_log=False, log_level="warning")
        except KeyboardInterrupt:
            console.print("Mobile bridge stopped.")
        finally:
            if os.environ.get("AXIOM_OPENAI_API_TOKEN") == token:
                os.environ.pop("AXIOM_OPENAI_API_TOKEN", None)
