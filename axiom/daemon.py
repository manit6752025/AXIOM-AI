"""Small, local-only HTTP daemon for the AXIOM CLI.

The daemon is intentionally dependency-free. It gives local scripts and future
integrations a stable health, local-context, and tool-discovery boundary without
turning AXIOM into a hosted service.
"""

from __future__ import annotations

import ipaddress
import json
import os
import shutil
import sys
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, TextIO
from urllib.parse import parse_qs, urlsplit

from axiom import daemon_api
from axiom.core.hardware import detect_hardware
from axiom.path_safety import (
    SafePathError,
    find_existing_relative_path,
    open_regular_file,
)
from axiom.tools.catalog import TOOL_CATALOG
from axiom.version import __version__

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
WEBUI_DIR = Path(__file__).resolve().parent / "webui"
PROJECT_PATHS: tuple[str, ...] = (
    "axiom.yaml",
    "README.md",
    "data",
    "models",
    "experiments",
    "evaluations",
    "outputs",
)

LOCAL_ACTIONS: tuple[dict[str, str], ...] = (
    {
        "command": "axiom guide",
        "description": "Explain the current project and choose a useful next command.",
    },
    {
        "command": "axiom summary",
        "description": "Print the current project, machine, tools, and recommendation.",
    },
    {
        "command": "axiom model inspect ./models/my-model",
        "description": "Inspect a local model directory before choosing a runtime.",
    },
    {
        "command": "axiom dataset validate ./data/train.jsonl",
        "description": "Find malformed records and duplicates in JSONL training data.",
    },
    {
        "command": "axiom system info",
        "description": "Read the local operating system, CPU, memory, GPU, and VRAM.",
    },
    {
        "command": "axiom train plan 7 --method qlora",
        "description": "Create a conservative hardware-aware starting plan; no training starts.",
    },
    {
        "command": "axiom tools doctor",
        "description": "Check which common developer tools are available on PATH.",
    },
    {
        "command": "axiom headroom",
        "description": "Check the optional Headroom local proxy without installing or sending data.",
    },
)

DAEMON_ENDPOINT_PATHS: tuple[tuple[str, str], ...] = (
    ("root", "/"),
    ("health", "/health"),
    ("info", "/api/info"),
    ("summary", "/api/summary"),
    ("actions", "/api/actions"),
    ("hardware", "/api/hardware"),
    ("tools", "/api/tools"),
    ("ui", "/ui/"),
)


def validate_bind_host(host: str, *, allow_network: bool = False) -> str:
    """Validate a daemon bind address before opening a listening socket."""

    normalized = host.strip()
    if not normalized:
        raise ValueError("host cannot be empty")

    is_loopback = normalized.lower() in LOOPBACK_HOSTS
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(normalized).is_loopback
        except ValueError:
            is_loopback = False

    if not is_loopback and not allow_network:
        raise ValueError(
            "AXIOM daemon is local-only by default; use a loopback host "
            "or pass --allow-network explicitly"
        )

    return normalized


def _display_url(host: str, port: int) -> str:
    display_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{display_host}:{port}"


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")


def _tool_snapshot() -> list[dict[str, Any]]:
    """Return executable presence without reading credentials or running tools."""

    snapshot: list[dict[str, Any]] = []
    for spec in TOOL_CATALOG:
        executable = None
        for name in spec.executable_names:
            executable = shutil.which(name)
            if executable:
                break
        snapshot.append(
            {
                "id": spec.id,
                "name": spec.name,
                "kind": spec.kind,
                "available": executable is not None,
                "executable": executable,
                "supported_platforms": list(spec.supported_platforms),
                "api_key_env_vars": list(spec.api_key_env_vars),
            }
        )
    return snapshot


def local_summary(cwd: Path | None = None) -> dict[str, Any]:
    """Build a local, read-only snapshot for the CLI and daemon."""

    root = Path.cwd() if cwd is None else cwd
    config_path = root / "axiom.yaml"
    missing = [path for path in PROJECT_PATHS if not (root / path).exists()]
    tools = _tool_snapshot()
    available_tools = [tool["name"] for tool in tools if tool["available"]]
    missing_tools = [tool["name"] for tool in tools if not tool["available"]]

    if not config_path.is_file():
        recommended = {
            "command": "axiom init my-ai",
            "reason": "No axiom.yaml was found in the current directory.",
        }
    elif missing:
        recommended = {
            "command": "axiom config validate",
            "reason": f"The project is missing: {', '.join(missing)}.",
        }
    else:
        recommended = {
            "command": "axiom model list",
            "reason": "The standard project paths are present.",
        }

    return {
        "name": "AXIOM",
        "version": __version__,
        "working_directory": str(root),
        "project": {
            "detected": config_path.is_file(),
            "config": str(config_path),
            "missing_paths": missing,
        },
        "recommended": recommended,
        "hardware": asdict(detect_hardware()),
        "tools": {
            "available": len(available_tools),
            "total": len(tools),
            "names": available_tools,
            "missing": missing_tools,
        },
    }


class _DaemonServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler: type[BaseHTTPRequestHandler],
    ):
        super().__init__(server_address, request_handler)
        self.axion_host = server_address[0]
        self.allow_network = False


class _RequestHandler(BaseHTTPRequestHandler):
    server: _DaemonServer

    protocol_version = "HTTP/1.1"

    def send_response(self, code: int, message: str | None = None) -> None:
        """Reject CR/LF in a custom status phrase before it reaches the wire."""
        if message is not None and any(char in str(message) for char in ("\r", "\n", "\x00")):
            raise ValueError("HTTP response status messages must not contain CR/LF or NUL.")
        super().send_response(code, message)

    def send_header(self, keyword: str, value: str) -> None:
        """Reject response-header injection before delegating to http.server."""
        header_name = str(keyword)
        header_value = str(value)
        if any(char in header_name for char in ("\r", "\n", ":")):
            raise ValueError("Invalid HTTP response header name.")
        if any(char in header_value for char in ("\r", "\n", "\x00")):
            raise ValueError("HTTP response header values must not contain CR/LF or NUL.")
        super().send_header(header_name, header_value)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _host_allowed(self) -> bool:
        if self.server.allow_network:
            return True
        header = self.headers.get("Host", "")
        host = header.rsplit(":", 1)[0] if not header.startswith("[") else header.split("]")[0] + "]"
        host = host.strip("[]").lower()
        if host in LOOPBACK_HOSTS:
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def _send_static(self, path: str, *, head_only: bool = False) -> bool:
        relative = path[len("/ui") :].lstrip("/") or "index.html"
        try:
            target = find_existing_relative_path(
                WEBUI_DIR, relative, allow_directories=False, skip_directories=()
            )
        except SafePathError:
            return False
        except (FileNotFoundError, OSError, RuntimeError):
            leaf = relative.rpartition("/")[2]
            if "." in leaf:
                return False
            try:
                target = find_existing_relative_path(
                    WEBUI_DIR, "index.html",
                    allow_directories=False,
                    skip_directories=(),
                )
            except (SafePathError, FileNotFoundError, OSError, RuntimeError):
                return False

        try:
            with open_regular_file(target) as stream:
                body = stream.read()
        except (OSError, SafePathError):
            return False

        content_type = {
            ".html": "text/html; charset=utf-8",
            ".htm": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".mjs": "text/javascript; charset=utf-8",
            ".json": "application/json",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".ico": "image/x-icon",
            ".woff": "font/woff",
            ".woff2": "font/woff2",
        }.get(target.suffix.lower(), "application/octet-stream")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)
        return True

    def do_GET(self) -> None:
        parts = urlsplit(self.path)
        path = parts.path.rstrip("/") or "/"

        if not self._host_allowed():
            self._send_json(
                HTTPStatus.FORBIDDEN,
                {"error": "forbidden_host", "message": "Use a loopback address."},
            )
            return

        if path == "/ui" or path.startswith("/ui/"):
            if parts.path == "/ui":
                self.send_response(HTTPStatus.MOVED_PERMANENTLY)
                self.send_header("Location", "/ui/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self._send_static(path):
                return
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {
                    "error": "web_ui_missing",
                    "hint": "Run scripts/build_webui.sh to bundle the web UI.",
                },
            )
            return

        routed = daemon_api.route(path, parse_qs(parts.query))
        if routed is not None:
            self._send_json(*routed)
            return

        if path == "/":
            self._send_json(
                HTTPStatus.OK,
                {
                    "name": "AXIOM",
                    "service": "local-daemon",
                    "message": "AXIOM is a local boundary for model, dataset, hardware, and tool checks.",
                    "health": "/health",
                    "info": "/api/info",
                    "summary": "/api/summary",
                    "actions": "/api/actions",
                    "hardware": "/api/hardware",
                    "tools": "/api/tools",
                },
            )
            return

        if path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {"name": "AXIOM", "status": "ok", "version": __version__},
            )
            return

        if path == "/api/info":
            cwd = Path.cwd()
            self._send_json(
                HTTPStatus.OK,
                {
                    "name": "AXIOM",
                    "version": __version__,
                    "service": "local-daemon",
                    "host": self.server.axion_host,
                    "port": self.server.server_address[1],
                    "working_directory": str(cwd),
                    "project_config": (cwd / "axiom.yaml").is_file(),
                    "capabilities": [
                        "project-scaffolding",
                        "model-inspection",
                        "dataset-validation",
                        "hardware-detection",
                        "training-plans",
                        "action-suggestions",
                        "tool-discovery",
                        "web-ui",
                        "model-registry",
                        "optimization-plans",
                        "headroom-status",
                    ],
                    "endpoints": {
                        "health": "/health",
                        "summary": "/api/summary",
                        "actions": "/api/actions",
                        "hardware": "/api/hardware",
                        "tools": "/api/tools",
                    },
                },
            )
            return

        if path == "/api/summary":
            self._send_json(HTTPStatus.OK, local_summary())
            return

        if path == "/api/actions":
            self._send_json(
                HTTPStatus.OK,
                {"name": "AXIOM", "actions": list(LOCAL_ACTIONS)},
            )
            return

        if path == "/api/hardware":
            self._send_json(
                HTTPStatus.OK,
                {"name": "AXIOM", "hardware": asdict(detect_hardware())},
            )
            return

        if path == "/api/tools":
            self._send_json(
                HTTPStatus.OK,
                {"name": "AXIOM", "tools": _tool_snapshot()},
            )
            return

        self._send_json(
            HTTPStatus.NOT_FOUND,
            {
                "error": "not_found",
                "path": path,
                "hint": "Try /ui/, /health, /api/info, /api/summary, /api/actions, /api/hardware, or /api/tools.",
            },
        )

    def do_HEAD(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if not self._host_allowed():
            self.send_response(HTTPStatus.FORBIDDEN)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path in {
            *daemon_api.API_PATHS,
            "/ui",
            "/",
            "/health",
            "/api/info",
            "/api/summary",
            "/api/actions",
            "/api/hardware",
            "/api/tools",
        }:
            self.send_response(HTTPStatus.OK)
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        # The CLI owns the one useful startup line; request logs would make the
        # daemon noisy when it is used from a terminal or a supervisor.
        return


def start_daemon(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    allow_network: bool = False,
) -> ThreadingHTTPServer:
    """Create a daemon server without starting its serving thread.

    ``port=0`` asks the operating system for a currently available port. The
    returned server is useful to callers that want to manage its lifecycle or
    run it in a background thread.
    """

    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")

    normalized_host = validate_bind_host(host, allow_network=allow_network)
    os.environ.setdefault("AXIOM_MODEL_ROOT", str(Path.cwd().resolve() / "models"))
    server = _DaemonServer((normalized_host, port), _RequestHandler)
    server.allow_network = allow_network
    return server


def daemon_url(server: ThreadingHTTPServer) -> str:
    """Return the URL clients should use for a started server."""

    host = getattr(server, "axion_host", server.server_address[0])
    return _display_url(host, server.server_address[1])


def daemon_ready_payload(server: ThreadingHTTPServer) -> dict[str, Any]:
    """Return stable connection details for a ready local daemon.

    The payload is intentionally useful to both humans and shell tooling.  It
    contains absolute endpoint URLs so callers never have to reconstruct the
    address selected by the operating system when ``port=0`` is used.
    """

    base_url = daemon_url(server)
    endpoints = {
        name: f"{base_url}{path}" for name, path in DAEMON_ENDPOINT_PATHS
    }

    return {
        "name": "AXIOM",
        "service": "local-daemon",
        "status": "ready",
        "url": base_url,
        "host": getattr(server, "axion_host", server.server_address[0]),
        "port": server.server_address[1],
        "endpoints": endpoints,
        "next_commands": [
            "axiom summary --json",
            f"curl -sS {endpoints['health']}",
            f"curl -sS {endpoints['summary']}",
        ],
    }


def _announce_daemon(
    server: ThreadingHTTPServer,
    stream: TextIO,
    *,
    machine_readable: bool,
) -> None:
    payload = daemon_ready_payload(server)

    if machine_readable:
        print(json.dumps(payload, sort_keys=True), file=stream, flush=True)
        return

    endpoints = payload["endpoints"]
    print(f"AXIOM daemon listening at {payload['url']}", file=stream, flush=True)
    print(f"Health:  {endpoints['health']}", file=stream, flush=True)
    print(f"Summary: {endpoints['summary']}", file=stream, flush=True)
    print("Try next:", file=stream, flush=True)
    for command in payload["next_commands"]:
        print(f"  {command}", file=stream, flush=True)
    print("Ctrl-C stops the daemon.", file=stream, flush=True)


def run_daemon(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    allow_network: bool = False,
    output: TextIO | None = None,
    machine_readable: bool = False,
) -> None:
    """Run the local daemon until Ctrl-C, then close its socket cleanly.

    ``machine_readable`` emits one JSON readiness record and one JSON stopped
    record, making ``axiom daemon --json`` safe to supervise without parsing
    human-oriented terminal output.
    """

    server = start_daemon(host=host, port=port, allow_network=allow_network)
    stream = sys.stdout if output is None else output
    _announce_daemon(server, stream, machine_readable=machine_readable)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if machine_readable:
            print(
                json.dumps(
                    {"name": "AXIOM", "service": "local-daemon", "status": "stopped"},
                    sort_keys=True,
                ),
                file=stream,
                flush=True,
            )
        else:
            print("\nAXIOM daemon stopped.", file=stream)
    finally:
        server.server_close()


def serve_in_thread(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    allow_network: bool = False,
) -> tuple[ThreadingHTTPServer, Thread]:
    """Start a managed daemon thread for embedding and integration tests."""

    server = start_daemon(host=host, port=port, allow_network=allow_network)
    thread = Thread(target=server.serve_forever, name="axiom-daemon", daemon=True)
    thread.start()
    return server, thread


__all__ = [
    "LOCAL_ACTIONS",
    "LOOPBACK_HOSTS",
    "DAEMON_ENDPOINT_PATHS",
    "daemon_ready_payload",
    "daemon_url",
    "local_summary",
    "run_daemon",
    "serve_in_thread",
    "start_daemon",
    "validate_bind_host",
]
