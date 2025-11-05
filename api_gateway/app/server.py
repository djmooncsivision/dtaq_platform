import os
import socket
from contextlib import closing

import uvicorn


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
MAX_PORT_OFFSET = 100


def find_available_port(start_port: int, max_offset: int = MAX_PORT_OFFSET) -> int:
    """Return the first available TCP port starting from `start_port`."""
    for port in range(start_port, start_port + max_offset + 1):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((DEFAULT_HOST, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_offset}")


def main() -> None:
    host = os.getenv("HOST", DEFAULT_HOST)
    start_port = int(os.getenv("PORT", DEFAULT_PORT))
    reload_enabled = os.getenv("RELOAD", "true").lower() in {"1", "true", "yes"}

    port = find_available_port(start_port)

    uvicorn.run(
        "api_gateway.app.main:app",
        host=host,
        port=port,
        reload=reload_enabled,
        log_level=os.getenv("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
