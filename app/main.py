"""Entry point for the Alter-Ego Web Demo application.

Usage:
    uv run python -m app [--port 8000] [--host 127.0.0.1] [--open]
    uv run python app/main.py [--port 8000] [--host 127.0.0.1] [--open]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.server import run_server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Alter-Ego Daylight Keynote Web Demo.")
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "127.0.0.1"),
        help="Bind host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Bind port (default: 8000)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Automatically open the demo URL in the default browser",
    )
    args = parser.parse_args(argv)
    run_server(host=args.host, port=args.port, open_browser=args.open)


if __name__ == "__main__":
    main()
