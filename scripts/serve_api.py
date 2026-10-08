#!/usr/bin/env python3
"""Run the API using host and port from the shared config.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project_settings import configure_cli, setting


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    configure_cli(parser)
    parser.add_argument("--host", default=setting("api.host"))
    parser.add_argument("--port", type=int, default=setting("api.port"))
    args = parser.parse_args()
    import uvicorn
    from api.main import app

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
