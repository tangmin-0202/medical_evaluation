from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from medical_evaluation.app import create_app
from medical_evaluation.settings import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the medical evaluation web demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--config", type=Path, help="Path to the model configuration YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = {"model_config_path": args.config} if args.config else {}
    app = create_app(Settings(**options))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
