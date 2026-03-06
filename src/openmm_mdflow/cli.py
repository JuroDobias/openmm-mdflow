from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_and_validate
from .workflow import run_workflow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openmm-mdflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate workflow YAML")
    validate_parser.add_argument("--config", required=True, help="Path to workflow YAML file")

    run_parser = subparsers.add_parser("run", help="Run workflow YAML")
    run_parser.add_argument("--config", required=True, help="Path to workflow YAML file")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_and_validate(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.command == "validate":
        print("Configuration is valid.")
        return 0

    if args.command == "run":
        run_workflow(config)
        print("Workflow completed.")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
