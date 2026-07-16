"""Command-line entrypoint for equiparity.

Thin dispatch layer: parses arguments, configures logging, and calls into workflows. It
carries no scientific logic (CODING_RULES.md Sections A and D).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from equiparity import __version__
from equiparity.io.config import load_experiment_config
from equiparity.logging_config import configure_logging
from equiparity.workflows.run_experiment import run_experiment

_log = logging.getLogger("equiparity.cli")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="equiparity", description=__doc__)
    parser.add_argument("--version", action="version", version=f"equiparity {__version__}")
    parser.add_argument(
        "--json-logs", action="store_true", help="Emit JSON logs (for final-result runs)."
    )
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="Run one experiment from a config YAML.")
    run.add_argument("config", type=Path, help="Path to an experiment config YAML.")
    run.add_argument(
        "--allow-dirty", action="store_true", help="Permit a dirty git tree (debug/smoke runs)."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(json_logs=args.json_logs)
    if args.command == "run":
        config = load_experiment_config(args.config)
        run_dir = run_experiment(config, allow_dirty=args.allow_dirty)
        _log.info("run complete: %s", run_dir)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
