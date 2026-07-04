"""Command-line entrypoint for equiparity.

Thin dispatch layer: parses arguments, configures logging, and calls into workflows. It
carries no scientific logic (CODING_RULES.md Sections A and D).
"""

from __future__ import annotations

import argparse

from equiparity import __version__
from equiparity.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="equiparity", description=__doc__)
    parser.add_argument("--version", action="version", version=f"equiparity {__version__}")
    parser.add_argument(
        "--json-logs", action="store_true", help="Emit JSON logs (for final-result runs)."
    )
    parser.set_defaults(subcommands_pending=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(json_logs=args.json_logs)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
