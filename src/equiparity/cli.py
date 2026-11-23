"""Command-line interface for equiparity.

Thin dispatch layer: parses arguments, configures logging, and calls into workflows. It carries no
scientific logic. Commands that print data accept ``--json`` for machine-readable output, and every
command exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from equiparity import __version__
from equiparity.logging_config import configure_logging

_log = logging.getLogger("equiparity.cli")

_EPILOG = """\
examples:
  equiparity verify                          check theory and claims against results/
  equiparity records list --json             catalogue of released records
  equiparity records show crystal_regression              one record's metadata and path
  equiparity grid generate main              write the 42 crystal regression configs
  equiparity run configs/crystal/tensor_huber_nequip_piezoelectric_o3_seed0.yaml
  equiparity data prepare mp --dataset ood   fetch the centrosymmetric population
  equiparity manifest                        rebuild results/run_manifest.json
"""


def _emit(payload: object, *, as_json: bool, text: str) -> None:
    sys.stdout.write((json.dumps(payload, indent=2) if as_json else text) + "\n")


def _root(args: argparse.Namespace) -> Path:
    from equiparity.io.records import find_repo_root

    return Path(args.root).resolve() if args.root else find_repo_root()


# ---- handlers -------------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    from equiparity.io.config import load_experiment_config
    from equiparity.workflows.run_experiment import run_experiment

    config = load_experiment_config(args.config)
    run_dir = run_experiment(config, allow_dirty=args.allow_dirty, resume=args.resume)
    _log.info("run complete: %s", run_dir)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from equiparity.verification.release import CHECKS, run_release_checks

    selected = tuple(c for c in CHECKS if getattr(args, c))
    results = run_release_checks(_root(args), selected or ("theory", "claims"))
    lines = []
    for r in results:
        lines.append(f"[{'pass' if r.passed else 'FAIL'}] {r.name}")
        lines += [f"    {p}" for p in r.problems]
    _emit([r.to_dict() for r in results], as_json=args.json, text="\n".join(lines))
    return 0 if all(r.passed for r in results) else 1


def _cmd_records_list(args: argparse.Namespace) -> int:
    from equiparity.io.records import load_records

    records = load_records(_root(args))
    if args.item:
        records = [r for r in records if any(args.item in i for i in r.manuscript_items)]
    width = max((len(r.file) for r in records), default=0)
    text = "\n".join(f"{r.file:<{width}}  {'; '.join(r.manuscript_items)}" for r in records)
    _emit([r.to_dict() for r in records], as_json=args.json, text=text)
    return 0


def _cmd_records_show(args: argparse.Namespace) -> int:
    from equiparity.io.records import get_record

    root = _root(args)
    try:
        record = get_record(root, args.name)
    except KeyError as exc:
        _log.error("%s", exc.args[0])
        return 2
    payload = {**record.to_dict(), "path": str(root / "results" / record.file)}
    text = "\n".join(f"{k}: {v}" for k, v in payload.items())
    _emit(payload, as_json=args.json, text=text)
    return 0


def _cmd_grid(args: argparse.Namespace) -> int:
    from equiparity.workflows.grids import GRIDS, generate_grid

    root = _root(args)
    names = sorted(GRIDS) if args.name == "all" else [args.name]
    summary = {name: generate_grid(name, root) for name in names}
    counts = {name: {p: len(v) for p, v in runs.items()} for name, runs in summary.items()}
    text = "\n".join(
        f"{name}: {sum(c.values())} configs -> {GRIDS[name].directory}/ {c}"
        for name, c in counts.items()
    )
    _emit(counts, as_json=args.json, text=text)
    return 0


def _cmd_data(args: argparse.Namespace) -> int:
    from equiparity.workflows import prepare_data

    root = _root(args)
    if args.source == "qm9":
        prepare_data.prepare_qm9(root)
    elif args.source == "mp":
        prepare_data.prepare_mp(root, args.dataset)
    else:
        n, failed = prepare_data.idealize_ood(root)
        _log.info("idealized %d structures (%d kept raw)", n, failed)
    return 0


def _cmd_tensor_datasets(args: argparse.Namespace) -> int:
    from equiparity.workflows.tensor_datasets import prepare_tensor_dataset

    result = prepare_tensor_dataset(_root(args), args.target, args.documents)
    _emit(result, as_json=args.json, text=str(result))
    return 0


def _cmd_zero_supervision(args: argparse.Namespace) -> int:
    from equiparity.workflows.zero_supervision import prepare_zero_supervision

    result = prepare_zero_supervision(_root(args), args.documents)
    _emit(result, as_json=args.json, text=str(result))
    return 0


def _cmd_evaluate_checkpoint(args: argparse.Namespace) -> int:
    from equiparity.workflows.evaluate_checkpoint import evaluate_checkpoint

    result = evaluate_checkpoint(
        _root(args), args.run_dir, args.output_dir, args.device, args.partition, args.protocol_lock
    )
    _emit(result, as_json=args.json, text=str(result))
    return 0


def _cmd_evaluation_lock(args: argparse.Namespace) -> int:
    from equiparity.workflows.evaluation_protocol import lock_evaluation_protocol

    result = lock_evaluation_protocol(_root(args), args.configs, args.output)
    _emit(result, as_json=args.json, text=str(result))
    return 0


def _cmd_regression_metrics(args: argparse.Namespace) -> int:
    from equiparity.workflows.regression_metrics import summarize_regressions

    result = summarize_regressions(args.protocol_lock, args.evaluations, args.output)
    _emit(result, as_json=args.json, text=str(result))
    return 0


def _cmd_measure_masks(args: argparse.Namespace) -> int:
    from equiparity.workflows.symmetry_masks import measure_masks

    measurements = measure_masks(_root(args) / "results/predictions.zip")
    _emit(measurements, as_json=args.json, text=json.dumps(measurements, indent=2))
    return 0


def _cmd_crystal_symmetry(args: argparse.Namespace) -> int:
    from equiparity.workflows.crystal_symmetry import evaluate_crystal_symmetry

    result = evaluate_crystal_symmetry(_root(args), args.run_dir, args.output_dir, args.device)
    _emit(result, as_json=args.json, text=str(result))
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    from equiparity.workflows.run_manifest import write_run_manifest

    path = write_run_manifest(_root(args))
    _emit({"path": str(path)}, as_json=args.json, text=f"wrote {path}")
    return 0


# ---- parser ---------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root", type=Path, help="Repository root (default: nearest ancestor with results/)."
    )
    common.add_argument("--json", action="store_true", help="Machine-readable JSON output.")

    parser = argparse.ArgumentParser(
        prog="equiparity",
        description="Matched-pair parity experiments and release checks for "
        "'The parity gap in crystal tensor prediction'.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"equiparity {__version__}")
    parser.add_argument(
        "--json-logs", action="store_true", help="Emit JSON logs (for final-result runs)."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    run = sub.add_parser("run", help="Train and evaluate one experiment from a config YAML.")
    run.add_argument("config", type=Path, help="Path to an experiment config YAML.")
    run.add_argument(
        "--allow-dirty", action="store_true", help="Permit a dirty git tree (debug/smoke runs)."
    )
    run.add_argument(
        "--resume", type=Path, help="Resume a validation_selected epoch-boundary checkpoint."
    )
    run.set_defaults(handler=_cmd_run)

    evaluate = sub.add_parser(
        "evaluate-checkpoint",
        parents=[common],
        help="Reload a selected checkpoint on validation or protocol-locked test data.",
    )
    evaluate.add_argument("run_dir", type=Path)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--partition", choices=("val", "test"), default="val")
    evaluate.add_argument("--protocol-lock", type=Path)
    protocol = sub.add_parser("lock-evaluation", parents=[common])
    protocol.add_argument("configs", nargs="+", type=Path)
    protocol.add_argument("--output", type=Path, required=True)
    protocol.set_defaults(handler=_cmd_evaluation_lock)
    evaluate.set_defaults(handler=_cmd_evaluate_checkpoint)
    summary = sub.add_parser("summarize-regressions", parents=[common])
    summary.add_argument("evaluations", nargs="+", type=Path)
    summary.add_argument("--protocol-lock", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.set_defaults(handler=_cmd_regression_metrics)
    masks = sub.add_parser("measure-masks", parents=[common], help="Measure external tensor norms.")
    masks.set_defaults(handler=_cmd_measure_masks)
    parity_audit = sub.add_parser(
        "audit-crystal-symmetry",
        parents=[common],
        help="Trained and random per-class parity controls.",
    )
    parity_audit.add_argument("run_dir", type=Path)
    parity_audit.add_argument("--output-dir", type=Path, required=True)
    parity_audit.add_argument("--device", default="cuda")
    parity_audit.set_defaults(handler=_cmd_crystal_symmetry)

    verify = sub.add_parser(
        "verify",
        parents=[common],
        help="Check theory, claims and proofs against released records (no GPU).",
        description="Runs --theory and --claims when no check is selected.",
    )
    verify.add_argument("--theory", action="store_true", help="Parity-gap identities and table.")
    verify.add_argument("--claims", action="store_true", help="Reported values vs records.")
    verify.add_argument("--proofs", action="store_true", help="Build and audit the Lean proofs.")
    verify.set_defaults(handler=_cmd_verify)

    records = sub.add_parser("records", help="Browse the released result records.")
    records_sub = records.add_subparsers(dest="records_command", metavar="ACTION", required=True)
    rec_list = records_sub.add_parser("list", parents=[common], help="List records.")
    rec_list.add_argument("--item", help="Filter by manuscript item, e.g. 'Figure 2'.")
    rec_list.set_defaults(handler=_cmd_records_list)
    rec_show = records_sub.add_parser("show", parents=[common], help="Show one record.")
    rec_show.add_argument("name", help="Record file name, with or without extension.")
    rec_show.set_defaults(handler=_cmd_records_show)

    grid = sub.add_parser("grid", help="Experiment matrices.")
    grid_sub = grid.add_subparsers(dest="grid_command", metavar="ACTION", required=True)
    grid_gen = grid_sub.add_parser(
        "generate", parents=[common], help="Write a grid's configs and run lists."
    )
    grid_gen.add_argument(
        "name",
        choices=["all", "main", "zero-labels", "sum-pooled", "molecular"],
    )
    grid_gen.set_defaults(handler=_cmd_grid)

    data = sub.add_parser("data", help="Dataset preparation (needs the `data` extra).")
    data_sub = data.add_subparsers(dest="data_command", metavar="ACTION", required=True)
    tensor_datasets = data_sub.add_parser(
        "tensor",
        parents=[common],
        help="Validate property-associated structures and write crystal tensor data.",
    )
    tensor_datasets.add_argument("target", choices=["piezoelectric", "elastic"])
    tensor_datasets.add_argument(
        "--documents",
        type=Path,
        help="Archived MP property-document JSON; otherwise fetch with MP_KEY or MP_TOKEN.",
    )
    tensor_datasets.set_defaults(handler=_cmd_tensor_datasets)
    zero_labels = data_sub.add_parser(
        "zero-labels", parents=[common], help="Prepare disjoint zero-label supervision."
    )
    zero_labels.add_argument("--documents", type=Path)
    zero_labels.set_defaults(handler=_cmd_zero_supervision)
    prepare = data_sub.add_parser(
        "prepare", parents=[common], help="Build processed archives, manifests and splits."
    )
    prepare.add_argument(
        "source",
        choices=["qm9", "mp", "idealize"],
        help="qm9 (from extracted .xyz), mp (Materials Project; needs MP_TOKEN), or idealize "
        "(snap the centrosymmetric population onto its space groups).",
    )
    prepare.add_argument(
        "--dataset",
        choices=["all", "piezo", "ood"],
        default="all",
        help="Materials Project subset (mp only).",
    )
    prepare.set_defaults(handler=_cmd_data)

    manifest = sub.add_parser(
        "manifest", parents=[common], help="Rebuild results/run_manifest.json from configs."
    )
    manifest.set_defaults(handler=_cmd_manifest)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(
        level=logging.DEBUG if args.verbose else logging.INFO, json_logs=args.json_logs
    )
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return int(handler(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _log.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
