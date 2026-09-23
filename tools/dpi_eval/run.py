"""
Run the DPI evaluation matrix.

    python -m tools.dpi_eval.run --quick
    python -m tools.dpi_eval.run --sources 369 371 --strategies native neural
    python -m tools.dpi_eval.run --dpis 300 75

Writes ``results/dpi_eval/<timestamp>/`` with a summary CSV, per-cell outputs,
one montage per (source, DPI) and a markdown report.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import metrics, report, sources
from .strategies import STRATEGIES, StrategyUnavailable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXAMPLE_DIR = REPO_ROOT / "example_images"
DEFAULT_OUT_ROOT = REPO_ROOT / "results" / "dpi_eval"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", default=list(sources.PRIMARY_SOURCES),
                        help="source image names without extension")
    parser.add_argument("--quick", action="store_true",
                        help=f"only the quick subset: {', '.join(sources.QUICK_SOURCES)}")
    parser.add_argument("--real-scans", action="store_true",
                        help="evaluate the four real scans of one drawing (lithic_75dpi to "
                             "lithic_600dpi), registered to the 600 DPI scan, instead of the crops")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES),
                        choices=list(STRATEGIES))
    parser.add_argument("--dpis", nargs="+", type=int, default=list(sources.DPIS))
    parser.add_argument("--reference-strategy", default="native", choices=list(STRATEGIES),
                        help="strategy whose output at the source DPI is the yardstick for every cell")
    parser.add_argument("--tolerance", type=float, default=metrics.TOLERANCE_PX,
                        help="skeleton match tolerance in pixels at the source DPI")
    parser.add_argument("--config", type=Path, default=None,
                        help="configuration file for this branch's strategies (sets LITHIC_EDITOR_CONFIG)")
    parser.add_argument("--example-dir", type=Path, default=DEFAULT_EXAMPLE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args(argv)
    if args.config is not None:
        os.environ["LITHIC_EDITOR_CONFIG"] = str(args.config.resolve())
    if args.quick:
        args.sources = list(sources.QUICK_SOURCES)
    args.dpis = sorted(set(args.dpis), reverse=True)
    if args.dpis[0] != sources.SOURCE_DPI:
        parser.error(f"--dpis must include the source DPI {sources.SOURCE_DPI}")
    if args.reference_strategy not in args.strategies:
        args.strategies = [args.reference_strategy] + args.strategies
    # The reference strategy runs first so its source-DPI output exists for every cell.
    args.strategies = [args.reference_strategy] + [s for s in args.strategies if s != args.reference_strategy]
    return args


def evaluate_source(
    name: str, args: argparse.Namespace, run_dir: Path, unavailable: set[str]
) -> list[dict]:
    """Run every strategy at every DPI for one source and return its rows."""
    source_dir = run_dir / name
    variants_dir = source_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    image = sources.load_grayscale(sources.source_path(name, args.example_dir))
    shape = image.shape
    ground_truth = None
    ground_truth_path = sources.ground_truth_path(name, args.example_dir)
    if ground_truth_path is not None:
        ground_truth = metrics.to_common_grid(sources.load_grayscale(ground_truth_path), shape)

    variant_paths: dict[int, Path] = {}
    variant_ink: dict[int, np.ndarray] = {}
    for dpi in args.dpis:
        variant = sources.make_variant(image, sources.SOURCE_DPI, dpi)
        variant_paths[dpi] = variants_dir / f"input_{dpi}.png"
        sources.save_with_dpi(variant, variant_paths[dpi], dpi)
        variant_ink[dpi] = metrics.to_common_grid(variant, shape)

    rows: list[dict] = []
    outputs: dict[str, dict[int, np.ndarray]] = {s: {} for s in args.strategies}
    reference: np.ndarray | None = None
    for strategy in args.strategies:
        if strategy in unavailable:
            continue
        for dpi in args.dpis:
            row = {"source": name, "strategy": strategy, "dpi": dpi}
            workdir = source_dir / "work" / f"{strategy}_{dpi}"
            workdir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            try:
                result = STRATEGIES[strategy](variant_paths[dpi], dpi, workdir)
            except StrategyUnavailable as exc:
                print(f"  skipping {strategy}: {exc}")
                unavailable.add(strategy)
                break
            except Exception as exc:  # one bad cell must not stop the matrix
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                rows.append(row)
                print(f"  {strategy:13s} {dpi:4d} DPI  ERROR {type(exc).__name__}")
                continue
            row["runtime_s"] = time.perf_counter() - started
            row.update(status="ok", output_width=result.shape[1], output_height=result.shape[0])
            output_dpi = round(sources.SOURCE_DPI * result.shape[0] / shape[0])
            sources.save_with_dpi(result, source_dir / f"{strategy}_{dpi}.png", output_dpi)

            ink = metrics.to_common_grid(result, shape)
            outputs[strategy][dpi] = ink
            row["residual_ripples"] = metrics.residual_ripples(metrics.skeleton(ink))
            is_reference_cell = strategy == args.reference_strategy and dpi == sources.SOURCE_DPI
            if is_reference_cell:
                reference = ink
            elif reference is not None:
                _add_scores(row, "ref", metrics.score_against_reference(ink, reference, args.tolerance))
            if ground_truth is not None:
                _add_scores(row, "gt", metrics.score_against_reference(ink, ground_truth, args.tolerance))
            rows.append(row)
            print(f"  {strategy:13s} {dpi:4d} DPI  {row['runtime_s']:5.1f}s  "
                  f"ripples={row['residual_ripples']:4d}  "
                  + (f"ref_f1={row['ref_f1']:.3f}" if "ref_f1" in row else "")
                  + (f"  gt_f1={row['gt_f1']:.3f}" if "gt_f1" in row else ""))

    for dpi in args.dpis:
        panels = [(f"input {dpi}", variant_ink[dpi])]
        if ground_truth is not None:
            panels.append(("ground truth", ground_truth))
        panels += [(s, outputs[s][dpi]) for s in args.strategies if dpi in outputs[s]]
        report.write_montage(panels, source_dir / f"montage_{dpi}.png")
    return rows


def evaluate_real_scans(args: argparse.Namespace, run_dir: Path, unavailable: set[str]) -> list[dict]:
    """
    Run every strategy on each real scan and score it against the reference strategy's
    output on the 600 DPI scan, after scaling by DPI ratio and rigid registration.
    """
    scan_dir = run_dir / "real_scans"
    scan_dir.mkdir(parents=True, exist_ok=True)
    reference_image = sources.load_grayscale(sources.source_path(sources.REAL_SCAN_REFERENCE, args.example_dir))
    reference_shape = reference_image.shape
    reference_dpi = max(dpi for _, dpi in sources.REAL_SCANS)

    # The scan-to-scan transform is a property of the scans, so it is estimated once per
    # DPI from the input scans, whose content is identical, and applied to every output.
    scale_of = {dpi: reference_dpi / dpi for _, dpi in sources.REAL_SCANS}
    inputs: dict[int, np.ndarray] = {}
    for name, dpi in sources.REAL_SCANS:
        source = sources.load_grayscale(sources.source_path(name, args.example_dir))
        scaled_shape = (round(source.shape[0] * scale_of[dpi]), round(source.shape[1] * scale_of[dpi]))
        inputs[dpi] = metrics.fit_to_shape(metrics.to_common_grid(source, scaled_shape), reference_shape)
    warps: dict[int, tuple[np.ndarray, float]] = {reference_dpi: (np.eye(2, 3, dtype=np.float32), 1.0)}
    for dpi in inputs:
        if dpi != reference_dpi:
            warps[dpi] = metrics.estimate_registration(inputs[dpi], inputs[reference_dpi])
            print(f"  registered {dpi} DPI scan to {reference_dpi}: correlation {warps[dpi][1]:.3f}")
    aligned_inputs = {dpi: metrics.apply_registration(inputs[dpi], warps[dpi][0], reference_shape) for dpi in inputs}
    for dpi, aligned in sorted(aligned_inputs.items(), reverse=True):
        if dpi != reference_dpi:
            ceiling = metrics.tolerance_scores(
                metrics.skeleton(aligned), metrics.skeleton(aligned_inputs[reference_dpi]), args.tolerance
            )[2]
            print(f"  ceiling at {dpi} DPI (input scan vs {reference_dpi} DPI input scan): F1 {ceiling:.3f}")

    rows: list[dict] = []
    reference: np.ndarray | None = None
    aligned_outputs: dict[tuple[str, int], np.ndarray] = {}
    for strategy in args.strategies:
        if strategy in unavailable:
            continue
        for name, dpi in sorted(sources.REAL_SCANS, key=lambda item: -item[1]):
            if dpi not in args.dpis:
                continue
            row = {"source": name, "strategy": strategy, "dpi": dpi}
            path = sources.source_path(name, args.example_dir)
            workdir = scan_dir / "work" / f"{strategy}_{dpi}"
            workdir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            try:
                result = STRATEGIES[strategy](path, dpi, workdir)
            except StrategyUnavailable as exc:
                print(f"  skipping {strategy}: {exc}")
                unavailable.add(strategy)
                break
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                rows.append(row)
                print(f"  {strategy:13s} {dpi:4d} DPI  ERROR {type(exc).__name__}")
                continue
            row["runtime_s"] = time.perf_counter() - started
            row.update(status="ok", output_width=result.shape[1], output_height=result.shape[0])
            sources.save_with_dpi(result, scan_dir / f"{strategy}_{dpi}.png", dpi)

            # Onto the 600 DPI grid: the input's size scaled by DPI ratio (a strategy may
            # return a larger image than its input), then the scan's transform.
            source_shape = sources.load_grayscale(path).shape
            scaled_shape = (round(source_shape[0] * scale_of[dpi]), round(source_shape[1] * scale_of[dpi]))
            ink = metrics.fit_to_shape(metrics.to_common_grid(result, scaled_shape), reference_shape)
            ink = metrics.apply_registration(ink, warps[dpi][0], reference_shape)
            row["registration_cc"] = warps[dpi][1]
            is_reference_cell = strategy == args.reference_strategy and dpi == reference_dpi
            if is_reference_cell:
                reference = ink
            elif reference is not None:
                _add_scores(row, "ref", metrics.score_against_reference(ink, reference, args.tolerance))
            aligned_outputs[(strategy, dpi)] = ink
            row["residual_ripples"] = metrics.residual_ripples(metrics.skeleton(ink))
            rows.append(row)
            print(f"  {strategy:13s} {dpi:4d} DPI  {row['runtime_s']:6.1f}s  ripples={row['residual_ripples']:4d}  "
                  + (f"ref_f1={row['ref_f1']:.3f}" if "ref_f1" in row else ""))

    for dpi in sorted({dpi for _, dpi in sources.REAL_SCANS if dpi in args.dpis}, reverse=True):
        panels = [(f"scan {dpi} (aligned)", aligned_inputs[dpi])]
        if reference is not None:
            panels.append((f"reference ({args.reference_strategy} 600)", reference))
        panels += [(s, aligned_outputs[(s, dpi)]) for s in args.strategies if (s, dpi) in aligned_outputs]
        report.write_montage(panels, scan_dir / f"montage_{dpi}.png")
    return rows


def _add_scores(row: dict, prefix: str, scores: metrics.ReferenceScores) -> None:
    row[f"{prefix}_precision"] = scores.precision
    row[f"{prefix}_recall"] = scores.recall
    row[f"{prefix}_f1"] = scores.f1
    row[f"{prefix}_structural_loss"] = scores.structural_loss
    row[f"{prefix}_cortex_ratio"] = scores.cortex_ratio


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {run_dir}")

    rows: list[dict] = []
    unavailable: set[str] = set()
    started = time.perf_counter()
    if args.real_scans:
        print("real scans:")
        rows.extend(evaluate_real_scans(args, run_dir, unavailable))
    else:
        for name in args.sources:
            print(f"{name}:")
            rows.extend(evaluate_source(name, args, run_dir, unavailable))

    report.write_csv(rows, run_dir / report.SUMMARY_FILENAME)
    previous_path = report.find_previous_summary(args.out_dir, run_dir)
    previous_rows = report.read_csv(previous_path) if previous_path else None
    strategies_run = [s for s in args.strategies if s not in unavailable]
    text = report.write_report(
        rows, run_dir / report.REPORT_FILENAME, strategies_run, args.dpis, previous_rows,
        args.reference_strategy,
    )
    print()
    print(text)
    print(f"Finished in {time.perf_counter() - started:.0f}s. Report: {run_dir / report.REPORT_FILENAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
