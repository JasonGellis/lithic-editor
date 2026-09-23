"""CSV, montage and markdown output for one evaluation run."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from lithic_editor.processing.resolution import measure_line_geometry

COLUMNS = [
    "source", "strategy", "dpi", "status", "runtime_s", "output_width", "output_height",
    "residual_ripples", "registration_cc",
    "ref_precision", "ref_recall", "ref_f1", "ref_structural_loss", "ref_cortex_ratio",
    "gt_precision", "gt_recall", "gt_f1", "gt_structural_loss", "gt_cortex_ratio",
    "error",
]
SUMMARY_FILENAME = "summary.csv"
REPORT_FILENAME = "report.md"
_MONTAGE_MAX_WIDTH = 3000
_LABEL_HEIGHT = 56
_LABEL_MARGIN = 10
_SEPARATOR_WIDTH = 6


def write_csv(rows: list[dict], path: Path) -> None:
    """Write one row per (source, strategy, dpi)."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_cell(row.get(column)) for column in COLUMNS})


def read_csv(path: Path) -> list[dict]:
    """Read a summary written by ``write_csv`` with numeric columns parsed."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for column, value in row.items():
            if column in ("source", "strategy", "status", "error"):
                continue
            row[column] = float(value) if value not in ("", None) else float("nan")
        row["dpi"] = int(row["dpi"])
    return rows


def find_previous_summary(out_root: Path, current: Path) -> Path | None:
    """The most recent earlier run's summary under ``out_root``, if any."""
    candidates = sorted(
        path for path in out_root.glob(f"*/{SUMMARY_FILENAME}")
        if path.parent != current and path.parent.name < current.name
    )
    return candidates[-1] if candidates else None


def write_montage(panels: list[tuple[str, np.ndarray]], path: Path) -> None:
    """
    Side-by-side panels of ink masks on the common grid, each with a label above it.

    ``panels`` is a list of (label, bool mask). The image row is scaled to at most
    ``_MONTAGE_MAX_WIDTH`` pixels wide first, and the labels are drawn afterwards at
    a fixed size so they stay readable whatever the scale.
    """
    if not panels:
        return
    tiles = []
    for _, mask in panels:
        image = np.where(mask, 0, 255).astype(np.uint8)
        tiles.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        tiles.append(np.full((image.shape[0], _SEPARATOR_WIDTH, 3), 120, dtype=np.uint8))
    row = np.hstack(tiles[:-1])
    scale = min(1.0, _MONTAGE_MAX_WIDTH / row.shape[1])
    if scale < 1.0:
        size = (round(row.shape[1] * scale), max(1, round(row.shape[0] * scale)))
        row = cv2.resize(row, size, interpolation=cv2.INTER_AREA)

    bar = np.full((_LABEL_HEIGHT, row.shape[1], 3), 230, dtype=np.uint8)
    x = 0
    for label, mask in panels:
        panel_width = round(mask.shape[1] * scale)
        # Measured line width in the label, since the scaled panel cannot show it
        width = measure_line_geometry(np.where(mask, 0, 255).astype(np.uint8)).line_width
        if not np.isnan(width):
            label = f"{label}  {width:.0f}px"
        font_scale = _fit_font_scale(label, panel_width - 2 * _LABEL_MARGIN)
        cv2.putText(bar, label, (x + _LABEL_MARGIN, _LABEL_HEIGHT - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        x += panel_width + round(_SEPARATOR_WIDTH * scale)
    cv2.imwrite(str(path), np.vstack([bar, row]))


def _fit_font_scale(text: str, width: int, preferred: float = 1.2, minimum: float = 0.6) -> float:
    """Largest font scale up to ``preferred`` at which ``text`` fits in ``width`` pixels."""
    scale = preferred
    while scale > minimum:
        (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        if text_width <= width:
            break
        scale -= 0.1
    return max(scale, minimum)


def summarize(rows: list[dict], prefix: str) -> dict[tuple[str, int], dict[str, float]]:
    """
    Mean of each ``prefix`` metric per (strategy, dpi) over sources with ok status.

    Rows whose metric is NaN are left out of that metric's mean.
    """
    metrics = ["precision", "recall", "f1", "structural_loss", "cortex_ratio"]
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (row["strategy"], int(row["dpi"]))
        for metric in metrics:
            value = row.get(f"{prefix}_{metric}")
            if value is not None and not _is_nan(value):
                grouped[key][metric].append(float(value))
        for metric in ("residual_ripples", "runtime_s"):
            value = row.get(metric)
            if value is not None and not _is_nan(value):
                grouped[key][metric].append(float(value))
    return {
        key: {metric: sum(values) / len(values) for metric, values in per_key.items() if values}
        for key, per_key in grouped.items()
    }


def render_table(summary: dict, strategies: list[str], dpis: list[int], metric: str) -> str:
    """Markdown table of one metric, strategies as rows and DPIs as columns."""
    header = "| strategy | " + " | ".join(f"{dpi} DPI" for dpi in dpis) + " |"
    rule = "|---|" + "|".join("---:" for _ in dpis) + "|"
    lines = [header, rule]
    for strategy in strategies:
        cells = []
        for dpi in dpis:
            value = summary.get((strategy, dpi), {}).get(metric)
            cells.append("" if value is None else f"{value:.3f}")
        lines.append(f"| {strategy} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    rows: list[dict], path: Path, strategies: list[str], dpis: list[int],
    previous_rows: list[dict] | None, reference: str,
) -> str:
    """Write the markdown report and return its text."""
    ref_summary = summarize(rows, "ref")
    gt_summary = summarize(rows, "gt")
    source_dpi = max(dpis)
    sections = ["# DPI evaluation report", ""]

    sections += [
        f"## Agreement with the reference: `{reference}` at {source_dpi} DPI", "",
        "Every cell is scored against the same reference output for its source, so",
        "strategies are comparable with each other. The reference cell itself is blank.", "",
        "Skeleton F1 at tolerance, mean over sources:", "",
        render_table(ref_summary, strategies, dpis, "f1"), "",
        "Precision (how much of the output is near the reference):", "",
        render_table(ref_summary, strategies, dpis, "precision"), "",
        "Recall (how much of the reference was kept):", "",
        render_table(ref_summary, strategies, dpis, "recall"), "",
        "Structural loss (fraction of reference ink not reproduced):", "",
        render_table(ref_summary, strategies, dpis, "structural_loss"), "",
    ]
    if any("f1" in values for values in gt_summary.values()):
        gt_strategies = sorted({key[0] for key in gt_summary}, key=strategies.index)
        sections += [
            "## Ground truth (reference: hand-cleaned drawing)", "",
            "Skeleton F1 at tolerance, mean over sources with a ground truth:", "",
            render_table(gt_summary, gt_strategies, dpis, "f1"), "",
            "Precision (how much of the output is real structure):", "",
            render_table(gt_summary, gt_strategies, dpis, "precision"), "",
            "Recall (how much real structure survived):", "",
            render_table(gt_summary, gt_strategies, dpis, "recall"), "",
        ]
    sections += [
        "## Reference-free", "",
        "Residual ripples (short free-ended skeleton segments), mean per source.",
        "Counts stubs left where hatching was cut as well as surviving hatch lines,",
        "so compare within a strategy across DPI rather than between strategies.", "",
        render_table(ref_summary, strategies, dpis, "residual_ripples"), "",
        "Runtime in seconds, mean per source:", "",
        render_table(ref_summary, strategies, dpis, "runtime_s"), "",
    ]

    sections += [f"## Best strategy per DPI (F1 against `{reference}` at {source_dpi} DPI)", ""]
    for dpi in dpis:
        ranked = sorted(
            ((ref_summary.get((s, dpi), {}).get("f1"), s) for s in strategies),
            key=lambda item: -1 if item[0] is None else item[0], reverse=True,
        )
        best = [f"{name} ({value:.3f})" for value, name in ranked if value is not None]
        sections.append(f"- {dpi} DPI: " + (", ".join(best[:3]) if best else "no data"))
    sections.append("")

    if previous_rows:
        # Compare only sources both runs evaluated, so a quick run and a full run
        # are not set against each other.
        common_sources = {row["source"] for row in rows} & {row["source"] for row in previous_rows}
        previous = summarize([r for r in previous_rows if r["source"] in common_sources], "ref")
        current = summarize([r for r in rows if r["source"] in common_sources], "ref")
        deltas = {
            key: {"f1": values["f1"] - previous[key]["f1"]}
            for key, values in current.items()
            if key in previous and "f1" in values and "f1" in previous[key]
        }
        sections += [
            "## Change in reference F1 since the previous run", "",
            f"Over the {len(common_sources)} source(s) present in both runs.", "",
            render_table(deltas, strategies, dpis, "f1"), "",
        ]

    failures = [row for row in rows if row["status"] != "ok"]
    if failures:
        sections += ["## Cells that did not run", ""]
        for row in failures:
            sections.append(
                f"- {row['source']} / {row['strategy']} / {row['dpi']} DPI: "
                f"{row['status']} {row.get('error', '')}".rstrip()
            )
        sections.append("")

    text = "\n".join(sections)
    path.write_text(text, encoding="utf-8")
    return text


def _format_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.6g}"
    return str(value)


def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)
