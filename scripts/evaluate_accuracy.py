"""Per-class accuracy evaluation: AI traffic counts vs manual ground truth.

Produces the Bab IV tables for the thesis: one row per vehicle class plus a
Total row, with absolute error, signed difference and percentage accuracy.

Counts come either from JSON files (--gt/--pred) or straight from the CLI
(--gt-counts/--pred-counts in Motor,Mobil,Truk/Bus order). JSON is accepted in
any of the shapes this repo already writes: a flat {"motor": n, ...}, a
stats_store snapshot {"counts": {...}}, or a manual_counter export
{"motor": n, "mobil": n, "bis/truk": n}. Class aliases are normalized by
`normalize_counts`, so "motorcycle", "car", "truk", "bis/truk" and "truk/bus"
all land in the three canonical buckets.

Usage:
    python scripts/evaluate_accuracy.py --scenario "Malam Ramai" \
        --gt-counts 842,356,78 --pred-counts 812,341,74
    python scripts/evaluate_accuracy.py --gt gt.json --pred pred.json \
        --output reports/akurasi.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

# Canonical key -> label printed in the table.
CLASSES = (("motor", "Motor"), ("mobil", "Mobil"), ("truk_bus", "Truk/Bus"))
CLASS_KEYS = tuple(key for key, _ in CLASSES)
TOTAL_LABEL = "Total Kendaraan"
DEFAULT_TARGET = 90.0

# Every spelling the repo and manual_counter.py use for the same bucket.
ALIASES = {
    "motor": "motor", "motorcycle": "motor", "motorbike": "motor",
    "sepeda_motor": "motor",
    "mobil": "mobil", "car": "mobil", "mobil_penumpang": "mobil",
    "truk_bus": "truk_bus", "truk/bus": "truk_bus", "bis/truk": "truk_bus",
    "truk": "truk_bus", "truck": "truk_bus", "truck/bus": "truk_bus",
    "bus": "truk_bus", "bis": "truk_bus",
}


def normalize_counts(raw) -> dict:
    """Any count mapping/sequence -> {"motor", "mobil", "truk_bus"} ints."""
    out = {key: 0 for key in CLASS_KEYS}
    if raw is None:
        return out
    if isinstance(raw, Mapping):
        for name, value in raw.items():
            key = ALIASES.get(str(name).strip().lower().replace(" ", "_"))
            if key is not None:
                out[key] += max(0, _as_int(value))
        return out
    values = list(raw)
    for key, value in zip(CLASS_KEYS, values):
        out[key] = max(0, _as_int(value))
    return out


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def accuracy_percent(gt: int, pred: int) -> float:
    """max(0, (1 - |err| / gt) * 100), with the zero-GT convention spelled out."""
    if gt > 0:
        return max(0.0, (1.0 - abs(pred - gt) / gt) * 100.0)
    return 100.0 if pred == 0 else 0.0


def compute_accuracy_report(gt, pred, scenario: str = "",
                            target: float = DEFAULT_TARGET) -> dict:
    """Build the per-class report dict used by the table/JSON writers."""
    gt_counts = normalize_counts(gt)
    pred_counts = normalize_counts(pred)

    rows = []
    for key, label in CLASSES:
        g, p = gt_counts[key], pred_counts[key]
        rows.append({"class": label, "key": key, "gt": g, "pred": p,
                     "abs_error": abs(p - g), "signed_diff": p - g,
                     "accuracy": accuracy_percent(g, p)})

    gt_total = sum(gt_counts.values())
    pred_total = sum(pred_counts.values())
    rows.append({"class": TOTAL_LABEL, "key": "total", "gt": gt_total,
                 "pred": pred_total, "abs_error": abs(pred_total - gt_total),
                 "signed_diff": pred_total - gt_total,
                 "accuracy": accuracy_percent(gt_total, pred_total)})

    # Weighted = every vehicle counts once, so a big Motor bucket dominates.
    weighted = accuracy_percent(gt_total, pred_total)
    macro = sum(r["accuracy"] for r in rows[:-1]) / len(CLASS_KEYS)

    return {"scenario": scenario, "target": float(target),
            "rows": rows, "gt_total": gt_total, "pred_total": pred_total,
            "weighted_accuracy": weighted, "macro_accuracy": macro,
            "passed": weighted >= float(target)}


def format_markdown(report: dict) -> str:
    """Thesis-ready Markdown table (paste straight into Bab IV)."""
    title = "Akurasi Penghitungan Lalu Lintas"
    if report["scenario"]:
        title += f" - {report['scenario']}"
    lines = [
        f"### {title}",
        "",
        "| Kelas Kendaraan | Ground Truth | AI System | Absolute Error "
        "| Signed Diff | Akurasi (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        cells = [row["class"], row["gt"], row["pred"], row["abs_error"],
                 f"{row['signed_diff']:+d}", f"{row['accuracy']:.2f}"]
        if row["key"] == "total":
            cells = [f"**{cells[0]}**"] + [f"**{c}**" for c in cells[1:]]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    status = "PASSED" if report["passed"] else "FAILED"
    lines += [
        "",
        f"- Overall Weighted Accuracy: **{report['weighted_accuracy']:.2f}%**",
        f"- Macro-Average Class Accuracy: **{report['macro_accuracy']:.2f}%**",
        f"- Target (>= {report['target']:.1f}%): **{status}**",
    ]
    return "\n".join(lines)


def load_counts(path) -> dict:
    """Read a JSON count file, tolerating the repo's several wrapper keys."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("counts", "gt", "pred", "predicted", "prediction", "data"):
            inner = data.get(key)
            if isinstance(inner, (dict, list)):
                return normalize_counts(inner)
    return normalize_counts(data)


def parse_cli_counts(text: str) -> dict:
    """'842,356,78' -> canonical dict, in Motor,Mobil,Truk/Bus order."""
    parts = [p for p in str(text).replace(";", ",").split(",")]
    if len(parts) != len(CLASS_KEYS):
        raise ValueError(f"--counts needs {len(CLASS_KEYS)} values "
                         f"(Motor,Mobil,Truk/Bus), got {len(parts)}")
    return normalize_counts(parts)


def write_outputs(report: dict, path) -> Path:
    """CSV when the path ends in .csv, otherwise the full JSON report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Kelas Kendaraan", "Ground Truth", "AI System",
                             "Absolute Error", "Signed Diff", "Akurasi (%)"])
            for row in report["rows"]:
                writer.writerow([row["class"], row["gt"], row["pred"],
                                 row["abs_error"], row["signed_diff"],
                                 f"{row['accuracy']:.2f}"])
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluasi akurasi penghitungan lalu lintas AI vs manual")
    p.add_argument("--gt", help="JSON ground-truth counts")
    p.add_argument("--pred", help="JSON AI predicted counts")
    p.add_argument("--gt-counts", help="Manual: Motor,Mobil,Truk/Bus")
    p.add_argument("--pred-counts", help="AI: Motor,Mobil,Truk/Bus")
    p.add_argument("--scenario", default="", help='Label, e.g. "Malam Ramai"')
    p.add_argument("--target", type=float, default=DEFAULT_TARGET,
                   help="Accuracy threshold in percent (default 90)")
    p.add_argument("--output", help="Report path (.csv or .json)")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.gt and args.pred:
        gt, pred = load_counts(args.gt), load_counts(args.pred)
    elif args.gt_counts and args.pred_counts:
        gt, pred = parse_cli_counts(args.gt_counts), parse_cli_counts(
            args.pred_counts)
    else:
        raise SystemExit("berikan --gt + --pred, atau --gt-counts + "
                         "--pred-counts")

    report = compute_accuracy_report(gt, pred, args.scenario, args.target)
    print(format_markdown(report))
    if args.output:
        print(f"\nDisimpan ke: {write_outputs(report, args.output)}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
