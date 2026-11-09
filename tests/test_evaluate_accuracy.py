"""Accuracy-report checks for scripts/evaluate_accuracy.py.

Run: python -m pytest tests/test_evaluate_accuracy.py
"""
import csv
import json

from scripts.evaluate_accuracy import (accuracy_percent,
                                       compute_accuracy_report,
                                       format_markdown, load_counts, main,
                                       normalize_counts, parse_cli_counts,
                                       write_outputs)


def test_night_peak_scenario_above_target():
    gt = {"motor": 842, "mobil": 356, "bis/truk": 78}     # manual_counter keys
    pred = {"motor": 812, "mobil": 341, "truk/bus": 74}   # ZoneCounter keys
    report = compute_accuracy_report(gt, pred, "Malam Ramai")

    rows = {r["key"]: r for r in report["rows"]}
    assert rows["motor"]["abs_error"] == 30 and rows["motor"]["signed_diff"] == -30
    assert rows["mobil"]["abs_error"] == 15
    assert rows["truk_bus"]["abs_error"] == 4
    assert round(rows["motor"]["accuracy"], 2) == 96.44
    assert round(rows["mobil"]["accuracy"], 2) == 95.79
    assert round(rows["truk_bus"]["accuracy"], 2) == 94.87

    # total row = sum of the classes, weighted accuracy follows the totals
    assert rows["total"]["gt"] == 1276 and rows["total"]["pred"] == 1227
    assert round(report["weighted_accuracy"], 2) == 96.16
    assert round(report["macro_accuracy"], 2) == 95.70
    assert report["passed"] is True

    table = format_markdown(report)
    assert "Malam Ramai" in table
    assert "| Motor | 842 | 812 | 30 | -30 | 96.44 |" in table
    assert "Overall Weighted Accuracy: **96.16%**" in table
    assert "PASSED" in table


def test_edge_cases_zero_and_over_under_counting():
    assert accuracy_percent(0, 0) == 100.0
    assert accuracy_percent(0, 5) == 0.0
    assert accuracy_percent(10, 10) == 100.0
    assert accuracy_percent(100, 95) == 95.0      # under-count
    assert accuracy_percent(100, 150) == 50.0     # over-count
    assert accuracy_percent(10, 40) == 0.0        # clamped at zero

    # threshold flag: 50% is under target, 95% passes
    fail = compute_accuracy_report({"motor": 100}, {"motor": 50})
    assert fail["passed"] is False
    assert "FAILED" in format_markdown(fail)
    ok = compute_accuracy_report({"motor": 100}, {"motor": 95})
    assert ok["passed"] is True

    # all-zero scenario is a pass (no vehicles seen, none predicted)
    empty = compute_accuracy_report({}, {})
    assert empty["weighted_accuracy"] == 100.0 and empty["passed"] is True


def test_normalize_counts_and_cli_parsing():
    assert normalize_counts({"motorcycle": 3, "car": 2, "truk": 1}) == {
        "motor": 3, "mobil": 2, "truk_bus": 1}
    assert normalize_counts({"counts": {"motor": 1}}) == {
        "motor": 0, "mobil": 0, "truk_bus": 0}   # wrapper keys are not counts
    assert normalize_counts(["7", "8", "9"]) == {
        "motor": 7, "mobil": 8, "truk_bus": 9}
    assert parse_cli_counts("842,356,78") == {
        "motor": 842, "mobil": 356, "truk_bus": 78}
    assert normalize_counts({"motor": -4, "mobil": "x"}) == {
        "motor": 0, "mobil": 0, "truk_bus": 0}


def test_json_roundtrip_and_cli(tmp_path, capsys):
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.json"
    gt_path.write_text(json.dumps({"motor": 120, "mobil": 60,
                                   "bis/truk": 20}), encoding="utf-8")
    # stats_store snapshot shape must load too
    pred_path.write_text(json.dumps({"counts": {"motor": 116, "mobil": 57,
                                                "truk": 19}}), encoding="utf-8")
    assert load_counts(gt_path) == {"motor": 120, "mobil": 60, "truk_bus": 20}
    assert load_counts(pred_path) == {"motor": 116, "mobil": 57, "truk_bus": 19}

    out_csv = tmp_path / "report.csv"
    code = main(["--gt", str(gt_path), "--pred", str(pred_path),
                 "--scenario", "Pagi Ramai", "--output", str(out_csv)])
    assert code == 0
    assert "Pagi Ramai" in capsys.readouterr().out

    with open(out_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "Kelas Kendaraan"
    assert len(rows) == 5                     # header + 3 classes + total
    assert rows[1][:3] == ["Motor", "120", "116"]
    assert rows[4][0] == "Total Kendaraan"
    assert rows[4][1:3] == ["200", "192"]

    # CLI counts path + JSON output
    out_json = tmp_path / "report.json"
    assert main(["--gt-counts", "842,356,78", "--pred-counts", "812,341,74",
                 "--output", str(out_json)]) == 0
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["rows"][0]["class"] == "Motor"
    assert round(report["weighted_accuracy"], 2) == 96.16
    assert write_outputs(report, tmp_path / "sub" / "r.json").exists()
