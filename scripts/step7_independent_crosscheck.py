#!/usr/bin/env python3
"""Cross-check frozen Step-7 metrics with the preserved independent evaluator."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"${THESIS_CONTROL_ROOT}\Final_Core_Evaluation")
INDEPENDENT = Path(r"${THESIS_CONTROL_ROOT}\Official_QuranQA_TaskA_Evaluation\02_Evaluation_Pipeline\quranqa_taskA_independent_eval.py")
LOGS = ROOT / "Evaluator_Logs" / "Independent_Crosschecks"
LOGS.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader((ROOT / "CORE_RESULTS_ALL_SPLITS.csv").open("r", encoding="utf-8", newline="")))
checks = []
for row in rows:
    label = "_".join([row["Population"].replace("-", "_").replace(" ", "_"), row["System"].replace(" ", "_"), row["Split"]])
    output = LOGS / f"{label}.json"
    completed = subprocess.run(
        [sys.executable, str(INDEPENDENT), "--run", row["Run_Path"], "--qrels", row["Qrels_Path"], "--output", str(output)],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    (LOGS / f"{label}.log").write_text(completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""), encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Independent evaluator failed: {label}")
    observed = json.loads(output.read_text(encoding="utf-8"))["overall"]
    map_delta = float(observed["map_cut_10"]) - float(row["MAP_at_10"])
    mrr_delta = float(observed["recip_rank"]) - float(row["MRR_at_10"])
    checks.append({
        "condition": label,
        "official_scorer_map_cut_10": float(row["MAP_at_10"]),
        "independent_map_cut_10": float(observed["map_cut_10"]),
        "map_delta": map_delta,
        "official_scorer_recip_rank": float(row["MRR_at_10"]),
        "independent_recip_rank": float(observed["recip_rank"]),
        "mrr_delta": mrr_delta,
        "passed": max(abs(map_delta), abs(mrr_delta)) <= 1e-12,
    })

summary = {
    "preserved_independent_evaluator": str(INDEPENDENT),
    "conditions_checked": len(checks),
    "maximum_absolute_difference": max(max(abs(item["map_delta"]), abs(item["mrr_delta"])) for item in checks),
    "tolerance": 1e-12,
    "passed": all(item["passed"] for item in checks),
    "checks": checks,
}
if not summary["passed"]:
    raise RuntimeError("Independent evaluator cross-check failed")
(ROOT / "CORE_EVALUATOR_CROSSCHECK.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: summary[key] for key in ["conditions_checked", "maximum_absolute_difference", "passed"]}, indent=2))
