#!/usr/bin/env python3
"""
Final Leaderboard for Calendar Scheduling SLM comparison.

- Uses professor-provided evaluation script as a black box
- Uses REAL solve rate (no normalization, no rescaling)
- Ranks models by solve rate
- Produces a PDF leaderboard with clear justification
"""

import json
import sys
import subprocess
import re
import glob
import os
from typing import Dict, Any, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors


# =====================================================
# Helpers
# =====================================================

def extract_model_info(filepath: str):
    """Extract clean model name and parameter size."""
    with open(filepath) as f:
        data = json.load(f)

    sample = data[list(data.keys())[0]]
    model = sample.get("pred_model", "baseline")

    # Strip org/ and other prefixes
    if model.startswith("org/"):
        model = model[len("org/"):]

    if "/" in model:
        model = model.split("/")[-1]

    match = re.search(r"(\d+(?:\.\d+)?)([bm])", model.lower())
    params = f"{match.group(1)}{match.group(2).upper()}" if match else "N/A"

    return model, params


def run_evaluation(filepath: str) -> float:
    """
    Run the professor's evaluation script and return
    the REAL solve rate (e.g. 0.2290).
    """
    result = subprocess.run(
        ["python3", "evaluate_calendar_scheduling.py", f"--data_path={filepath}"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    output = result.stdout

    match = re.search(
        r"Overall solve rate of \d+ samples:\s*([\d.]+)",
        output
    )

    if not match:
        raise RuntimeError(
            f"Could not extract solve rate from evaluator output:\n{output}"
        )

    return float(match.group(1))


# =====================================================
# Ranking & Interpretation Logic
# =====================================================

def verdict_from_rank(rank: int) -> str:
    if rank == 1:
        return "Best overall"
    elif rank <= 3:
        return "Strong alternative"
    elif rank <= 5:
        return "Acceptable"
    else:
        return "Not recommended"


def strengths_and_limitations(rank: int, solve_rate: float):
    """
    Rank-aware qualitative interpretation.
    This is what justifies WHY the top models are chosen.
    """
    if rank == 1:
        strengths = [
            "Highest solve rate among all evaluated SLMs",
            "Demonstrates comparatively better handling of complex scheduling constraints",
        ]
        limitations = [
            "Overall solve rate remains low (<30%)",
            "Still far from solving most real-world scheduling cases",
        ]

    elif rank <= 3:
        strengths = [
            "Second-tier performance close to the top-ranked model",
            "Solves a non-trivial subset of scheduling instances",
        ]
        limitations = [
            "Clear performance gap compared to the top-ranked model",
            "Limited robustness on complex scheduling constraints",
        ]

    elif rank <= 5:
        strengths = [
            "Capable of solving simple scheduling instances",
            "Demonstrates basic understanding of the scheduling task",
        ]
        limitations = [
            "Low solve rate compared to higher-ranked models",
            "Fails on the majority of complex scheduling scenarios",
        ]

    else:
        strengths = [
            "Produces syntactically valid scheduling outputs",
        ]
        limitations = [
            "Fails to solve most scheduling instances",
            "Not suitable for practical use on this task",
        ]

    return strengths, limitations


# =====================================================
# Leaderboard Construction
# =====================================================

def build_leaderboard(files: List[str]) -> List[Dict[str, Any]]:
    rows = []

    for fp in files:
        if not os.path.exists(fp):
            print(f"⚠️ Warning: File not found, skipping: {fp}")
            continue

        model, params = extract_model_info(fp)
        solve_rate = run_evaluation(fp)

        rows.append({
            "model": model,
            "params": params,
            "solve_rate": solve_rate,
        })

    # Sort by REAL solve rate (descending)
    rows.sort(key=lambda x: x["solve_rate"], reverse=True)

    for i, r in enumerate(rows, start=1):
        r["rank"] = i
        r["verdict"] = verdict_from_rank(i)
        r["strengths"], r["limitations"] = strengths_and_limitations(
            i, r["solve_rate"]
        )

    return rows


# =====================================================
# PDF Export
# =====================================================

def export_pdf(rows: List[Dict[str, Any]]):
    doc = SimpleDocTemplate("leaderboard.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("<b>Calendar Scheduling Leaderboard</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    # ---- Leaderboard Table ----
    table_data = [
        ["Rank", "Model", "Params", "Solve Rate", "Verdict"]
    ]

    for r in rows:
        table_data.append([
            r["rank"],
            r["model"],
            r["params"],
            f"{r['solve_rate']:.4f}",
            r["verdict"],
        ])

    leaderboard_table = Table(table_data, repeatRows=1)
    leaderboard_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))

    elements.append(leaderboard_table)
    elements.append(Spacer(1, 20))

    # ---- Strengths & Limitations ----
    elements.append(Paragraph("<b>Strengths & Limitations</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    sl_data = [["Model", "Strengths", "Limitations"]]

    for r in rows:
        sl_data.append([
            Paragraph(r["model"], styles["Normal"]),
            Paragraph("<br/>".join(r["strengths"]), styles["Normal"]),
            Paragraph("<br/>".join(r["limitations"]), styles["Normal"]),
        ])

    sl_table = Table(sl_data, colWidths=[120, 220, 220], repeatRows=1)
    sl_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    elements.append(sl_table)

    doc.build(elements)
    print("✅ Exported leaderboard.pdf")


# =====================================================
# Entrypoint
# =====================================================

if __name__ == "__main__":
    files = sys.argv[1:] or glob.glob("data/output_*.json")

    leaderboard = build_leaderboard(files)

    # Console sanity output
    print("\nLEADERBOARD — Calendar Scheduling\n")
    for r in leaderboard:
        print(
            f"{r['rank']:>2}. {r['model']:<20} | {r['params']:<4} | "
            f"Solve Rate: {r['solve_rate']:.4f}"
        )

    export_pdf(leaderboard)
