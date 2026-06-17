#!/usr/bin/env python
"""Compare ND Fig.4-like source models: point, uniform, and dk2nu.

The script reads the spectra CSV files produced by scan_dune_nd_fig4.c and
prints/writes an integrated comparison table for each panel/component.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_BASE = Path("data/dune_nd/minimal_onaxis/point_70/plots_validation")
DEFAULT_INPUTS = {
    "point": DEFAULT_BASE / "fig4_nd_point_source_iss23_vs_nosterile.csv",
    "uniform": DEFAULT_BASE / "fig4_nd_source_line_iss23_vs_nosterile.csv",
    "dk2nu": DEFAULT_BASE / "fig4_nd_dk2nu_iss23_vs_nosterile.csv",
}
DEFAULT_OUT = DEFAULT_BASE / "fig4_nd_source_model_comparison.csv"

PANEL_TOTALS = {
    "FHC_app": ["nc", "numu", "beam", "signal"],
    "RHC_app": ["nc", "numu", "beam", "signal"],
    "FHC_dis": ["nc", "wrong_mu", "tau", "signal"],
    "RHC_dis": ["nc", "wrong_mu", "tau", "signal"],
}
COMPONENT_ORDER = ["total", "nc", "numu", "beam", "wrong_mu", "tau", "signal"]


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0.0)
    except ValueError:
        return 0.0


def read_spectrum(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"panel", "component", "Erec_GeV", "globes_events", "iss23_events"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path}: colonnes manquantes: {', '.join(sorted(missing))}")
        for row in reader:
            rows.append(
                {
                    "panel": row["panel"],
                    "component": row["component"],
                    "energy": as_float(row, "Erec_GeV"),
                    "globes": as_float(row, "globes_events"),
                    "iss": as_float(row, "iss23_events"),
                }
            )
    return rows


def aggregate(rows: list[dict[str, object]], panel: str, component: str) -> list[tuple[float, float, float]]:
    selected = PANEL_TOTALS[panel] if component == "total" else [component]
    bins: dict[float, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        if row["panel"] != panel or row["component"] not in selected:
            continue
        energy = float(row["energy"])
        bins[energy][0] += float(row["globes"])
        bins[energy][1] += float(row["iss"])
    return [(energy, values[0], values[1]) for energy, values in sorted(bins.items())]


def summarize(model: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    panels = sorted({str(row["panel"]) for row in rows}, key=lambda p: list(PANEL_TOTALS).index(p) if p in PANEL_TOTALS else 999)
    for panel in panels:
        components = ["total"] + sorted(
            {str(row["component"]) for row in rows if row["panel"] == panel},
            key=lambda c: COMPONENT_ORDER.index(c) if c in COMPONENT_ORDER else 999,
        )
        for component in components:
            points = aggregate(rows, panel, component)
            if not points:
                continue
            sum_3nu = sum(g for _, g, _ in points)
            sum_iss = sum(y for _, _, y in points)
            max_abs_rel = 0.0
            for _, g, y in points:
                if abs(g) > 1e-12:
                    max_abs_rel = max(max_abs_rel, abs((y - g) / g))
            out.append(
                {
                    "source_model": model,
                    "panel": panel,
                    "component": component,
                    "sum_3nu": sum_3nu,
                    "sum_iss": sum_iss,
                    "ratio_iss_over_3nu": sum_iss / sum_3nu if sum_3nu > 0.0 else 0.0,
                    "max_abs_bin_rel_iss_vs_3nu": max_abs_rel,
                }
            )
    return out


def add_cross_model_references(rows: list[dict[str, object]]) -> None:
    by_key = {
        (str(row["source_model"]), str(row["panel"]), str(row["component"])): row
        for row in rows
    }
    for row in rows:
        panel = str(row["panel"])
        component = str(row["component"])
        sum_iss = float(row["sum_iss"])
        for ref_model in ("uniform", "dk2nu"):
            ref = by_key.get((ref_model, panel, component))
            key = f"rel_sum_iss_vs_{ref_model}"
            if ref and float(ref["sum_iss"]) != 0.0:
                row[key] = (sum_iss - float(ref["sum_iss"])) / float(ref["sum_iss"])
            else:
                row[key] = ""


def fmt(value: object, percent: bool = False) -> str:
    if value == "":
        return ""
    number = float(value)
    if percent:
        return f"{100.0 * number:+.4f}%"
    return f"{number:.6g}"


def print_table(rows: list[dict[str, object]]) -> None:
    display = [row for row in rows if row["component"] == "total"]
    headers = [
        "model",
        "panel",
        "sum_3nu",
        "sum_iss",
        "ISS/3nu",
        "max bin rel",
        "ISS vs uniform",
        "ISS vs dk2nu",
    ]
    print(" | ".join(headers))
    print(" | ".join("-" * len(h) for h in headers))
    for row in display:
        print(
            " | ".join(
                [
                    str(row["source_model"]),
                    str(row["panel"]),
                    fmt(row["sum_3nu"]),
                    fmt(row["sum_iss"]),
                    fmt(row["ratio_iss_over_3nu"]),
                    fmt(row["max_abs_bin_rel_iss_vs_3nu"], percent=True),
                    fmt(row["rel_sum_iss_vs_uniform"], percent=True),
                    fmt(row["rel_sum_iss_vs_dk2nu"], percent=True),
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare les modeles de source ND point/uniform/dk2nu.")
    parser.add_argument("--point", type=Path, default=DEFAULT_INPUTS["point"])
    parser.add_argument("--uniform", type=Path, default=DEFAULT_INPUTS["uniform"])
    parser.add_argument("--dk2nu", type=Path, default=DEFAULT_INPUTS["dk2nu"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {"point": args.point, "uniform": args.uniform, "dk2nu": args.dk2nu}
    summaries: list[dict[str, object]] = []
    for model, path in paths.items():
        if not path.exists():
            print(f"[warn] fichier absent pour {model}: {path}")
            continue
        summaries.extend(summarize(model, read_spectrum(path)))
    if not summaries:
        raise SystemExit("Aucun CSV de spectre disponible pour la comparaison.")
    add_cross_model_references(summaries)

    fieldnames = [
        "source_model",
        "panel",
        "component",
        "sum_3nu",
        "sum_iss",
        "ratio_iss_over_3nu",
        "max_abs_bin_rel_iss_vs_3nu",
        "rel_sum_iss_vs_uniform",
        "rel_sum_iss_vs_dk2nu",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print_table(summaries)
    print(f"\nCSV comparatif ecrit: {args.out.resolve()}")


if __name__ == "__main__":
    main()
