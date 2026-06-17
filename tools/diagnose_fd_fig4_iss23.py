import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("data/dune_nd/minimal_onaxis/point_70/plots_validation/fig4_fd_iss23_vs_globes.csv")
DEFAULT_SUMMARY = Path("data/dune_nd/minimal_onaxis/point_70/plots_validation/fig4_fd_iss23_diagnostics.csv")
DEFAULT_TOP_BINS = Path("data/dune_nd/minimal_onaxis/point_70/plots_validation/fig4_fd_iss23_largest_bin_diffs.csv")


def read_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            rows.append(
                {
                    "point_id": int(row["point_id"]),
                    "panel": row["panel"],
                    "component": row["component"],
                    "Erec_GeV": float(row["Erec_GeV"]),
                    "globes_events": float(row["globes_events"]),
                    "iss23_events": float(row["iss23_events"]),
                    "rel_diff": float(row["rel_diff"]),
                }
            )
    return rows


def summarize_group(rows):
    globes_sum = sum(row["globes_events"] for row in rows)
    iss_sum = sum(row["iss23_events"] for row in rows)
    delta = iss_sum - globes_sum
    ratio = iss_sum / globes_sum if globes_sum > 0.0 else float("nan")
    stat_pull = delta / math.sqrt(globes_sum) if globes_sum > 0.0 else float("nan")

    chi2 = 0.0
    shape_chi2 = 0.0
    max_abs_rel = 0.0
    max_abs_rel_energy = 0.0
    for row in rows:
        g = row["globes_events"]
        y = row["iss23_events"]
        if g > 0.0:
            chi2 += (y - g) * (y - g) / g
            y_norm = ratio * g if math.isfinite(ratio) else y
            shape_chi2 += (y - y_norm) * (y - y_norm) / g
        if abs(row["rel_diff"]) > max_abs_rel:
            max_abs_rel = abs(row["rel_diff"])
            max_abs_rel_energy = row["Erec_GeV"]

    return {
        "globes_events_sum": globes_sum,
        "iss23_events_sum": iss_sum,
        "delta_events": delta,
        "ratio_iss_over_globes": ratio,
        "stat_pull_delta_over_sqrt_globes": stat_pull,
        "chi2_stat_diag": chi2,
        "shape_chi2_after_free_norm": shape_chi2,
        "shape_fraction": shape_chi2 / chi2 if chi2 > 0.0 else 0.0,
        "max_abs_rel_bin": max_abs_rel,
        "max_abs_rel_bin_Erec_GeV": max_abs_rel_energy,
        "n_bins": len(rows),
    }


def write_summary(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "level",
        "panel",
        "component",
        "n_bins",
        "globes_events_sum",
        "iss23_events_sum",
        "delta_events",
        "ratio_iss_over_globes",
        "stat_pull_delta_over_sqrt_globes",
        "chi2_stat_diag",
        "shape_chi2_after_free_norm",
        "shape_fraction",
        "max_abs_rel_bin",
        "max_abs_rel_bin_Erec_GeV",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_top_bins(path: Path, rows, top_n: int):
    ranked = sorted(
        rows,
        key=lambda row: abs(row["iss23_events"] - row["globes_events"]),
        reverse=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "panel",
        "component",
        "Erec_GeV",
        "globes_events",
        "iss23_events",
        "delta_events",
        "rel_diff",
        "stat_pull_bin",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in ranked[:top_n]:
            g = row["globes_events"]
            delta = row["iss23_events"] - g
            writer.writerow(
                {
                    "panel": row["panel"],
                    "component": row["component"],
                    "Erec_GeV": f"{row['Erec_GeV']:.10g}",
                    "globes_events": f"{g:.12g}",
                    "iss23_events": f"{row['iss23_events']:.12g}",
                    "delta_events": f"{delta:.12g}",
                    "rel_diff": f"{row['rel_diff']:.12g}",
                    "stat_pull_bin": f"{delta / math.sqrt(g):.12g}" if g > 0.0 else "nan",
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic numerique des ecarts ISS(2,3) point 70 vs spectres Fig.4 GLoBES."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--top-bins", type=Path, default=DEFAULT_TOP_BINS)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    rows = read_rows(args.input)
    groups = defaultdict(list)
    panels = defaultdict(list)
    for row in rows:
        groups[(row["panel"], row["component"])].append(row)
        panels[row["panel"]].append(row)

    summary_rows = []
    for panel in sorted(panels):
        summary = summarize_group(panels[panel])
        summary_rows.append({"level": "panel_total", "panel": panel, "component": "all", **summary})

    for (panel, component) in sorted(groups):
        summary = summarize_group(groups[(panel, component)])
        summary_rows.append({"level": "component", "panel": panel, "component": component, **summary})

    write_summary(args.summary, summary_rows)
    write_top_bins(args.top_bins, rows, args.top_n)

    print(f"Diagnostics sauvegardes: {args.summary}")
    print(f"Plus grands ecarts bin par bin: {args.top_bins}")
    print("\nResume panneaux:")
    for row in summary_rows:
        if row["level"] != "panel_total":
            continue
        print(
            f"  {row['panel']}: ratio={row['ratio_iss_over_globes']:.5g}, "
            f"delta={row['delta_events']:.5g} evt, "
            f"pull={row['stat_pull_delta_over_sqrt_globes']:.4g} sigma, "
            f"chi2_diag={row['chi2_stat_diag']:.5g}, "
            f"shape_fraction={row['shape_fraction']:.3g}"
        )


if __name__ == "__main__":
    main()
