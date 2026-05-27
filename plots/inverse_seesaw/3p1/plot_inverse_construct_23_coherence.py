import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path("data/inverse_seesaw/3p1/inverse_construct_23_3p1.csv")
OUTPUT_PATH = Path("figures/inverse_seesaw/3p1/inverse_construct_23_coherence.png")


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
    return rows


def to_float_array(rows, key):
    values = []
    for row in rows:
        text = row.get(key, "")
        if text is None or text == "" or text.lower() == "nan":
            continue
        values.append(float(text))
    return np.array(values, dtype=float)


def to_int_array(rows, key):
    values = []
    for row in rows:
        text = row.get(key, "")
        if text is None or text == "":
            continue
        values.append(int(float(text)))
    return np.array(values, dtype=int)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV introuvable: {CSV_PATH}")

    rows = load_rows(CSV_PATH)
    if not rows:
        raise RuntimeError("CSV vide.")

    ok_rows = [row for row in rows if row.get("solve_ok", "0") == "1" and row.get("pmns_pass", "0") == "1"]
    if not ok_rows:
        raise RuntimeError("Aucun point pmns_pass=1 dans le CSV.")

    dm41_target = to_float_array(ok_rows, "dm41_target_eV2")
    dm41_calc = to_float_array(ok_rows, "dm41_calc_eV2")
    pmns_err = to_float_array(ok_rows, "pmns_rms_abs_error")
    ml_rel = to_float_array(ok_rows, "mL_rel_frob_error")
    eta_pass = to_int_array(ok_rows, "eta_pass")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    ax0 = axes[0]
    pass_mask = eta_pass == 1
    fail_mask = eta_pass == 0
    if np.any(pass_mask):
        ax0.scatter(dm41_target[pass_mask], dm41_calc[pass_mask], s=10, alpha=0.5, color="red", label=r"PMNS + $\eta$ OK")
    if np.any(fail_mask):
        ax0.scatter(dm41_target[fail_mask], dm41_calc[fail_mask], s=10, alpha=0.5, color="#1f77b4", label=r"PMNS OK, $\eta$ non OK")
    lo = max(min(dm41_target.min(), dm41_calc.min()), 1e-12)
    hi = max(dm41_target.max(), dm41_calc.max())
    ax0.plot([lo, hi], [lo, hi], "k--", lw=1.0)
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel(r"$\Delta m_{41,\,target}^2\,[\mathrm{eV}^2]$")
    ax0.set_ylabel(r"$\Delta m_{41,\,calc}^2\,[\mathrm{eV}^2]$")
    ax0.set_title("Cohérence masse stérile")
    ax0.legend(loc="best", framealpha=0.9)
    ax0.grid(alpha=0.25)

    ax1 = axes[1]
    if np.any(pass_mask):
        ax1.scatter(pmns_err[pass_mask], ml_rel[pass_mask], s=10, alpha=0.5, color="red", label=r"PMNS + $\eta$ OK")
    if np.any(fail_mask):
        ax1.scatter(pmns_err[fail_mask], ml_rel[fail_mask], s=10, alpha=0.5, color="#1f77b4", label=r"PMNS OK, $\eta$ non OK")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Erreur RMS sur $|U_{PMNS}|$")
    ax1.set_ylabel(r"Erreur relative $M_L$ (Frobenius)")
    ax1.set_title("Cohérence globale du modèle")
    ax1.legend(loc="best", framealpha=0.9)
    ax1.grid(alpha=0.25)

    n_pmns = len(ok_rows)
    n_total = len(rows)
    n_eta = int(np.sum(eta_pass == 1))
    fig.suptitle(f"Construction adaptée (2,3) — PMNS OK: {n_pmns}/{n_total} pts, Rouge=+η OK ({n_eta})")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Figure sauvegardée: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
