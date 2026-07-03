from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MPLCONFIGDIR = ROOT / ".matplotlib_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from inverse_construct_24_kept_points import load_kept_points_dataframe

OUTPUT_PATH = ROOT / "figures" / "inverse_seesaw" / "3p2" / "inverse_construct_24_dm41_dm51_acceptance.png"


def main() -> None:
    df = load_kept_points_dataframe()
    if df.empty:
        raise RuntimeError("CSV construct_24 introuvable ou vide.")
    df = df.copy()
    coherent = df.get("coherence_pass", pd.Series(1, index=df.index)).astype(int) == 1
    df = df.loc[coherent].copy()
    df["status"] = "all"
    df.loc[df["pmns_pass"].astype(int) == 1, "status"] = "PMNS"
    df.loc[(df["pmns_pass"].astype(int) == 1) & (df["eta_pass"].astype(int) == 1), "status"] = "PMNS+eta"

    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    specs = [("all", "0.75", 8, 0.35), ("PMNS", "tab:blue", 10, 0.55), ("PMNS+eta", "tab:red", 12, 0.75)]
    for label, color, size, alpha in specs:
        sub = df[df["status"] == label]
        if sub.empty:
            continue
        ax.scatter(sub["dm41_target_eV2"], sub["dm51_target_eV2"], s=size, c=color, alpha=alpha, edgecolors="none", label=label)
    ax.set_xlabel(r"$\Delta m^2_{41}$ [eV$^2$]")
    ax.set_ylabel(r"$\Delta m^2_{51}$ [eV$^2$]")
    ax.set_title("ISS(2,4) construct_24 - acceptance PMNS/eta (9x9 coherent)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220)
    print(f"Figure sauvegardee: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
