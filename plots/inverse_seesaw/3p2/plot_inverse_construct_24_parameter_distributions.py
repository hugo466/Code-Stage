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
import numpy as np

from inverse_construct_24_kept_points import load_kept_points_dataframe

OUTPUT_PATH = ROOT / "figures" / "inverse_seesaw" / "3p2" / "inverse_construct_24_parameter_distributions.png"


def parameter_schema():
    return [
        ("dm41_target_eV2", r"$\Delta m^2_{41}$ [eV$^2$]"),
        ("dm51_target_eV2", r"$\Delta m^2_{51}$ [eV$^2$]"),
        ("s1", r"$s_1$"),
        ("s2", r"$s_2$"),
        ("V_angle_deg", r"$\theta_V$ [deg]"),
        ("W_angle_deg", r"$\theta_W$ [deg]"),
        ("V_alpha_deg", r"$\alpha_V$ [deg]"),
        ("V_beta_deg", r"$\beta_V$ [deg]"),
        ("V_gamma_deg", r"$\gamma_V$ [deg]"),
        ("W_alpha_deg", r"$\alpha_W$ [deg]"),
        ("W_beta_deg", r"$\beta_W$ [deg]"),
        ("W_gamma_deg", r"$\gamma_W$ [deg]"),
        ("majorana_alpha21_deg", r"$\alpha_{21}$ [deg]"),
        ("majorana_alpha31_deg", r"$\alpha_{31}$ [deg]"),
        ("M1_GeV", r"$M_1$ [GeV]"),
        ("M2_GeV", r"$M_2$ [GeV]"),
        ("f11_abs", r"$|f_{11}|$"),
        ("f12_abs", r"$|f_{12}|$"),
        ("f21_abs", r"$|f_{21}|$"),
        ("f22_abs", r"$|f_{22}|$"),
        ("f11_phase_deg", r"$\phi_{f_{11}}$ [deg]"),
        ("f12_phase_deg", r"$\phi_{f_{12}}$ [deg]"),
        ("f21_phase_deg", r"$\phi_{f_{21}}$ [deg]"),
        ("f22_phase_deg", r"$\phi_{f_{22}}$ [deg]"),
    ]


def main() -> None:
    df = load_kept_points_dataframe()
    if df.empty:
        raise RuntimeError("CSV construct_24 introuvable ou vide.")

    coherent = df.get("coherence_pass", 1).astype(int) == 1
    pmns = (df.get("pmns_pass", 0).astype(int) == 1) & coherent
    eta = df.get("eta_pass", 0).astype(int) == 1
    df_pmns = df.loc[pmns].copy()
    if df_pmns.empty:
        raise RuntimeError("Aucun point PMNS coherent.")

    df_eta_pass = df_pmns.loc[eta.loc[df_pmns.index]].copy()
    df_eta_fail = df_pmns.loc[~eta.loc[df_pmns.index]].copy()

    schema = [(col, label) for col, label in parameter_schema() if col in df_pmns.columns]
    ncols = 4
    nrows = int(np.ceil(len(schema) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18.0, 3.5 * nrows))
    axes = np.asarray(axes).ravel()

    for ax, (col, label) in zip(axes, schema):
        pass_values = np.asarray(df_eta_pass[col], dtype=float)
        fail_values = np.asarray(df_eta_fail[col], dtype=float)
        pass_values = pass_values[np.isfinite(pass_values)]
        fail_values = fail_values[np.isfinite(fail_values)]
        all_values = np.concatenate([pass_values, fail_values])
        if all_values.size:
            bins = max(35, min(120, int(np.sqrt(all_values.size))))
            ax.hist(
                [pass_values, fail_values],
                bins=bins,
                density=True,
                stacked=True,
                color=["red", "#1f77b4"],
                alpha=0.82,
                linewidth=0.5,
                edgecolor="black",
                label=["PMNS+eta", "PMNS eta KO"],
            )
        ax.set_title(col, fontsize=9)
        ax.set_xlabel(label)
        ax.set_ylabel("Densite")
        ax.grid(alpha=0.25)

    for ax in axes[len(schema):]:
        ax.axis("off")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"ISS(2,4) construct_24 - distributions des parametres libres "
        f"({len(df_pmns)} PMNS; {len(df_eta_pass)} PMNS+eta)",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Figure sauvegardee: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
