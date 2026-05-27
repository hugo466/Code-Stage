import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from inverse_pmns_filter_3p2_config import get_inverse_kept_points_dir

DATA_DIR = get_inverse_kept_points_dir()
OUTPUT_PATH = Path("figures/inverse_seesaw/3p2/inverse_pmns_filter_3p2_parameter_distributions.png")
FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# All free parameters of ISS(2,3+2):
#   M_2x2_GeV   : M11, M12, M21, M22                    (4 entries)
#   mD_3x2_GeV  : mD11..mD32                           (6 entries)
#   mu00_2x2_eV : symmetric → 11, 12, 22               (3 entries)
#   mu_H_2x2_eV : symmetric → 11, 12, 22               (3 entries)
#   mu_H0_2x2_eV: general  → 11, 12, 21, 22            (4 entries)
ALL_KEYS = [
    "M11", "M12", "M21", "M22",
    "mD11", "mD12", "mD21", "mD22", "mD31", "mD32",
    "mu00_11", "mu00_12", "mu00_22",
    "muH_11", "muH_12", "muH_22",
    "muH0_11", "muH0_12", "muH0_21", "muH0_22",
]

LABELS = {
    "M11": r"$M_{11}$ [GeV]", "M12": r"$M_{12}$ [GeV]",
    "M21": r"$M_{21}$ [GeV]", "M22": r"$M_{22}$ [GeV]",
    "mD11": r"$m_{D,11}$ [GeV]", "mD12": r"$m_{D,12}$ [GeV]",
    "mD21": r"$m_{D,21}$ [GeV]", "mD22": r"$m_{D,22}$ [GeV]",
    "mD31": r"$m_{D,31}$ [GeV]", "mD32": r"$m_{D,32}$ [GeV]",
    "mu00_11": r"$\mu_{0,11}$ [eV]", "mu00_12": r"$\mu_{0,12}$ [eV]",
    "mu00_22": r"$\mu_{0,22}$ [eV]",
    "muH_11": r"$\mu_{H,11}$ [eV]", "muH_12": r"$\mu_{H,12}$ [eV]",
    "muH_22": r"$\mu_{H,22}$ [eV]",
    "muH0_11": r"$\mu_{H0,11}$ [eV]", "muH0_12": r"$\mu_{H0,12}$ [eV]",
    "muH0_21": r"$\mu_{H0,21}$ [eV]", "muH0_22": r"$\mu_{H0,22}$ [eV]",
}


def parse_all_floats(text: str):
    rhs = text.split("=", 1)[1] if "=" in text else text
    return [float(x) for x in FLOAT_PATTERN.findall(rhs)]


def load_distributions(data_dir: Path):
    eta_pass_vals = {k: [] for k in ALL_KEYS}
    eta_fail_vals = {k: [] for k in ALL_KEYS}

    for point_file in sorted(data_dir.glob("*.txt"), key=lambda p: int(p.stem)):
        eta_pass = 0
        values = {}
        for line in point_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("eta_pass"):
                eta_pass = int(float(line.split("=", 1)[1].strip()))
            elif line.startswith("M_2x2_GeV"):
                v = parse_all_floats(line)
                if len(v) >= 4:
                    values["M11"] = abs(v[0])
                    values["M12"] = abs(v[1])
                    values["M21"] = abs(v[2])
                    values["M22"] = abs(v[3])
            elif line.startswith("mD_3x2_GeV"):
                v = parse_all_floats(line)
                if len(v) >= 6:
                    values["mD11"] = abs(v[0])
                    values["mD12"] = abs(v[1])
                    values["mD21"] = abs(v[2])
                    values["mD22"] = abs(v[3])
                    values["mD31"] = abs(v[4])
                    values["mD32"] = abs(v[5])
            elif line.startswith("mu00_2x2_eV"):
                v = parse_all_floats(line)
                if len(v) >= 4:
                    values["mu00_11"] = abs(v[0])
                    values["mu00_12"] = abs(v[1])
                    values["mu00_22"] = abs(v[3])
            elif line.startswith("mu_H0_2x2_eV"):
                v = parse_all_floats(line)
                if len(v) >= 4:
                    values["muH0_11"] = abs(v[0])
                    values["muH0_12"] = abs(v[1])
                    values["muH0_21"] = abs(v[2])
                    values["muH0_22"] = abs(v[3])
            elif line.startswith("mu_H_2x2_eV"):
                v = parse_all_floats(line)
                if len(v) >= 4:
                    values["muH_11"] = abs(v[0])
                    values["muH_12"] = abs(v[1])
                    values["muH_22"] = abs(v[3])

        if all(k in values for k in ALL_KEYS):
            target = eta_pass_vals if eta_pass == 1 else eta_fail_vals
            for k, val in values.items():
                target[k].append(val)

    return eta_pass_vals, eta_fail_vals


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dossier introuvable: {DATA_DIR}")

    eta_pass_vals, eta_fail_vals = load_distributions(DATA_DIR)

    n_params = len(ALL_KEYS)
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols  # ceil → 5 rows (20 params)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, key in enumerate(ALL_KEYS):
        ax = axes[idx]
        vals_pass = eta_pass_vals[key]
        vals_fail = eta_fail_vals[key]
        all_vals = vals_pass + vals_fail
        if not all_vals:
            ax.set_title(f"{key} (aucune donnée)")
            continue
        bins = max(30, min(120, int(len(all_vals) ** 0.5)))
        ax.hist([vals_pass, vals_fail], bins=bins, density=True, stacked=True,
                color=["red", "#1f77b4"], alpha=0.8, linewidth=0.6, edgecolor="black")
        ax.set_xlabel(LABELS.get(key, key), fontsize=9)
        ax.set_ylabel("densité", fontsize=8)
        ax.grid(alpha=0.25)

    # Hide unused axes
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)

    axes[0].legend(["PMNS + η", "PMNS sans η"], loc="best", framealpha=0.9, fontsize=8)
    fig.suptitle("ISS 3+2 — Distributions de tous les paramètres libres", fontsize=13)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Figure sauvegardée: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
