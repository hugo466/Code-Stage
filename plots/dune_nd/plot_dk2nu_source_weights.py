from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FHC_CSV = ROOT / "data" / "dune" / "dk2nu" / "source_profile_z_FHC_ND.csv"
RHC_CSV = ROOT / "data" / "dune" / "dk2nu" / "source_profile_z_RHC_ND.csv"
OUT = ROOT / "figures" / "dune_nd" / "flux" / "dk2nu_source_weights_nd.png"

FLAVOR_LABELS = {
    "nue": r"$\nu_e$",
    "numu": r"$\nu_\mu$",
    "nuebar": r"$\bar{\nu}_e$",
    "numubar": r"$\bar{\nu}_\mu$",
}

FLAVOR_COLORS = {
    "nue": "tab:orange",
    "numu": "tab:blue",
    "nuebar": "tab:red",
    "numubar": "tab:green",
}


def load_profile(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [
        "E_GeV_bin_low",
        "E_GeV_bin_high",
        "z_decay_m_bin_low",
        "z_decay_m_bin_high",
        "weight",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols + ["flavor"]).copy()
    df["E_center_GeV"] = 0.5 * (df["E_GeV_bin_low"] + df["E_GeV_bin_high"])
    df["z_center_m"] = 0.5 * (df["z_decay_m_bin_low"] + df["z_decay_m_bin_high"])
    df["delta_z_m"] = df["z_decay_m_bin_high"] - df["z_decay_m_bin_low"]
    return df


def heatmap_table(df: pd.DataFrame, flavor: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = df[df["flavor"] == flavor].copy()
    pivot = sub.pivot_table(
        index="E_center_GeV",
        columns="z_center_m",
        values="weight",
        aggfunc="mean",
        fill_value=0.0,
    )
    return (
        pivot.columns.to_numpy(dtype=float),
        pivot.index.to_numpy(dtype=float),
        pivot.to_numpy(dtype=float),
    )


def normalized_z_shape(df: pd.DataFrame, flavor: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df[df["flavor"] == flavor].copy()
    grouped = sub.groupby("z_center_m", sort=True)["weight"].sum().reset_index()
    z = grouped["z_center_m"].to_numpy(dtype=float)
    w = grouped["weight"].to_numpy(dtype=float)
    total = float(np.sum(w))
    if total > 0.0:
        w = w / total
    return z, w


def draw_heatmap(ax, df: pd.DataFrame, flavor: str, title: str) -> None:
    z, e, values = heatmap_table(df, flavor)
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=(z.min(), z.max(), e.min(), e.max()),
        cmap="magma",
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Position de désintégration $z$ [m]")
    ax.set_ylabel(r"Énergie du neutrino $E_\nu$ [GeV]")
    ax.tick_params(direction="in", top=True, right=True)
    plt.colorbar(image, ax=ax, pad=0.01, label=r"Poids normalisé $p(z|E,\alpha)$ par bin en $z$")


def draw_integrated_shapes(ax, df: pd.DataFrame, title: str) -> None:
    for flavor in ["numu", "nue", "numubar", "nuebar"]:
        z, w = normalized_z_shape(df, flavor)
        if len(z) == 0:
            continue
        ax.step(
            z,
            w,
            where="mid",
            color=FLAVOR_COLORS[flavor],
            linewidth=1.4,
            label=FLAVOR_LABELS[flavor],
        )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Position de désintégration $z$ [m]")
    ax.set_ylabel("Poids normalisé intégré")
    ax.tick_params(direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")


def main() -> None:
    fhc = load_profile(FHC_CSV)
    rhc = load_profile(RHC_CSV)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
    draw_heatmap(axes[0, 0], fhc, "numu", r"FHC ND : poids dk2nu $p(z|E,\nu_\mu)$")
    draw_heatmap(axes[0, 1], rhc, "numubar", r"RHC ND : poids dk2nu $p(z|E,\bar{\nu}_\mu)$")
    draw_integrated_shapes(axes[1, 0], fhc, "FHC : profils en z intégrés par saveur")
    draw_integrated_shapes(axes[1, 1], rhc, "RHC : profils en z intégrés par saveur")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {OUT}")


if __name__ == "__main__":
    main()
