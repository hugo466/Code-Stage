from pathlib import Path
import os
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GLOBES_DIR = ROOT / "data" / "dune" / "2103.04797v2" / "dune_globes"
EFF_DIR = GLOBES_DIR / "eff"
OUT = ROOT / "figures" / "dune_fd" / "detector_response" / "fd_efficiencies.png"


def parse_globes_vector(path: Path, name: str) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"\${re.escape(name)}\s*=\s*\{{([^{{}}]+)\}}", text, flags=re.S)
    if not match:
        raise ValueError(f"Vector ${name} not found in {path}")
    return np.array([float(x.strip()) for x in match.group(1).split(",") if x.strip()], dtype=float)


def read_efficiency(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\{([^{}]+)\}", text, flags=re.S)
    if not match:
        raise ValueError(f"No efficiency vector found in {path}")
    return np.array([float(x.strip()) for x in match.group(1).split(",") if x.strip()], dtype=float)


def energy_centers() -> np.ndarray:
    widths = parse_globes_vector(GLOBES_DIR / "DUNE_GLoBES.glb", "binsize")
    edges = np.concatenate(([0.0], np.cumsum(widths)))
    return 0.5 * (edges[:-1] + edges[1:])


PANELS = [
    (
        "FHC apparition",
        [
            ("post_app_FHC_nue_sig.txt", r"signal $\nu_e$", "black", "-", 2.1),
            ("post_app_FHC_nuebar_sig.txt", r"signal $\bar\nu_e$", "0.35", "-", 1.7),
            ("post_app_FHC_nue_bkg.txt", r"beam $\nu_e$", "#1f77b4", "--", 1.5),
            ("post_app_FHC_nuebar_bkg.txt", r"beam $\bar\nu_e$", "#17becf", "--", 1.5),
            ("post_app_FHC_numu_bkg.txt", r"mis-ID $\nu_\mu$", "#2ca02c", ":", 1.7),
            ("post_app_FHC_numubar_bkg.txt", r"mis-ID $\bar\nu_\mu$", "#98df8a", ":", 1.7),
            ("post_app_FHC_NC_bkg.txt", r"NC $\nu$", "#d62728", "-.", 1.5),
            ("post_app_FHC_aNC_bkg.txt", r"NC $\bar\nu$", "#ff9896", "-.", 1.5),
        ],
    ),
    (
        "RHC apparition",
        [
            ("post_app_RHC_nuebar_sig.txt", r"signal $\bar\nu_e$", "black", "-", 2.1),
            ("post_app_RHC_nue_sig.txt", r"signal $\nu_e$", "0.35", "-", 1.7),
            ("post_app_RHC_nuebar_bkg.txt", r"beam $\bar\nu_e$", "#1f77b4", "--", 1.5),
            ("post_app_RHC_nue_bkg.txt", r"beam $\nu_e$", "#17becf", "--", 1.5),
            ("post_app_RHC_numubar_bkg.txt", r"mis-ID $\bar\nu_\mu$", "#2ca02c", ":", 1.7),
            ("post_app_RHC_numu_bkg.txt", r"mis-ID $\nu_\mu$", "#98df8a", ":", 1.7),
            ("post_app_RHC_aNC_bkg.txt", r"NC $\bar\nu$", "#d62728", "-.", 1.5),
            ("post_app_RHC_NC_bkg.txt", r"NC $\nu$", "#ff9896", "-.", 1.5),
        ],
    ),
    (
        "FHC disparition",
        [
            ("post_dis_FHC_numu_sig.txt", r"signal $\nu_\mu$", "black", "-", 2.1),
            ("post_dis_FHC_numubar_sig.txt", r"wrong-sign $\bar\nu_\mu$", "#2ca02c", "--", 1.7),
            ("post_dis_FHC_nutau_bkg.txt", r"$\nu_\tau$", "#ff7f0e", ":", 1.7),
            ("post_dis_FHC_nutaubar_bkg.txt", r"$\bar\nu_\tau$", "#f6b26b", ":", 1.7),
            ("post_dis_FHC_NC_bkg.txt", r"NC $\nu$", "#d62728", "-.", 1.5),
            ("post_dis_FHC_aNC_bkg.txt", r"NC $\bar\nu$", "#ff9896", "-.", 1.5),
        ],
    ),
    (
        "RHC disparition",
        [
            ("post_dis_RHC_numubar_sig.txt", r"signal $\bar\nu_\mu$", "black", "-", 2.1),
            ("post_dis_RHC_numu_sig.txt", r"wrong-sign $\nu_\mu$", "#2ca02c", "--", 1.7),
            ("post_dis_RHC_nutaubar_bkg.txt", r"$\bar\nu_\tau$", "#ff7f0e", ":", 1.7),
            ("post_dis_RHC_nutau_bkg.txt", r"$\nu_\tau$", "#f6b26b", ":", 1.7),
            ("post_dis_RHC_aNC_bkg.txt", r"NC $\bar\nu$", "#d62728", "-.", 1.5),
            ("post_dis_RHC_NC_bkg.txt", r"NC $\nu$", "#ff9896", "-.", 1.5),
        ],
    ),
]


def main() -> None:
    e = energy_centers()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.ravel()

    summary = []
    for ax, (title, curves) in zip(axes, PANELS):
        for filename, label, color, linestyle, lw in curves:
            path = EFF_DIR / filename
            y = read_efficiency(path)
            if len(y) != len(e):
                raise ValueError(f"{filename}: {len(y)} values, expected {len(e)}")
            ax.step(e, y, where="mid", label=label, color=color, linestyle=linestyle, linewidth=lw)
            summary.append((title, filename, float(np.nanmin(y)), float(np.nanmax(y)), float(np.nanmean(y))))
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(0.5, 8.0)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.8, ncol=2, frameon=False, loc="best")

    for ax in axes[2:]:
        ax.set_xlabel(r"Energie reconstruite $E_{rec}$ [GeV]")
    for ax in axes[::2]:
        ax.set_ylabel("Efficacite FD")

    fig.tight_layout()
    fig.savefig(OUT, dpi=220)
    plt.close(fig)

    print(f"saved {OUT}")
    print("efficiency summary: panel,file,min,max,mean")
    for row in summary:
        print(f"{row[0]},{row[1]},{row[2]:.6g},{row[3]:.6g},{row[4]:.6g}")


if __name__ == "__main__":
    main()
