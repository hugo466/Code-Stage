"""Reproduce the dk2nu meson decay/origin figure for the DUNE ND.

This is intentionally different from plot_dk2nu_source_weights.py:
that script plots normalized source profiles p(z|E, flavor), while this
one keeps the decay-parent information from the dk2nu ROOT files and
plots arbitrary-unit distributions by parent meson.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = ROOT / ".matplotlib_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_GLOB = ROOT / "data" / "dune" / "dk2nu" / "raw" / "OptimizedEngineeredNov2017_FHC" / "*.root"
DEFAULT_OUT = ROOT / "figures" / "dune_nd" / "flux" / "dk2nu_meson_decay_origin_nd_article.png"
DEFAULT_OUT_CSV = ROOT / "data" / "dune" / "dk2nu" / "meson_decay_origin_FHC_ND.csv"

TREE_NAME = "dk2nuTree"
META_TREE_NAME = "dkmetaTree"
PARENT_BRANCH = "dk2nu/decay/decay.ptype"
Z_BRANCH = "dk2nu/decay/decay.vz"
IMPORTANCE_BRANCH = "dk2nu/decay/decay.nimpwt"
RAY_WEIGHT_BRANCH = "dk2nu/nuray/nuray.wgt"

PARENTS = {
    211: (r"$\pi^+$", "#9f1d16"),
    -211: (r"$\pi^-$", "#1b16b7"),
    321: (r"$K^+$", "#42a635"),
    -321: (r"$K^-$", "#7a7a7a"),
}


def detector_index(root_file, requested: int | None) -> int:
    if requested is not None:
        return requested
    try:
        import awkward as ak

        meta = root_file[META_TREE_NAME]
        arrays = meta.arrays(
            ["dkmeta/location/location.name", "dkmeta/location/location.z"],
            entry_stop=1,
            library="ak",
        )
        names = [str(item).lower() for item in ak.to_list(arrays["dkmeta/location/location.name"])[0]]
        for index, name in enumerate(names):
            if "near" in name:
                return index
    except Exception:
        pass
    return 1


def detector_ray_weight(array, index: int) -> np.ndarray:
    import awkward as ak

    return np.asarray(ak.to_numpy(array[:, index]), dtype=float)


def flat_numpy(array) -> np.ndarray:
    import awkward as ak

    return np.asarray(ak.to_numpy(ak.flatten(array, axis=None)))


def accumulate(files: list[Path], z_edges: np.ndarray, nd_index: int | None, max_files: int) -> pd.DataFrame:
    import uproot

    if max_files > 0:
        files = files[:max_files]
    decay_hist = {pdg: np.zeros(len(z_edges) - 1, dtype=float) for pdg in PARENTS}
    nd_origin_hist = {pdg: np.zeros(len(z_edges) - 1, dtype=float) for pdg in PARENTS}
    used_files = 0
    skipped_files = 0
    read_events = 0

    for file_number, path in enumerate(files, start=1):
        try:
            with uproot.open(path) as root_file:
                tree = root_file[TREE_NAME]
                det_index = detector_index(root_file, nd_index)
                arrays = tree.arrays(
                    [PARENT_BRANCH, Z_BRANCH, IMPORTANCE_BRANCH, RAY_WEIGHT_BRANCH],
                    library="ak",
                )
                parent = flat_numpy(arrays[PARENT_BRANCH]).astype(int)
                z_m = flat_numpy(arrays[Z_BRANCH]).astype(float) * 0.01
                importance = flat_numpy(arrays[IMPORTANCE_BRANCH]).astype(float)
                ray_weight = detector_ray_weight(arrays[RAY_WEIGHT_BRANCH], det_index)
        except Exception as exc:
            skipped_files += 1
            print(f"[warn] fichier ROOT ignore: {path.name} ({exc})")
            continue

        n = min(len(parent), len(z_m), len(importance), len(ray_weight))
        parent = parent[:n]
        z_m = z_m[:n]
        importance = importance[:n]
        ray_weight = ray_weight[:n]
        read_events += n

        valid = (
            np.isfinite(z_m)
            & np.isfinite(importance)
            & np.isfinite(ray_weight)
            & (z_m >= z_edges[0])
            & (z_m < z_edges[-1])
            & (importance > 0.0)
            & (ray_weight > 0.0)
        )
        for pdg in PARENTS:
            mask = valid & (parent == pdg)
            if not np.any(mask):
                continue
            decay_hist[pdg] += np.histogram(z_m[mask], bins=z_edges, weights=importance[mask])[0]
            nd_origin_hist[pdg] += np.histogram(z_m[mask], bins=z_edges, weights=importance[mask] * ray_weight[mask])[0]
        used_files += 1
        if file_number == 1 or file_number % 10 == 0 or file_number == len(files):
            print(f"Fichiers traites: {file_number}/{len(files)}")

    rows = []
    for pdg, (label, _) in PARENTS.items():
        for i in range(len(z_edges) - 1):
            rows.append(
                {
                    "parent_pdg": pdg,
                    "parent_label": label.replace("$", ""),
                    "z_low_m": z_edges[i],
                    "z_high_m": z_edges[i + 1],
                    "z_center_m": 0.5 * (z_edges[i] + z_edges[i + 1]),
                    "meson_decay_weight": decay_hist[pdg][i],
                    "nd_flux_origin_weight": nd_origin_hist[pdg][i],
                }
            )
    out = pd.DataFrame(rows)
    print(f"Fichiers utilises: {used_files}")
    if skipped_files:
        print(f"Fichiers ignores: {skipped_files}")
    print(f"Evenements lus: {read_events}")
    return out


def normalize_article_units(df: pd.DataFrame, left_target_max: float, right_target_max: float) -> pd.DataFrame:
    out = df.copy()
    left_max = float(out["meson_decay_weight"].max())
    right_pi_plus = out[out["parent_pdg"] == 211]["nd_flux_origin_weight"]
    right_max = float(right_pi_plus.max()) if len(right_pi_plus) else float(out["nd_flux_origin_weight"].max())
    out["meson_decay_article_units"] = out["meson_decay_weight"] * (left_target_max / left_max if left_max > 0.0 else 1.0)
    out["nd_flux_origin_article_units"] = out["nd_flux_origin_weight"] * (right_target_max / right_max if right_max > 0.0 else 1.0)
    return out


def plot_article(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.55), sharex=True)
    left, right = axes

    for pdg, (label, color) in PARENTS.items():
        sub = df[df["parent_pdg"] == pdg].sort_values("z_center_m")
        left.step(
            sub["z_center_m"],
            sub["meson_decay_article_units"],
            where="mid",
            color=color,
            linewidth=1.25,
            label=label,
        )
        right.step(
            sub["z_center_m"],
            sub["nd_flux_origin_article_units"],
            where="mid",
            color=color,
            linewidth=1.25,
            label=label,
        )

    left.set_title("Meson decay distribution", fontsize=14)
    right.set_title("ND (anti)neutrino flux origin", fontsize=14)
    left.set_yscale("log")
    left.set_ylim(10.0, 3.0e5)
    right.set_ylim(0.0, 6.05)
    for ax in axes:
        ax.set_xlim(-20.0, 250.0)
        ax.set_xlabel(r"$z$ [m]", fontsize=12)
        ax.set_ylabel("Arbitrary units", fontsize=12)
        ax.tick_params(direction="in", top=True, right=True, which="both")
        ax.minorticks_on()
    right.legend(frameon=False, bbox_to_anchor=(1.05, 0.5), loc="center left", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the DUNE dk2nu meson decay/origin ND figure.")
    parser.add_argument("--input-glob", type=str, default=str(DEFAULT_INPUT_GLOB))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--nd-index", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--z-min-m", type=float, default=-20.0)
    parser.add_argument("--z-max-m", type=float, default=240.0)
    parser.add_argument("--z-bins", type=int, default=520)
    parser.add_argument("--left-target-max", type=float, default=1.0e5)
    parser.add_argument("--right-target-max", type=float, default=5.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(Path(p) for p in glob.glob(args.input_glob))
    if not files:
        raise RuntimeError(f"Aucun ROOT dk2nu trouve avec: {args.input_glob}")
    z_edges = np.linspace(args.z_min_m, args.z_max_m, args.z_bins + 1)
    table = accumulate(files, z_edges, args.nd_index, args.max_files)
    table = normalize_article_units(table, args.left_target_max, args.right_target_max)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    plot_article(table, args.out)
    print(f"CSV sauvegarde: {args.out_csv.resolve()}")
    print(f"Figure sauvegardee: {args.out.resolve()}")


if __name__ == "__main__":
    main()
