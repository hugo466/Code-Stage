"""Quantitative consistency check between dk2nu ROOT and source_profile_z CSV.

The source-profile plot uses p(z | E, flavor), then integrates those
conditional profiles over E and normalizes the resulting z-shape.  This
script rebuilds the same object from the ROOT dk2nu ntuples while keeping
the parent meson label, then compares the summed-parent result with the
stored source_profile_z_FHC_ND.csv.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = ROOT / ".matplotlib_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_GLOB = ROOT / "data" / "dune" / "dk2nu" / "raw" / "OptimizedEngineeredNov2017_FHC" / "*.root"
DEFAULT_PROFILE = ROOT / "data" / "dune" / "dk2nu" / "source_profile_z_FHC_ND.csv"
DEFAULT_OUT = ROOT / "figures" / "dune_nd" / "flux" / "dk2nu_source_profile_consistency_fhc_nd.png"
DEFAULT_OUT_CSV = ROOT / "data" / "dune" / "dk2nu" / "source_profile_parent_flavor_FHC_ND_from_root.csv"
DEFAULT_SUMMARY_CSV = ROOT / "data" / "dune" / "dk2nu" / "source_profile_consistency_FHC_ND.csv"

TREE_NAME = "dk2nuTree"
META_TREE_NAME = "dkmetaTree"
ENERGY_BRANCH = "dk2nu/nuray/nuray.E"
RAY_WEIGHT_BRANCH = "dk2nu/nuray/nuray.wgt"
FLAVOR_BRANCH = "dk2nu/decay/decay.ntype"
PARENT_BRANCH = "dk2nu/decay/decay.ptype"
Z_BRANCH = "dk2nu/decay/decay.vz"
IMPORTANCE_BRANCH = "dk2nu/decay/decay.nimpwt"

FLAVOR_LABELS = {
    12: ("nue", r"$\nu_e$", "tab:orange"),
    14: ("numu", r"$\nu_\mu$", "tab:blue"),
    -12: ("nuebar", r"$\bar{\nu}_e$", "tab:red"),
    -14: ("numubar", r"$\bar{\nu}_\mu$", "tab:green"),
}
PARENT_LABELS = {
    211: r"$\pi^+$",
    -211: r"$\pi^-$",
    321: r"$K^+$",
    -321: r"$K^-$",
    130: r"$K^0_L$",
    13: r"$\mu^-$",
    -13: r"$\mu^+$",
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


def flat_numpy(array) -> np.ndarray:
    import awkward as ak

    return np.asarray(ak.to_numpy(ak.flatten(array, axis=None)))


def detector_ray_numpy(array, index: int) -> np.ndarray:
    import awkward as ak

    return np.asarray(ak.to_numpy(array[:, index]), dtype=float)


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
    return df


def edges_from_profile(profile: pd.DataFrame, low_col: str, high_col: str) -> np.ndarray:
    values = np.unique(np.concatenate([profile[low_col].to_numpy(dtype=float), profile[high_col].to_numpy(dtype=float)]))
    values.sort()
    return values


def accumulate_from_root(files: list[Path], profile: pd.DataFrame, nd_index: int | None, max_files: int) -> pd.DataFrame:
    import uproot

    if max_files > 0:
        files = files[:max_files]

    e_edges = edges_from_profile(profile, "E_GeV_bin_low", "E_GeV_bin_high")
    z_edges = edges_from_profile(profile, "z_decay_m_bin_low", "z_decay_m_bin_high")
    parent_values = [211, -211, 321, -321, 130, 13, -13]
    shape = (len(e_edges) - 1, len(z_edges) - 1)
    hist = {
        (pdg, parent): np.zeros(shape, dtype=float)
        for pdg in FLAVOR_LABELS
        for parent in parent_values
    }

    used_files = 0
    skipped_files = 0
    read_events = 0
    kept_events = 0

    for file_number, path in enumerate(files, start=1):
        try:
            with uproot.open(path) as root_file:
                tree = root_file[TREE_NAME]
                det_index = detector_index(root_file, nd_index)
                arrays = tree.arrays(
                    [
                        ENERGY_BRANCH,
                        RAY_WEIGHT_BRANCH,
                        FLAVOR_BRANCH,
                        PARENT_BRANCH,
                        Z_BRANCH,
                        IMPORTANCE_BRANCH,
                    ],
                    library="ak",
                )
                energy = detector_ray_numpy(arrays[ENERGY_BRANCH], det_index)
                ray_weight = detector_ray_numpy(arrays[RAY_WEIGHT_BRANCH], det_index)
                flavor = flat_numpy(arrays[FLAVOR_BRANCH]).astype(int)
                parent = flat_numpy(arrays[PARENT_BRANCH]).astype(int)
                z_m = flat_numpy(arrays[Z_BRANCH]).astype(float) * 0.01
                importance = flat_numpy(arrays[IMPORTANCE_BRANCH]).astype(float)
        except Exception as exc:
            skipped_files += 1
            print(f"[warn] fichier ROOT ignore: {path.name} ({exc})")
            continue

        n = min(len(energy), len(ray_weight), len(flavor), len(parent), len(z_m), len(importance))
        energy = energy[:n]
        ray_weight = ray_weight[:n]
        flavor = flavor[:n]
        parent = parent[:n]
        z_m = z_m[:n]
        importance = importance[:n]
        read_events += n

        weight = ray_weight * importance
        valid = (
            np.isfinite(energy)
            & np.isfinite(z_m)
            & np.isfinite(weight)
            & (weight > 0.0)
            & (energy >= e_edges[0])
            & (energy < e_edges[-1])
            & (z_m >= z_edges[0])
            & (z_m < z_edges[-1])
        )
        kept_events += int(np.count_nonzero(valid))
        for pdg in FLAVOR_LABELS:
            flavor_mask = valid & (flavor == pdg)
            if not np.any(flavor_mask):
                continue
            for parent_pdg in parent_values:
                mask = flavor_mask & (parent == parent_pdg)
                if np.any(mask):
                    hist[(pdg, parent_pdg)] += np.histogram2d(
                        energy[mask],
                        z_m[mask],
                        bins=[e_edges, z_edges],
                        weights=weight[mask],
                    )[0]
        used_files += 1
        if file_number == 1 or file_number % 10 == 0 or file_number == len(files):
            print(f"Fichiers traites: {file_number}/{len(files)}")

    rows = []
    for pdg, (flavor_label, _, _) in FLAVOR_LABELS.items():
        for parent_pdg in parent_values:
            parent_label = PARENT_LABELS.get(parent_pdg, str(parent_pdg)).replace("$", "")
            h = hist[(pdg, parent_pdg)]
            for ie in range(len(e_edges) - 1):
                for iz in range(len(z_edges) - 1):
                    rows.append(
                        {
                            "flavor": flavor_label,
                            "parent_pdg": parent_pdg,
                            "parent_label": parent_label,
                            "E_GeV_bin_low": e_edges[ie],
                            "E_GeV_bin_high": e_edges[ie + 1],
                            "z_decay_m_bin_low": z_edges[iz],
                            "z_decay_m_bin_high": z_edges[iz + 1],
                            "raw_weight": h[ie, iz],
                        }
                    )
    print(f"Fichiers utilises: {used_files}")
    if skipped_files:
        print(f"Fichiers ignores: {skipped_files}")
    print(f"Evenements lus: {read_events}")
    print(f"Evenements gardes: {kept_events}")
    return pd.DataFrame(rows)


def integrated_normalized_profile(profile: pd.DataFrame, weight_col: str, *, normalize_per_energy: bool) -> pd.DataFrame:
    df = profile.copy()
    if normalize_per_energy:
        grouped = df.groupby(["flavor", "E_GeV_bin_low", "E_GeV_bin_high"], sort=False)[weight_col].transform("sum")
        df["conditional_weight"] = np.divide(
            df[weight_col].to_numpy(dtype=float),
            grouped.to_numpy(dtype=float),
            out=np.zeros(len(df), dtype=float),
            where=grouped.to_numpy(dtype=float) > 0.0,
        )
    else:
        df["conditional_weight"] = df[weight_col].to_numpy(dtype=float)
    out = (
        df.groupby(["flavor", "z_decay_m_bin_low", "z_decay_m_bin_high"], sort=True)["conditional_weight"]
        .sum()
        .reset_index()
    )
    norm = out.groupby("flavor", sort=False)["conditional_weight"].transform("sum")
    out["shape"] = np.divide(
        out["conditional_weight"].to_numpy(dtype=float),
        norm.to_numpy(dtype=float),
        out=np.zeros(len(out), dtype=float),
        where=norm.to_numpy(dtype=float) > 0.0,
    )
    out["z_center_m"] = 0.5 * (out["z_decay_m_bin_low"] + out["z_decay_m_bin_high"])
    return out


def make_consistency_table(source_profile: pd.DataFrame, root_detail: pd.DataFrame) -> pd.DataFrame:
    source_shape = integrated_normalized_profile(source_profile, "weight", normalize_per_energy=False)
    root_by_flavor = (
        root_detail.groupby(["flavor", "E_GeV_bin_low", "E_GeV_bin_high", "z_decay_m_bin_low", "z_decay_m_bin_high"], sort=False)[
            "raw_weight"
        ]
        .sum()
        .reset_index()
    )
    root_shape = integrated_normalized_profile(root_by_flavor, "raw_weight", normalize_per_energy=True)
    merged = source_shape.rename(columns={"shape": "source_profile_shape"})[
        ["flavor", "z_decay_m_bin_low", "z_decay_m_bin_high", "z_center_m", "source_profile_shape"]
    ].merge(
        root_shape.rename(columns={"shape": "root_rebuilt_shape"})[
            ["flavor", "z_decay_m_bin_low", "z_decay_m_bin_high", "root_rebuilt_shape"]
        ],
        on=["flavor", "z_decay_m_bin_low", "z_decay_m_bin_high"],
        how="outer",
    )
    merged = merged.fillna(0.0)
    merged["delta_shape"] = merged["root_rebuilt_shape"] - merged["source_profile_shape"]
    merged["relative_delta"] = np.divide(
        merged["delta_shape"].to_numpy(dtype=float),
        merged["source_profile_shape"].to_numpy(dtype=float),
        out=np.full(len(merged), np.nan, dtype=float),
        where=merged["source_profile_shape"].to_numpy(dtype=float) > 0.0,
    )
    return merged


def plot_consistency(summary: pd.DataFrame, out_path: Path) -> None:
    flavors = ["numu", "nue", "numubar", "nuebar"]
    fig, axes = plt.subplots(len(flavors), 2, figsize=(12.0, 10.5), sharex=True)
    for row, flavor in enumerate(flavors):
        label = next(v[1] for v in FLAVOR_LABELS.values() if v[0] == flavor)
        color = next(v[2] for v in FLAVOR_LABELS.values() if v[0] == flavor)
        sub = summary[summary["flavor"] == flavor].sort_values("z_center_m")
        ax, rax = axes[row]
        ax.step(
            sub["z_center_m"],
            sub["source_profile_shape"],
            where="mid",
            color="black",
            linewidth=1.25,
            label="source_profile CSV",
        )
        ax.step(
            sub["z_center_m"],
            sub["root_rebuilt_shape"],
            where="mid",
            color=color,
            linewidth=1.0,
            linestyle="--",
            label="ROOT rebuilt, summed parents",
        )
        rax.axhline(0.0, color="black", linewidth=0.8)
        rax.step(
            sub["z_center_m"],
            100.0 * sub["relative_delta"],
            where="mid",
            color=color,
            linewidth=1.0,
        )
        ax.set_ylabel(label)
        rax.set_ylabel(r"$(ROOT-CSV)/CSV$ [%]")
        ax.grid(alpha=0.25)
        rax.grid(alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True)
        rax.tick_params(direction="in", top=True, right=True)
        if row == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper right")
            ax.set_title("Normalized integrated z-shape")
            rax.set_title(r"$(ROOT-CSV)/CSV$")
    for ax in axes[-1]:
        ax.set_xlabel("Decay position z (m)")
    fig.suptitle("Strict dk2nu source-profile consistency: ROOT parent/flavor/z vs stored source_profile_z", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary: pd.DataFrame) -> None:
    print("Max absolute differences by flavor:")
    for flavor in ["numu", "nue", "numubar", "nuebar"]:
        sub = summary[summary["flavor"] == flavor]
        abs_delta = float(np.nanmax(np.abs(sub["delta_shape"])))
        valid_rel = sub["relative_delta"].replace([np.inf, -np.inf], np.nan).dropna()
        abs_rel = float(np.nanmax(np.abs(valid_rel))) if len(valid_rel) else np.nan
        rms_delta = float(np.sqrt(np.nanmean(sub["delta_shape"].to_numpy(dtype=float) ** 2)))
        print(f"  {flavor:7s}: max |Delta shape|={abs_delta:.6g}, RMS={rms_delta:.6g}, max |rel|={100.0 * abs_rel:.6g}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict source_profile_z consistency check against dk2nu ROOT.")
    parser.add_argument("--input-glob", type=str, default=str(DEFAULT_INPUT_GLOB))
    parser.add_argument("--source-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--nd-index", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(Path(path) for path in glob.glob(args.input_glob))
    if not files:
        raise RuntimeError(f"Aucun fichier ROOT trouve avec: {args.input_glob}")
    source_profile = load_profile(args.source_profile)
    root_detail = accumulate_from_root(files, source_profile, args.nd_index, args.max_files)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    root_detail.to_csv(args.out_csv, index=False)
    summary = make_consistency_table(source_profile, root_detail)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    plot_consistency(summary, args.out)
    print_summary(summary)
    print(f"Detail CSV: {args.out_csv.resolve()}")
    print(f"Summary CSV: {args.summary_csv.resolve()}")
    print(f"Figure: {args.out.resolve()}")


if __name__ == "__main__":
    main()
