#!/usr/bin/env python
"""Convert dk2nu ROOT ntuples into flavor/E/z differential flux tables.

Output CSV columns:
flavor,E_GeV_bin_low,E_GeV_bin_high,z_decay_m_bin_low,z_decay_m_bin_high,weight
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np


PUBLIC_DK2NU_TARBALLS = {
    "OptimizedEngineeredNov2017_FHC": "https://www.dropbox.com/s/hyhjq90psosyuhc/OptimizedEngineeredNov2017_Neutrino.tar.gz?dl=1",
    "OptimizedEngineeredNov2017_RHC": "https://www.dropbox.com/s/tyqub9jrcq4b4x2/OptimizedEngineeredNov2017_Antineutrino.tar.gz?dl=1",
}

FLAVOR_LABELS = {
    12: "nue",
    14: "numu",
    16: "nutau",
    -12: "nuebar",
    -14: "numubar",
    -16: "nutaubar",
}
GLOBES_COLUMN = {
    12: 1,
    14: 2,
    16: 3,
    -12: 4,
    -14: 5,
    -16: 6,
}

ENERGY_CANDIDATES = [
    "dk2nu/nuray/nuray.E",
    "dk2nu.nu.E",
    "nu.E",
    "dk2nu.nuray.E",
    "nuray.E",
    "E",
]
WEIGHT_CANDIDATES = [
    "dk2nu/nuray/nuray.wgt",
    "dk2nu.nu.wgt",
    "nu.wgt",
    "dk2nu.nu.wgt_xy",
    "nu.wgt_xy",
    "wgt",
    "weight",
]
FLAVOR_CANDIDATES = [
    "dk2nu/decay/decay.ntype",
    "dk2nu.decay.ntype",
    "decay.ntype",
    "dk2nu.neu",
    "neu",
    "ntype",
]
Z_CANDIDATES = [
    "dk2nu/decay/decay.vz",
    "dk2nu.decay.vz",
    "decay.vz",
    "dk2nu.vz",
    "vz",
    "dk2nu.ppvz",
    "ppvz",
]
IMPORTANCE_CANDIDATES = [
    "dk2nu/decay/decay.nimpwt",
    "dk2nu.decay.nimpwt",
    "decay.nimpwt",
    "nimpwt",
]
POT_CANDIDATES = [
    "dkmeta/pots",
    "pots",
]


def resolve_inputs(paths: list[str], input_glob: str | None) -> list[Path]:
    out: list[Path] = []
    for item in paths:
        out.extend(Path(p) for p in glob.glob(item))
    if input_glob:
        out.extend(Path(p) for p in glob.glob(input_glob))
    return sorted({p.resolve() for p in out if p.exists()})


def first_existing(branches: set[str], candidates: list[str], explicit: str | None, label: str) -> str:
    if explicit:
        if explicit not in branches:
            raise KeyError(f"Branche {label} introuvable: {explicit}")
        return explicit
    for candidate in candidates:
        if candidate in branches:
            return candidate
    preview = ", ".join(sorted(list(branches))[:40])
    raise KeyError(f"Impossible d'auto-detecter la branche {label}. Branches disponibles: {preview}")


def find_tree(root_file, tree_name: str | None):
    if tree_name:
        return root_file[tree_name]
    for key in root_file.keys():
        obj = root_file[key]
        if hasattr(obj, "arrays"):
            return obj
    raise KeyError("Aucun TTree trouve dans le fichier ROOT.")


def to_flat_numpy(array, det_index: int | None = None) -> np.ndarray:
    try:
        import awkward as ak

        arr = array
        if det_index is not None and getattr(arr, "ndim", 1) > 1:
            arr = arr[:, det_index]
        return np.asarray(ak.to_numpy(ak.flatten(arr, axis=None)))
    except Exception:
        arr = np.asarray(array)
        if det_index is not None and arr.ndim > 1:
            arr = arr[:, det_index]
        return arr.reshape(-1)


def detector_locations(root_file) -> list[tuple[str, float]]:
    try:
        import awkward as ak

        tree = root_file["dkmetaTree"]
        arrays = tree.arrays(
            ["dkmeta/location/location.name", "dkmeta/location/location.z"],
            entry_stop=1,
            library="ak",
        )
        names = ak.to_list(arrays["dkmeta/location/location.name"])[0]
        positions_cm = ak.to_list(arrays["dkmeta/location/location.z"])[0]
        return [(str(name), float(z_cm) * 0.01) for name, z_cm in zip(names, positions_cm)]
    except Exception as exc:
        raise RuntimeError(f"Impossible de lire les positions detecteur dans dkmetaTree: {exc}") from exc


def resolve_detector_indices(root_file, nd_index: int | None, fd_index: int | None):
    locations = detector_locations(root_file)

    def find_index(needles):
        for index, (name, _) in enumerate(locations):
            lower = name.lower()
            if any(needle in lower for needle in needles):
                return index
        return None

    resolved_nd = nd_index if nd_index is not None else find_index(("neardet", "near det", "near"))
    resolved_fd = fd_index if fd_index is not None else find_index(("fardet", "far det", "far"))
    if resolved_nd is None or resolved_fd is None:
        raise RuntimeError(f"Indices ND/FD introuvables dans les locations dk2nu: {locations}")
    if resolved_nd == resolved_fd:
        raise RuntimeError("Les indices ND et FD doivent etre differents.")
    if resolved_nd >= len(locations) or resolved_fd >= len(locations):
        raise RuntimeError(
            f"Indice detecteur hors plage: ND={resolved_nd}, FD={resolved_fd}, locations={locations}"
        )
    return resolved_nd, resolved_fd, locations


def detector_ray_numpy(array, detector_index: int) -> np.ndarray:
    try:
        import awkward as ak

        return np.asarray(ak.to_numpy(array[:, detector_index]), dtype=float)
    except Exception as exc:
        raise RuntimeError(f"Impossible d'extraire le rayon detecteur d'indice {detector_index}: {exc}") from exc


def file_pot(root_file) -> float:
    tree = root_file["dkmetaTree"]
    branches = set(tree.keys())
    branch = first_existing(branches, POT_CANDIDATES, None, "POT")
    values = tree[branch].array(library="np")
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError(f"POT invalide dans dkmetaTree: {total}")
    return total


def accumulate_detector_histogram(args, files: list[Path]):
    """Accumulate one detector's raw dk2nu weights without GLoBES normalization."""
    try:
        import uproot
    except ImportError as exc:
        raise SystemExit("Module Python manquant: uproot. Installe-le avec `pip install uproot awkward`.") from exc

    e_edges = np.linspace(args.e_min, args.e_max, args.e_bins + 1)
    z_edges = np.linspace(args.z_min, args.z_max, args.z_bins + 1)
    histograms = {
        pdg: np.zeros((args.e_bins, args.z_bins), dtype=float)
        for pdg in FLAVOR_LABELS
    }
    total_pot = 0.0
    total_events = 0
    kept_events = 0
    skipped = 0
    resolved_detector_index = None
    resolved_locations = None

    for file_number, path in enumerate(files, start=1):
        try:
            with uproot.open(path) as root_file:
                tree = find_tree(root_file, args.tree)
                branches = set(tree.keys())
                e_branch = first_existing(branches, ENERGY_CANDIDATES, args.energy_branch, "energie")
                w_branch = first_existing(branches, WEIGHT_CANDIDATES, args.weight_branch, "poids")
                f_branch = first_existing(branches, FLAVOR_CANDIDATES, args.flavor_branch, "saveur")
                z_branch = first_existing(branches, Z_CANDIDATES, args.z_branch, "z decay")
                importance_branch = first_existing(
                    branches,
                    IMPORTANCE_CANDIDATES,
                    args.importance_branch,
                    "poids d'importance",
                )
                current_pot = file_pot(root_file)
                if args.det_index is None:
                    nd_index, fd_index, locations = resolve_detector_indices(
                        root_file,
                        args.nd_index,
                        args.fd_index,
                    )
                    detector_index = nd_index if args.det == "ND" else fd_index
                    if resolved_locations is None:
                        resolved_locations = locations
                        print(f"Locations dk2nu: {locations}")
                        print(f"Rayon utilise: {args.det} index={detector_index}")
                else:
                    detector_index = args.det_index
                if resolved_detector_index is None:
                    resolved_detector_index = detector_index
                elif resolved_detector_index != detector_index:
                    raise RuntimeError(
                        f"Indice detecteur variable entre fichiers: "
                        f"{resolved_detector_index} puis {detector_index}"
                    )

                arrays = tree.arrays(
                    [e_branch, w_branch, f_branch, z_branch, importance_branch],
                    library="ak",
                )
                energy = detector_ray_numpy(arrays[e_branch], detector_index)
                ray_weight = detector_ray_numpy(arrays[w_branch], detector_index)
                flavor = to_flat_numpy(arrays[f_branch], None).astype(int)
                z_raw = to_flat_numpy(arrays[z_branch], None)
                importance = to_flat_numpy(arrays[importance_branch], None)
        except Exception as exc:
            skipped += 1
            print(f"[warn] fichier ROOT ignore: {path} ({exc})", file=sys.stderr)
            continue

        n = min(
            len(energy),
            len(ray_weight),
            len(flavor),
            len(z_raw),
            len(importance),
        )
        total_pot += current_pot
        total_events += n
        if n == 0:
            continue

        energy = energy[:n]
        flavor = flavor[:n]
        z_m = z_raw[:n] * (0.01 if args.z_unit == "cm" else 1.0) - args.z_offset_m
        weight = ray_weight[:n] * importance[:n]
        valid = (
            np.isfinite(energy)
            & np.isfinite(weight)
            & np.isfinite(z_m)
            & (weight > 0.0)
            & (energy >= args.e_min)
            & (energy < args.e_max)
            & (z_m >= args.z_min)
            & (z_m < args.z_max)
        )
        kept_events += int(np.count_nonzero(valid))
        for pdg in FLAVOR_LABELS:
            mask = valid & (flavor == pdg)
            if np.any(mask):
                histograms[pdg] += np.histogram2d(
                    energy[mask],
                    z_m[mask],
                    bins=[e_edges, z_edges],
                    weights=weight[mask],
                )[0]

        if file_number == 1 or file_number % 20 == 0 or file_number == len(files):
            print(f"Fichiers traites: {file_number}/{len(files)}")

    if total_pot <= 0.0 or total_events == 0:
        raise RuntimeError("Aucun evenement dk2nu exploitable pour le flux absolu.")
    if skipped:
        print(f"[warn] fichiers ROOT ignores: {skipped}/{len(files)}", file=sys.stderr)
    print(f"Evenements lus: {total_events}")
    print(f"Evenements gardes: {kept_events}")
    print(f"POT total utilise: {total_pot:.12g}")
    return histograms, e_edges, z_edges, total_pot


def load_arrays(args, files: list[Path]):
    try:
        import uproot
    except ImportError as exc:
        raise SystemExit("Module Python manquant: uproot. Installe-le avec `pip install uproot awkward`.") from exc

    all_energy: list[np.ndarray] = []
    all_weight: list[np.ndarray] = []
    all_flavor: list[np.ndarray] = []
    all_z: list[np.ndarray] = []
    branches_seen: set[str] | None = None

    skipped = 0
    for path in files:
        try:
            with uproot.open(path) as root_file:
                tree = find_tree(root_file, args.tree)
                branches = set(tree.keys())
                branches_seen = branches
                e_branch = first_existing(branches, ENERGY_CANDIDATES, args.energy_branch, "energie")
                w_branch = first_existing(branches, WEIGHT_CANDIDATES, args.weight_branch, "poids")
                f_branch = first_existing(branches, FLAVOR_CANDIDATES, args.flavor_branch, "saveur")
                z_branch = first_existing(branches, Z_CANDIDATES, args.z_branch, "z decay")
                importance_branch = first_existing(
                    branches,
                    IMPORTANCE_CANDIDATES,
                    args.importance_branch,
                    "poids d'importance",
                )
                if args.det_index is None:
                    nd_index, fd_index, locations = resolve_detector_indices(
                        root_file,
                        args.nd_index,
                        args.fd_index,
                    )
                    detector_index = nd_index if args.det == "ND" else fd_index
                else:
                    detector_index = args.det_index

                arrays = tree.arrays(
                    [e_branch, w_branch, f_branch, z_branch, importance_branch],
                    library="ak",
                )
                energy = to_flat_numpy(arrays[e_branch], detector_index)
                weight = (
                    to_flat_numpy(arrays[w_branch], detector_index)
                    * to_flat_numpy(arrays[importance_branch], None)
                )
                flavor = to_flat_numpy(arrays[f_branch], None).astype(int)
                z_raw = to_flat_numpy(arrays[z_branch], None)
        except Exception as exc:
            skipped += 1
            print(f"[warn] fichier ROOT ignore: {path} ({exc})", file=sys.stderr)
            continue

        n = min(len(energy), len(weight), len(flavor), len(z_raw))
        if n == 0:
            continue
        z_m = z_raw[:n] * (0.01 if args.z_unit == "cm" else 1.0) - args.z_offset_m
        all_energy.append(energy[:n])
        all_weight.append(weight[:n])
        all_flavor.append(flavor[:n])
        all_z.append(z_m)

    if not all_energy:
        preview = "" if branches_seen is None else f" Dernieres branches vues: {sorted(list(branches_seen))[:40]}"
        raise RuntimeError("Aucun evenement dk2nu exploitable lu." + preview)

    if skipped:
        print(f"[warn] fichiers ROOT ignores: {skipped}/{len(files)}", file=sys.stderr)

    return (
        np.concatenate(all_energy),
        np.concatenate(all_weight),
        np.concatenate(all_flavor),
        np.concatenate(all_z),
    )


def accumulate_paired_histograms(args, files: list[Path]):
    try:
        import uproot
    except ImportError as exc:
        raise SystemExit("Module Python manquant: uproot. Installe-le avec `pip install uproot awkward`.") from exc

    e_edges = np.linspace(args.e_min, args.e_max, args.e_bins + 1)
    z_edges = np.linspace(args.z_min, args.z_max, args.z_bins + 1)
    histograms = {
        pdg: {
            "nd_ez": np.zeros((args.e_bins, args.z_bins), dtype=float),
            "fd_ez": np.zeros((args.e_bins, args.z_bins), dtype=float),
            "joint_fd": np.zeros((args.e_bins, args.e_bins), dtype=float),
            "joint_count": np.zeros((args.e_bins, args.e_bins), dtype=np.int64),
            "nd_e": np.zeros(args.e_bins, dtype=float),
            "fd_e": np.zeros(args.e_bins, dtype=float),
        }
        for pdg in FLAVOR_LABELS
    }

    total_events = 0
    kept_events = 0
    skipped = 0
    resolved_indices = None
    resolved_locations = None

    for file_number, path in enumerate(files, start=1):
        try:
            with uproot.open(path) as root_file:
                tree = find_tree(root_file, args.tree)
                branches = set(tree.keys())
                e_branch = first_existing(branches, ENERGY_CANDIDATES, args.energy_branch, "energie")
                w_branch = first_existing(branches, WEIGHT_CANDIDATES, args.weight_branch, "poids")
                f_branch = first_existing(branches, FLAVOR_CANDIDATES, args.flavor_branch, "saveur")
                z_branch = first_existing(branches, Z_CANDIDATES, args.z_branch, "z decay")
                importance_branch = first_existing(
                    branches,
                    IMPORTANCE_CANDIDATES,
                    args.importance_branch,
                    "poids d'importance",
                )
                nd_index, fd_index, locations = resolve_detector_indices(
                    root_file,
                    args.nd_index,
                    args.fd_index,
                )
                if resolved_indices is None:
                    resolved_indices = (nd_index, fd_index)
                    resolved_locations = locations
                    print(f"Locations dk2nu: {locations}")
                    print(f"Rayons utilises: ND index={nd_index}, FD index={fd_index}")
                elif resolved_indices != (nd_index, fd_index):
                    raise RuntimeError(
                        f"Indices detecteur variables entre fichiers: {resolved_indices} puis {(nd_index, fd_index)}"
                    )

                arrays = tree.arrays(
                    [e_branch, w_branch, f_branch, z_branch, importance_branch],
                    library="ak",
                )
                e_nd = detector_ray_numpy(arrays[e_branch], nd_index)
                e_fd = detector_ray_numpy(arrays[e_branch], fd_index)
                ray_w_nd = detector_ray_numpy(arrays[w_branch], nd_index)
                ray_w_fd = detector_ray_numpy(arrays[w_branch], fd_index)
                flavor = to_flat_numpy(arrays[f_branch], None).astype(int)
                z_raw = to_flat_numpy(arrays[z_branch], None)
                importance = to_flat_numpy(arrays[importance_branch], None)
        except Exception as exc:
            skipped += 1
            print(f"[warn] fichier ROOT ignore: {path} ({exc})", file=sys.stderr)
            continue

        n = min(
            len(e_nd),
            len(e_fd),
            len(ray_w_nd),
            len(ray_w_fd),
            len(flavor),
            len(z_raw),
            len(importance),
        )
        if n == 0:
            continue
        total_events += n
        e_nd = e_nd[:n]
        e_fd = e_fd[:n]
        flavor = flavor[:n]
        z_m = z_raw[:n] * (0.01 if args.z_unit == "cm" else 1.0) - args.z_offset_m
        weight_nd = ray_w_nd[:n] * importance[:n]
        weight_fd = ray_w_fd[:n] * importance[:n]

        common = (
            np.isfinite(e_nd)
            & np.isfinite(e_fd)
            & np.isfinite(weight_nd)
            & np.isfinite(weight_fd)
            & np.isfinite(z_m)
            & (weight_nd > 0.0)
            & (weight_fd > 0.0)
            & (e_nd >= args.e_min)
            & (e_nd < args.e_max)
            & (e_fd >= args.e_min)
            & (e_fd < args.e_max)
            & (z_m >= args.z_min)
            & (z_m < args.z_max)
        )
        kept_events += int(np.count_nonzero(common))

        for pdg in FLAVOR_LABELS:
            mask = common & (flavor == pdg)
            if not np.any(mask):
                continue
            item = histograms[pdg]
            item["nd_ez"] += np.histogram2d(
                e_nd[mask],
                z_m[mask],
                bins=[e_edges, z_edges],
                weights=weight_nd[mask],
            )[0]
            item["fd_ez"] += np.histogram2d(
                e_fd[mask],
                z_m[mask],
                bins=[e_edges, z_edges],
                weights=weight_fd[mask],
            )[0]
            item["joint_fd"] += np.histogram2d(
                e_nd[mask],
                e_fd[mask],
                bins=[e_edges, e_edges],
                weights=weight_fd[mask],
            )[0]
            item["joint_count"] += np.histogram2d(
                e_nd[mask],
                e_fd[mask],
                bins=[e_edges, e_edges],
            )[0].astype(np.int64)
            item["nd_e"] += np.histogram(e_nd[mask], bins=e_edges, weights=weight_nd[mask])[0]
            item["fd_e"] += np.histogram(e_fd[mask], bins=e_edges, weights=weight_fd[mask])[0]

        if file_number == 1 or file_number % 20 == 0 or file_number == len(files):
            print(f"Fichiers traites: {file_number}/{len(files)}")

    if total_events == 0:
        raise RuntimeError("Aucun evenement dk2nu apparie exploitable.")
    if skipped:
        print(f"[warn] fichiers ROOT ignores: {skipped}/{len(files)}", file=sys.stderr)
    print(f"Evenements lus: {total_events}")
    print(f"Evenements gardes pour ND/FD: {kept_events}")
    return histograms, e_edges, z_edges, resolved_indices, resolved_locations


def parse_number(text: str) -> float:
    return float(text.replace(",", "."))


def load_reference_flux(path: Path | None):
    if path is None:
        return None
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                rows.append([parse_number(part) for part in parts[:7]])
            except ValueError:
                continue
    if not rows:
        raise RuntimeError(f"Flux de reference illisible: {path}")
    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1:]


def reference_flux_at(reference, pdg: int, energy_GeV: float) -> float:
    if reference is None:
        return 0.0
    e_grid, flux = reference
    col = GLOBES_COLUMN[pdg] - 1
    return float(np.interp(energy_GeV, e_grid, flux[:, col]))


def write_flux_z_csv(args, energy, weight, flavor, z_m):
    e_edges = np.linspace(args.e_min, args.e_max, args.e_bins + 1)
    z_edges = np.linspace(args.z_min, args.z_max, args.z_bins + 1)
    reference = load_reference_flux(args.reference_flux_table)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    kept = (
        np.isfinite(energy)
        & np.isfinite(weight)
        & np.isfinite(z_m)
        & (weight > 0.0)
        & (energy >= args.e_min)
        & (energy < args.e_max)
        & (z_m >= args.z_min)
        & (z_m < args.z_max)
    )

    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["flavor", "E_GeV_bin_low", "E_GeV_bin_high", "z_decay_m_bin_low", "z_decay_m_bin_high", "weight"])
        for pdg, label in FLAVOR_LABELS.items():
            mask = kept & (flavor == pdg)
            if not np.any(mask):
                continue
            hist, _, _ = np.histogram2d(energy[mask], z_m[mask], bins=[e_edges, z_edges], weights=weight[mask])
            for ie in range(args.e_bins):
                row_sum = float(np.sum(hist[ie, :]))
                if reference is not None and row_sum > 0.0:
                    e_center = 0.5 * (e_edges[ie] + e_edges[ie + 1])
                    target = reference_flux_at(reference, pdg, e_center)
                    if target > 0.0:
                        hist[ie, :] *= target / row_sum
                for iz in range(args.z_bins):
                    value = hist[ie, iz]
                    if value <= 0.0:
                        continue
                    writer.writerow([label, e_edges[ie], e_edges[ie + 1], z_edges[iz], z_edges[iz + 1], value])

    print(f"CSV ecrit: {args.out.resolve()}")
    print(f"Evenements lus: {len(energy)}")
    print(f"Evenements gardes dans les plages E/z: {int(np.count_nonzero(kept))}")
    print(f"E range lu: [{np.nanmin(energy):.6g}, {np.nanmax(energy):.6g}] GeV")
    print(f"z range lu: [{np.nanmin(z_m):.6g}, {np.nanmax(z_m):.6g}] m")


def write_binned_flux_z_csv(path, histograms, detector_key, e_edges, z_edges, reference_path):
    reference = load_reference_flux(reference_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "flavor",
                "E_GeV_bin_low",
                "E_GeV_bin_high",
                "z_decay_m_bin_low",
                "z_decay_m_bin_high",
                "weight",
            ]
        )
        for pdg, label in FLAVOR_LABELS.items():
            hist = histograms[pdg][detector_key].copy()
            for ie in range(len(e_edges) - 1):
                row_sum = float(np.sum(hist[ie, :]))
                if reference is not None and row_sum > 0.0:
                    e_center = 0.5 * (e_edges[ie] + e_edges[ie + 1])
                    target = reference_flux_at(reference, pdg, e_center)
                    if target > 0.0:
                        hist[ie, :] *= target / row_sum
                for iz in range(len(z_edges) - 1):
                    value = float(hist[ie, iz])
                    if value <= 0.0:
                        continue
                    writer.writerow(
                        [
                            label,
                            e_edges[ie],
                            e_edges[ie + 1],
                            z_edges[iz],
                            z_edges[iz + 1],
                            value,
                        ]
                    )
    print(f"CSV flux(E,z) ecrit: {path.resolve()}")


def write_absolute_flux_z_csv(path, histograms, e_edges, z_edges, total_pot):
    """Write dPhi/dE contributions in neutrinos / m^2 / GeV / POT."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "flavor",
                "E_GeV_bin_low",
                "E_GeV_bin_high",
                "z_decay_m_bin_low",
                "z_decay_m_bin_high",
                "weight",
            ]
        )
        for pdg, label in FLAVOR_LABELS.items():
            hist = histograms[pdg]
            for ie in range(len(e_edges) - 1):
                delta_e = e_edges[ie + 1] - e_edges[ie]
                # nuray.wgt is the probability through a 100 cm-radius disk.
                # Dividing by pi*(100 cm)^2 and converting cm^-2 to m^-2
                # leaves the factor 1/pi.
                scale = 1.0 / (np.pi * total_pot * delta_e)
                for iz in range(len(z_edges) - 1):
                    value = float(hist[ie, iz] * scale)
                    if value <= 0.0:
                        continue
                    writer.writerow(
                        [
                            label,
                            e_edges[ie],
                            e_edges[ie + 1],
                            z_edges[iz],
                            z_edges[iz + 1],
                            value,
                        ]
                    )
    print(f"CSV flux absolu dk2nu ecrit: {path.resolve()}")
    print("Unite de weight: nu / m^2 / GeV / POT, integree dans le bin de z")


def write_source_profile_z_csv(path, histograms, e_edges, z_edges):
    """Write p(z | E, flavor), normalized independently in each populated E bin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "flavor",
                "E_GeV_bin_low",
                "E_GeV_bin_high",
                "z_decay_m_bin_low",
                "z_decay_m_bin_high",
                "weight",
            ]
        )
        for pdg, label in FLAVOR_LABELS.items():
            hist = histograms[pdg]
            for ie in range(len(e_edges) - 1):
                row_sum = float(np.sum(hist[ie, :]))
                if row_sum <= 0.0:
                    continue
                profile = hist[ie, :] / row_sum
                for iz in range(len(z_edges) - 1):
                    value = float(profile[iz])
                    if value <= 0.0:
                        continue
                    writer.writerow(
                        [
                            label,
                            e_edges[ie],
                            e_edges[ie + 1],
                            z_edges[iz],
                            z_edges[iz + 1],
                            value,
                        ]
                    )
    print(f"CSV profil de source dk2nu ecrit: {path.resolve()}")
    print("Unite de weight: sans dimension; sum_z weight = 1 par bin (saveur, E)")


def write_transfer_csv(path, histograms, e_edges):
    path.parent.mkdir(parents=True, exist_ok=True)
    max_relative_residual = 0.0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "flavor",
                "E_ND_GeV_bin_low",
                "E_ND_GeV_bin_high",
                "E_FD_GeV_bin_low",
                "E_FD_GeV_bin_high",
                "transfer_weight",
                "conditional_EFD_given_END",
                "fd_weight_sum",
                "nd_weight_sum",
                "event_count",
            ]
        )
        for pdg, label in FLAVOR_LABELS.items():
            item = histograms[pdg]
            joint_fd = item["joint_fd"]
            nd_weight = item["nd_e"]
            fd_by_nd = joint_fd.sum(axis=1)
            transfer = np.divide(
                joint_fd,
                nd_weight[:, None],
                out=np.zeros_like(joint_fd),
                where=nd_weight[:, None] > 0.0,
            )
            conditional = np.divide(
                joint_fd,
                fd_by_nd[:, None],
                out=np.zeros_like(joint_fd),
                where=fd_by_nd[:, None] > 0.0,
            )

            predicted_fd = nd_weight @ transfer
            direct_fd = item["fd_e"]
            nonzero = direct_fd > 0.0
            if np.any(nonzero):
                residual = np.max(
                    np.abs(predicted_fd[nonzero] - direct_fd[nonzero]) / direct_fd[nonzero]
                )
                max_relative_residual = max(max_relative_residual, float(residual))

            for i_nd in range(len(e_edges) - 1):
                if nd_weight[i_nd] <= 0.0:
                    continue
                for i_fd in range(len(e_edges) - 1):
                    if joint_fd[i_nd, i_fd] <= 0.0:
                        continue
                    writer.writerow(
                        [
                            label,
                            e_edges[i_nd],
                            e_edges[i_nd + 1],
                            e_edges[i_fd],
                            e_edges[i_fd + 1],
                            transfer[i_nd, i_fd],
                            conditional[i_nd, i_fd],
                            joint_fd[i_nd, i_fd],
                            nd_weight[i_nd],
                            int(item["joint_count"][i_nd, i_fd]),
                        ]
                    )
    print(f"Matrice de transfert ND->FD ecrite: {path.resolve()}")
    print(f"Validation Phi_FD = T Phi_ND: max relative residual={max_relative_residual:.6g}")


def download_public_sources(args):
    args.download_dir.mkdir(parents=True, exist_ok=True)
    for name, url in PUBLIC_DK2NU_TARBALLS.items():
        if args.download_mode != "all" and args.download_mode not in name:
            continue
        target = args.download_dir / f"{name}.tar.gz"
        print(f"Telechargement public: {name}")
        print(f"  URL: {url}")
        print(f"  destination: {target}")
        urllib.request.urlretrieve(url, target)
        if args.extract:
            extract_dir = args.download_dir / name
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(target, "r:gz") as archive:
                archive.extractall(extract_dir)
            print(f"  extrait dans: {extract_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convertit des ntuples dk2nu ROOT en table flux(E,z) CSV.")
    parser.add_argument("inputs", nargs="*", help="Fichiers ROOT dk2nu ou motifs glob.")
    parser.add_argument("--input-glob", help="Motif glob supplementaire pour les fichiers ROOT.")
    parser.add_argument("--out", type=Path, default=Path("data/dune/dk2nu/flux_z_FHC_ND.csv"))
    parser.add_argument("--mode", choices=["FHC", "RHC"], default="FHC")
    parser.add_argument("--det", choices=["ND", "FD"], default="ND")
    parser.add_argument("--tree")
    parser.add_argument("--energy-branch")
    parser.add_argument("--weight-branch")
    parser.add_argument("--importance-branch")
    parser.add_argument("--flavor-branch")
    parser.add_argument("--z-branch")
    parser.add_argument("--det-index", type=int, help="Indice explicite du rayon; sinon utilise --det et dkmetaTree.")
    parser.add_argument("--nd-index", type=int, help="Indice du rayon ND; auto-detecte via dkmetaTree par defaut.")
    parser.add_argument("--fd-index", type=int, help="Indice du rayon FD; auto-detecte via dkmetaTree par defaut.")
    parser.add_argument("--z-unit", choices=["cm", "m"], default="cm")
    parser.add_argument("--z-offset-m", type=float, default=0.0)
    parser.add_argument("--e-min", type=float, default=0.5)
    parser.add_argument("--e-max", type=float, default=8.0)
    parser.add_argument("--e-bins", type=int, default=30)
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=194.0)
    parser.add_argument("--z-bins", type=int, default=80)
    parser.add_argument("--reference-flux-table", type=Path, help="Flux GLoBES tabule utilise pour normaliser sum_z w(E,z).")
    parser.add_argument("--transfer-out", type=Path, help="CSV de la matrice T(E_ND,E_FD).")
    parser.add_argument("--nd-flux-z-out", type=Path, help="CSV flux(E_ND,z) extrait avec le rayon ND.")
    parser.add_argument("--fd-flux-z-out", type=Path, help="CSV flux(E_FD,z) extrait avec le rayon FD.")
    parser.add_argument("--nd-reference-flux-table", type=Path, help="Flux ND utilise pour normaliser le CSV ND apparie.")
    parser.add_argument("--fd-reference-flux-table", type=Path, help="Flux FD utilise pour normaliser le CSV FD apparie.")
    parser.add_argument(
        "--raw-flux-z-out",
        type=Path,
        help="CSV absolu dk2nu: dPhi/dE par m2, GeV et POT, detaille en z.",
    )
    parser.add_argument(
        "--source-profile-z-out",
        type=Path,
        help="CSV du profil conditionnel p(z|E,saveur), normalise a 1 sur z.",
    )
    parser.add_argument("--print-public-sources", action="store_true")
    parser.add_argument("--download-public", action="store_true")
    parser.add_argument("--download-mode", choices=["all", "FHC", "RHC"], default="all")
    parser.add_argument("--download-dir", type=Path, default=Path("data/dune/dk2nu/raw"))
    parser.add_argument("--extract", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.print_public_sources:
        for name, url in PUBLIC_DK2NU_TARBALLS.items():
            print(f"{name}: {url}")
        if not args.download_public and not args.inputs and not args.input_glob:
            return
    if args.download_public:
        download_public_sources(args)
        if not args.inputs and not args.input_glob:
            return

    files = resolve_inputs(args.inputs, args.input_glob)
    if not files:
        raise SystemExit(
            "Aucun fichier dk2nu ROOT fourni/trouve. "
            "Utilise --print-public-sources ou --download-public si tu veux recuperer les tarballs publics."
        )

    paired_mode = any((args.transfer_out, args.nd_flux_z_out, args.fd_flux_z_out))
    physical_outputs = any((args.raw_flux_z_out, args.source_profile_z_out))
    if paired_mode and physical_outputs:
        raise SystemExit(
            "--raw-flux-z-out/--source-profile-z-out utilisent un seul rayon detecteur; "
            "ne pas les combiner avec les sorties appariees ND/FD."
        )
    print(
        f"Mode={args.mode}, det={args.det}, fichiers={len(files)}, "
        f"paired={int(paired_mode)}, physical_outputs={int(physical_outputs)}"
    )
    if paired_mode:
        histograms, e_edges, z_edges, _, _ = accumulate_paired_histograms(args, files)
        if args.nd_flux_z_out:
            write_binned_flux_z_csv(
                args.nd_flux_z_out,
                histograms,
                "nd_ez",
                e_edges,
                z_edges,
                args.nd_reference_flux_table,
            )
        if args.fd_flux_z_out:
            write_binned_flux_z_csv(
                args.fd_flux_z_out,
                histograms,
                "fd_ez",
                e_edges,
                z_edges,
                args.fd_reference_flux_table,
            )
        if args.transfer_out:
            write_transfer_csv(args.transfer_out, histograms, e_edges)
    elif physical_outputs:
        histograms, e_edges, z_edges, total_pot = accumulate_detector_histogram(args, files)
        if args.raw_flux_z_out:
            write_absolute_flux_z_csv(
                args.raw_flux_z_out,
                histograms,
                e_edges,
                z_edges,
                total_pot,
            )
        if args.source_profile_z_out:
            write_source_profile_z_csv(
                args.source_profile_z_out,
                histograms,
                e_edges,
                z_edges,
            )
    else:
        energy, weight, flavor, z_m = load_arrays(args, files)
        write_flux_z_csv(args, energy, weight, flavor, z_m)


if __name__ == "__main__":
    main()
