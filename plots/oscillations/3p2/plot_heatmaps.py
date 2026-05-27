import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

DATA_PATH = Path("data/oscillations/3p2/delta_pmue_heatmap_3p2.csv")
OUTPUT_PATH = Path("figures/oscillations/3p2/delta_pmue_heatmap_3p2_4panels.png")


def load_data(path: Path):
	rows = []
	with path.open("r", newline="", encoding="utf-8") as file:
		reader = csv.DictReader(file)
		for row in reader:
			if not row:
				continue
			required = ("energy_GeV", "dm41_eV2", "dm54_eV2", "delta_P_mue")
			if any(row.get(key) in (None, "") for key in required):
				continue
			rows.append((
				float(row["energy_GeV"]),
				float(row["dm41_eV2"]),
				float(row["dm54_eV2"]),
				float(row["delta_P_mue"]),
			))

	if not rows:
		raise RuntimeError("CSV heatmap vide.")

	return rows


def build_grid(rows, dm41_target):
	filtered = [row for row in rows if abs(row[1] - dm41_target) < 1e-12]
	energies = sorted({item[0] for item in filtered})
	dm54_values = sorted({item[2] for item in filtered})

	energy_idx = {value: idx for idx, value in enumerate(energies)}
	dm54_idx = {value: idx for idx, value in enumerate(dm54_values)}

	grid = np.full((len(dm54_values), len(energies)), np.nan)
	for energy, _, dm54, delta in filtered:
		grid[dm54_idx[dm54], energy_idx[energy]] = delta

	return np.array(energies), np.array(dm54_values), grid


def main():
	if not DATA_PATH.exists():
		raise FileNotFoundError(f"CSV introuvable: {DATA_PATH}")

	rows = load_data(DATA_PATH)
	dm41_values = sorted({item[1] for item in rows})
	if len(dm41_values) < 4:
		raise RuntimeError("Le CSV doit contenir au moins 4 valeurs distinctes de dm41_eV2.")

	dm41_values = dm41_values[:4]

	all_delta = np.array([row[3] for row in rows], dtype=float)
	abs_max = float(np.nanmax(np.abs(all_delta))) if all_delta.size else 1.0
	if not np.isfinite(abs_max) or abs_max == 0.0:
		abs_max = 1.0
	norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

	fig = plt.figure(figsize=(12.5, 8.2), constrained_layout=True)
	grid_spec = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.06])
	axes = [
		fig.add_subplot(grid_spec[0, 0]),
		fig.add_subplot(grid_spec[0, 1]),
		fig.add_subplot(grid_spec[1, 0]),
		fig.add_subplot(grid_spec[1, 1]),
	]
	cbar_ax = fig.add_subplot(grid_spec[:, 2])
	image = None

	for idx, dm41 in enumerate(dm41_values):
		ax = axes[idx]
		energies, dm54_values, grid = build_grid(rows, dm41)
		image = ax.pcolormesh(energies, dm54_values, grid, shading="nearest", cmap="RdBu_r", norm=norm)
		ax.set_title(rf"$\Delta m_{{41}}^2 = {dm41:g}\,\mathrm{{eV}}^2$", pad=8)

	# Axes labels only where needed to avoid visual crowding
	for idx, ax in enumerate(axes):
		if idx in (2, 3):
			ax.set_xlabel("Énergie [GeV]")
		else:
			ax.set_xlabel("")
		if idx in (0, 2):
			ax.set_ylabel(r"$\Delta m_{45}^2$ [eV$^2$]")
		else:
			ax.set_ylabel("")

	colorbar = fig.colorbar(image, cax=cbar_ax)
	colorbar.set_label(r"$\Delta P_{\mu e}$", fontsize = 10)
	fig.suptitle(r"$\Delta P_{\mu e}=P_{\mu e}^{3+2}-P_{\mu e}^{3\nu}$ en fonction de $(E,\Delta m_{45}^2)$")

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(OUTPUT_PATH, dpi=180)
	print(f"Figure sauvegardée: {OUTPUT_PATH}")
	plt.close(fig)


if __name__ == "__main__":
	main()
