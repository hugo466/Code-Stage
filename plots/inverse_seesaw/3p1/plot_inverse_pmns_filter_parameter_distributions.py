import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from inverse_pmns_filter_config import get_inverse_kept_points_dir

DATA_DIR = get_inverse_kept_points_dir()
OUTPUT_PATH = Path("figures/inverse_seesaw/3p1/inverse_pmns_filter_parameter_distributions.png")
OUTPUT_CORR_PEARSON_PATH = Path("figures/inverse_seesaw/3p1/inverse_pmns_filter_correlation_pearson.png")
OUTPUT_CORR_SPEARMAN_PATH = Path("figures/inverse_seesaw/3p1/inverse_pmns_filter_correlation_spearman.png")
OUTPUT_CORR_CSV_PATH = Path("data/inverse_seesaw/3p1/inverse_pmns_filter_correlation_matrices.csv")
MIN_POINT_ID = 1

FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_scalar(line: str) -> float:
    return float(line.split("=", 1)[1].strip())


def parse_all_floats(text: str):
    rhs = text.split("=", 1)[1] if "=" in text else text
    return [float(x) for x in FLOAT_PATTERN.findall(rhs)]


def load_distributions(data_dir: Path, min_point_id: int):
    def empty_parameter_groups():
        return {
            "mu": {
                "mu00": [],
                "muH11": [],
                "muH12": [],
                "muH21": [],
                "muH22": [],
                "muH01": [],
                "muH02": [],
            },
            "md": {
                "mD11": [], "mD12": [],
                "mD21": [], "mD22": [],
                "mD31": [], "mD32": [],
            },
            "mr": {
                "M11": [], "M12": [],
                "M21": [], "M22": [],
            },
        }

    eta_fail = empty_parameter_groups()
    eta_pass_group = empty_parameter_groups()

    selected_files = sorted(
        [p for p in data_dir.glob("*.txt") if p.stem.isdigit() and int(p.stem) >= min_point_id],
        key=lambda p: int(p.stem),
    )

    for point_file in selected_files:
        lines = point_file.read_text(encoding="utf-8").splitlines()
        eta_pass = 0
        point_mu = {}
        point_md = {}
        point_mr = {}

        for line in lines:
            if line.startswith("eta_pass"):
                eta_pass = int(float(line.split("=", 1)[1].strip()))
            elif line.startswith("mu00_eV"):
                point_mu["mu00"] = parse_scalar(line)
            elif line.startswith("M_2x2_GeV"):
                values = parse_all_floats(line)
                point_mr["M11"] = values[0]
                point_mr["M12"] = values[1]
                point_mr["M21"] = values[2]
                point_mr["M22"] = values[3]
            elif line.startswith("mD_3x2_GeV"):
                values = parse_all_floats(line)
                point_md["mD11"] = values[0]
                point_md["mD12"] = values[1]
                point_md["mD21"] = values[2]
                point_md["mD22"] = values[3]
                point_md["mD31"] = values[4]
                point_md["mD32"] = values[5]
            elif line.startswith("mu_H_2x2_eV"):
                values = parse_all_floats(line)
                point_mu["muH11"] = values[0]
                point_mu["muH12"] = values[1]
                point_mu["muH21"] = values[2]
                point_mu["muH22"] = values[3]
            elif line.startswith("mu_H0_2x1_eV"):
                values = parse_all_floats(line)
                point_mu["muH01"] = values[0]
                point_mu["muH02"] = values[1]

        if eta_pass == 0:
            target = eta_fail
        else:
            target = eta_pass_group

        for key, value in point_mu.items():
            target["mu"][key].append(value)
        for key, value in point_md.items():
            target["md"][key].append(value)
        for key, value in point_mr.items():
            target["mr"][key].append(value)

    return selected_files, eta_pass_group, eta_fail


def plot_parameter(ax, label, eta_pass_values, eta_fail_values, xlabel):
    values = list(eta_pass_values) + list(eta_fail_values)
    if values:
        bins = max(35, min(120, int(len(values) ** 0.5)))
        ax.hist(
            [eta_pass_values, eta_fail_values],
            bins=bins,
            density=True,
            stacked=True,
            color=["red", "#1f77b4"],
            alpha=0.8,
            linewidth=0.6,
            edgecolor="black",
        )

    ax.set_title(label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densité de probabilité")
    ax.grid(alpha=0.25)


def build_parameter_matrix(parameter_specs):
    labels = []
    columns = []
    for label, values, _ in parameter_specs:
        labels.append(label)
        columns.append(np.array(values, dtype=float))

    if not columns:
        raise RuntimeError("Aucune variable disponible pour l'analyse de corrélation.")

    sizes = {col.size for col in columns}
    if len(sizes) != 1:
        raise RuntimeError("Tailles incohérentes entre variables: impossible de calculer les corrélations.")

    matrix = np.column_stack(columns)
    return labels, matrix


def rank_columns(matrix):
    ranks = np.zeros_like(matrix, dtype=float)
    for idx in range(matrix.shape[1]):
        order = np.argsort(matrix[:, idx], kind="mergesort")
        ranks[order, idx] = np.arange(1, matrix.shape[0] + 1, dtype=float)
    return ranks


def correlation_matrices(matrix):
    pearson = np.corrcoef(matrix, rowvar=False)
    spearman = np.corrcoef(rank_columns(matrix), rowvar=False)
    return pearson, spearman


def plot_correlation_heatmap(corr, labels, title, output_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Coefficient de corrélation")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_correlation_csv(labels, pearson, spearman, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        file.write("[Pearson]\n")
        file.write("," + ",".join(labels) + "\n")
        for label, row in zip(labels, pearson):
            file.write(label + "," + ",".join(f"{value:.8f}" for value in row) + "\n")

        file.write("\n[Spearman]\n")
        file.write("," + ",".join(labels) + "\n")
        for label, row in zip(labels, spearman):
            file.write(label + "," + ",".join(f"{value:.8f}" for value in row) + "\n")


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dossier introuvable. Vérifie inverse_kept_points_dir dans la config: {DATA_DIR}"
        )

    selected_files, eta_pass_group, eta_fail = load_distributions(DATA_DIR, MIN_POINT_ID)
    point_count = len(selected_files)
    if point_count == 0:
        raise RuntimeError(f"Aucun point trouvé à partir de l'identifiant {MIN_POINT_ID}.")

    eta_pass_count = len(eta_pass_group["mu"]["mu00"])
    eta_fail_count = len(eta_fail["mu"]["mu00"])

    parameter_specs = [
        ("mu00", eta_pass_group["mu"]["mu00"], eta_fail["mu"]["mu00"], "Valeur [eV]"),
        ("muH11", eta_pass_group["mu"]["muH11"], eta_fail["mu"]["muH11"], "Valeur [eV]"),
        ("muH12", eta_pass_group["mu"]["muH12"], eta_fail["mu"]["muH12"], "Valeur [eV]"),
        ("muH21", eta_pass_group["mu"]["muH21"], eta_fail["mu"]["muH21"], "Valeur [eV]"),
        ("muH22", eta_pass_group["mu"]["muH22"], eta_fail["mu"]["muH22"], "Valeur [eV]"),
        ("muH01", eta_pass_group["mu"]["muH01"], eta_fail["mu"]["muH01"], "Valeur [eV]"),
        ("muH02", eta_pass_group["mu"]["muH02"], eta_fail["mu"]["muH02"], "Valeur [eV]"),
        ("mD11", eta_pass_group["md"]["mD11"], eta_fail["md"]["mD11"], "Valeur [GeV]"),
        ("mD12", eta_pass_group["md"]["mD12"], eta_fail["md"]["mD12"], "Valeur [GeV]"),
        ("mD21", eta_pass_group["md"]["mD21"], eta_fail["md"]["mD21"], "Valeur [GeV]"),
        ("mD22", eta_pass_group["md"]["mD22"], eta_fail["md"]["mD22"], "Valeur [GeV]"),
        ("mD31", eta_pass_group["md"]["mD31"], eta_fail["md"]["mD31"], "Valeur [GeV]"),
        ("mD32", eta_pass_group["md"]["mD32"], eta_fail["md"]["mD32"], "Valeur [GeV]"),
        ("M11", eta_pass_group["mr"]["M11"], eta_fail["mr"]["M11"], "Valeur [GeV]"),
        ("M12", eta_pass_group["mr"]["M12"], eta_fail["mr"]["M12"], "Valeur [GeV]"),
        ("M21", eta_pass_group["mr"]["M21"], eta_fail["mr"]["M21"], "Valeur [GeV]"),
        ("M22", eta_pass_group["mr"]["M22"], eta_fail["mr"]["M22"], "Valeur [GeV]"),
    ]

    ncols = 3
    nrows = (len(parameter_specs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.0 * nrows))
    axes_flat = axes.flatten()

    for ax, (label, eta_pass_values, eta_fail_values, xlabel) in zip(axes_flat, parameter_specs):
        plot_parameter(ax, label, eta_pass_values, eta_fail_values, xlabel)

    for ax in axes_flat[len(parameter_specs):]:
        ax.axis("off")

    axes_flat[0].legend(
        ["PMNS + eta", "PMNS sans eta"],
        loc="best",
        framealpha=0.9,
    )

    fig.suptitle(
        rf"Distributions des paramètres gardés ({point_count} points, rouge={eta_pass_count}, bleu={eta_fail_count})",
        fontsize=15,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Figure sauvegardée: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
