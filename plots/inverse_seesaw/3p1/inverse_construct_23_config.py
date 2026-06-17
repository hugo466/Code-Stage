from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "presets" / "inverse_seesaw" / "3p1" / "construct_23.txt"
DEFAULT_KEPT_POINTS_DIR = PROJECT_ROOT / "data" / "inverse_seesaw" / "3p1" / "inverse_construct_23_kept_points"


def load_config_value(key: str, default: str | None = None) -> str | None:
    if not CONFIG_PATH.exists():
        return default

    prefix = f"{key} ="
    for raw_line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    return default


def get_inverse_kept_points_dir() -> Path:
    raw_value = load_config_value("inverse_kept_points_dir")
    if not raw_value:
        return DEFAULT_KEPT_POINTS_DIR

    path = Path(raw_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def get_inverse_construct_23_csv_path() -> Path:
    raw_value = load_config_value("output_inverse_construct_23_csv_path")
    if raw_value:
        path = Path(raw_value)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path

    return get_inverse_kept_points_dir() / "inverse_construct_23_kept_points.csv"
