from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

from inverse_construct_23_config import get_inverse_construct_23_csv_path


FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_all_floats(value: str) -> list[float]:
    return [float(x) for x in FLOAT_PATTERN.findall(value)]


def _parse_kept_point(path: Path) -> dict[str, float | int]:
    data: dict[str, float | int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("===") or "=" not in line:
            continue
        key, value = [x.strip() for x in line.split("=", 1)]

        if key == "eta_abs_3x3":
            continue

        if key == "mu_H_2x2_eV":
            values = _parse_all_floats(value)
            if len(values) == 4:
                data["muH11_eV"] = values[0]
                data["muH12_eV"] = values[1]
                data["muH21_eV"] = values[2]
                data["muH22_eV"] = values[3]
            continue

        if key == "mu_H0_2x1_eV":
            values = _parse_all_floats(value)
            if len(values) == 2:
                data["muH01_eV"] = values[0]
                data["muH02_eV"] = values[1]
            continue

        if key == "mu3_eV":
            values = _parse_all_floats(value)
            if len(values) == 9:
                data["mu3_11_eV"] = values[0]
                data["mu3_12_eV"] = values[1]
                data["mu3_13_eV"] = values[2]
                data["mu3_21_eV"] = values[3]
                data["mu3_22_eV"] = values[4]
                data["mu3_23_eV"] = values[5]
                data["mu3_31_eV"] = values[6]
                data["mu3_32_eV"] = values[7]
                data["mu3_33_eV"] = values[8]
            continue

        try:
            num = float(value)
        except ValueError:
            continue

        if key in {"point_id", "sample_id", "solve_ok", "pmns_pass", "eta_pass"}:
            data[key] = int(num)
        else:
            data[key] = num

    return data


def load_kept_points_dataframe(
    data_dir: Path,
    min_point_id: int = 1,
    max_workers: int | None = None,
    csv_path: Path | None = None,
) -> pd.DataFrame:
    resolved_csv_path = csv_path if csv_path is not None else get_inverse_construct_23_csv_path()
    if resolved_csv_path.exists():
        df = pd.read_csv(resolved_csv_path)
        if df.empty:
            return df

        if min_point_id > 1 and "point_id" in df.columns:
            df = df[df["point_id"] >= min_point_id].copy()

        for col in ["solve_ok", "pmns_pass", "eta_pass"]:
            if col not in df.columns:
                df[col] = 1 if col in {"solve_ok", "pmns_pass"} else 0
            df[col] = df[col].fillna(0).astype(int)

        return df

    files = sorted(
        [p for p in data_dir.glob("*.txt") if p.stem.isdigit() and int(p.stem) >= min_point_id],
        key=lambda p: int(p.stem),
    )
    total = len(files)
    if total == 0:
        return pd.DataFrame()

    rows: list[dict] = [None] * total  # type: ignore[list-item]

    def _worker(args: tuple[int, Path]) -> tuple[int, dict | None]:
        i, p = args
        try:
            return i, _parse_kept_point(p)
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            return i, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, (i, p)): i for i, p in enumerate(files)}
        done = 0
        for fut in as_completed(futures):
            i, result = fut.result()
            rows[i] = result
            done += 1
            if done == 1 or done % 2000 == 0 or done == total:
                print(f"[kept-points] parsed {done}/{total} files", end="\r", flush=True)

    print(" " * 60, end="\r")
    print(f"[kept-points] parsed {total}/{total} files")

    valid_rows = [r for r in rows if r is not None]
    if not valid_rows:
        return pd.DataFrame()

    df = pd.DataFrame(valid_rows)
    for col in ["solve_ok", "pmns_pass", "eta_pass"]:
        if col not in df.columns:
            df[col] = 1 if col in {"solve_ok", "pmns_pass"} else 0
        df[col] = df[col].fillna(0).astype(int)

    return df
