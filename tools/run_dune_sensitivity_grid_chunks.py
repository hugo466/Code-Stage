from __future__ import annotations

import argparse
import concurrent.futures as futures
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRESET_DIR = ROOT / "config" / "presets" / "dune" / "sensitivity"
CHUNK_PRESET_DIR = PRESET_DIR / "_chunks"
DATA_DIR = ROOT / "data" / "dune" / "sensitivity"
CHUNK_RESULT_DIR = DATA_DIR / "chunks"
LOG_DIR = DATA_DIR / "logs"
APP = ROOT / "bin" / "app.exe"


GRIDS = {
    "theta14_uniform": (
        "baseline_effects_analytic_3p1_theta14_dense_noshape.ini",
        "baseline_effects_analytic_3p1_theta14_uniform_dense_noshape.csv",
    ),
    "theta14_point": (
        "baseline_effects_analytic_3p1_theta14_point_dense_noshape.ini",
        "baseline_effects_analytic_3p1_theta14_point_dense_noshape.csv",
    ),
    "theta24_uniform": (
        "baseline_effects_analytic_3p1_theta24_dense_noshape.ini",
        "baseline_effects_analytic_3p1_theta24_uniform_dense_noshape.csv",
    ),
    "theta24_point": (
        "baseline_effects_analytic_3p1_theta24_point_dense_noshape.ini",
        "baseline_effects_analytic_3p1_theta24_point_dense_noshape.csv",
    ),
}


def write_chunk_preset(name: str, base_preset: str, offset: int, count: int) -> tuple[Path, Path, Path]:
    CHUNK_PRESET_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_offset{offset:04d}_n{count:04d}"
    preset = CHUNK_PRESET_DIR / f"{stem}.ini"
    csv = CHUNK_RESULT_DIR / f"{stem}.csv"
    log = LOG_DIR / f"{stem}.log"
    text = (
        f"include = ../{base_preset}\n\n"
        "[sensitivity]\n"
        f"point_offset = {offset}\n"
        f"max_points = {count}\n"
        f"output_csv = {csv.as_posix()}\n"
    )
    preset.write_text(text, encoding="utf-8")
    return preset, csv, log


def run_chunk(task: tuple[str, str, int, int]) -> tuple[str, int, int, Path]:
    name, base_preset, offset, count = task
    preset, csv, log = write_chunk_preset(name, base_preset, offset, count)
    with log.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            [str(APP), str(preset)],
            cwd=ROOT,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"{name} offset={offset} failed, see {log}")
    return name, offset, count, csv


def combine_csvs(name: str, csvs: list[Path], final_csv: Path) -> None:
    final_csv.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    with final_csv.open("w", encoding="utf-8", newline="") as out:
        for csv in sorted(csvs):
            lines = csv.read_text(encoding="utf-8").splitlines()
            if not lines:
                continue
            if not wrote_header:
                out.write(lines[0] + "\n")
                wrote_header = True
            for line in lines[1:]:
                if line.strip():
                    out.write(line + "\n")
    if not wrote_header:
        raise RuntimeError(f"No data produced for {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dense DUNE sensitivity grids in robust chunks.")
    parser.add_argument("--grids", nargs="+", default=list(GRIDS), choices=list(GRIDS))
    parser.add_argument("--total-points", type=int, default=625)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    tasks: list[tuple[str, str, int, int]] = []
    for name in args.grids:
        base, _ = GRIDS[name]
        for offset in range(0, args.total_points, args.chunk_size):
            count = min(args.chunk_size, args.total_points - offset)
            tasks.append((name, base, offset, count))

    produced: dict[str, list[Path]] = {name: [] for name in args.grids}
    done_chunks = 0
    done_points = 0
    total_points = sum(task[3] for task in tasks)
    start = time.monotonic()
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(run_chunk, task): task for task in tasks}
        for fut in futures.as_completed(future_map):
            name, offset, count, csv = fut.result()
            produced[name].append(csv)
            done_chunks += 1
            done_points += count
            elapsed = max(1.0e-9, time.monotonic() - start)
            points_per_sec = done_points / elapsed
            eta = (total_points - done_points) / points_per_sec if points_per_sec > 0.0 else 0.0
            bar_width = 28
            filled = int(round(bar_width * done_points / total_points)) if total_points > 0 else bar_width
            bar = "#" * filled + "-" * (bar_width - filled)
            print(
                f"[{bar}] {done_points}/{total_points} pts "
                f"({100.0 * done_points / total_points:5.1f}%) | "
                f"{done_chunks}/{len(tasks)} chunks | "
                f"{points_per_sec:5.2f} pts/s | ETA {eta/60.0:5.1f} min | "
                f"{name} offset={offset} n={count}",
                flush=True,
            )

    for name in args.grids:
        _, final_name = GRIDS[name]
        combine_csvs(name, produced[name], DATA_DIR / final_name)
        print(f"combined {name} -> data/dune/sensitivity/{final_name}")


if __name__ == "__main__":
    main()
