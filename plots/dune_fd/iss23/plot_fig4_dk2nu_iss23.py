from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MPLCONFIGDIR = ROOT / ".matplotlib_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
sys.path.insert(0, str(ROOT / "plots" / "dune_fd"))

from plot_fig4_iss23_vs_globes import main as plot_main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--input",
                str(ROOT / "data" / "dune_nd" / "minimal_onaxis" / "point_70" / "plots_validation" / "fig4_fd_iss23_vs_globes.csv"),
                "--out",
                str(ROOT / "figures" / "dune_fd" / "iss23" / "construct23_point70" / "fig4" / "fig4_fd_dk2nu_iss23_vs_active3nu.png"),
            ]
        )
    plot_main()
