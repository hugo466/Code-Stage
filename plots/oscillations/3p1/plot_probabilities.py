from pathlib import Path
import runpy

SCRIPT = Path(__file__).resolve().parent / "plot_3p1_probabilities.py"
runpy.run_path(str(SCRIPT), run_name="__main__")
