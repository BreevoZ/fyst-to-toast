"""The script help keeps copy-pasteable examples on separate lines."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("script,example", [
    ("patch_to_ecsv.py", "  python scripts/patch_to_ecsv.py ces --sample-rate 5\n"),
    ("window_ces_to_ecsv.py", "      --window-file examples/windows/deep56_e50_rising.yaml \\\n"),
    ("save_trajectory_h5.py", "  python scripts/save_trajectory_h5.py output/generated/ces/fyst_trajectory.ecsv\n"),
])
def test_cli_help(script, example):
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / script), "--help"],
        cwd=repo, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "\nExamples (from the repository root" in result.stdout
    assert example in result.stdout
