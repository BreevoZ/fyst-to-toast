"""Optional semantic contracts against the installed FYST implementation."""

import os

# Configure direct collection before Astropy / NumPy or FYST can load them.
os.environ["MPI_DISABLE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import runpy
from pathlib import Path

import pytest
from astropy.utils import iers

from fyst_toast import fyst_flags

iers.conf.auto_download = False
pytest.importorskip(
    "fyst_trajectories",
    reason="FYST contract tests require the optional fyst extra (unavailable or blocked)",
    exc_type=ImportError,
)

REPO = Path(__file__).resolve().parents[1]
SCHEDULE_ROOT = (
    REPO
    / Path(
        os.environ.get(
            "FYST_TOAST_TEST_SCHEDULE_ROOT",
            "../legacy_repo/fyst-trajectories-sandbox-2026-10/"
            "examples/fyst_to_toast/output",
        )
    ).expanduser()
).resolve()


@pytest.mark.parametrize(
    "name",
    [
        "SCAN_FLAG_UNCLASSIFIED",
        "SCAN_FLAG_SCIENCE",
        "SCAN_FLAG_TURNAROUND",
        "SCAN_FLAG_RETUNE",
    ],
)
def test_scan_flag_contract(name):
    from fyst_trajectories import trajectory

    value = getattr(fyst_flags, name)
    assert type(value) is int
    assert value == getattr(trajectory, name)


@pytest.mark.parametrize("case", ["ces", "daisy"])
def test_legacy_site_contract(case):
    from fyst_trajectories.overhead import read_timeline

    schedule = SCHEDULE_ROOT / case / "schedule.ecsv"
    if not schedule.is_file():
        pytest.skip(
            f"Legacy schedule absent: {schedule}; "
            "set FYST_TOAST_TEST_SCHEDULE_ROOT to supply it"
        )
    read_site_metadata = runpy.run_path(str(REPO / "examples/02_save_trajectory.py"))[
        "read_site_metadata"
    ]
    native = read_timeline(schedule).site
    assert read_site_metadata(schedule) == (
        native.name,
        native.latitude,
        native.longitude,
        native.elevation,
    )
