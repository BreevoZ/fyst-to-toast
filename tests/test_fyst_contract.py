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

from fyst_to_toast import fyst_flags

iers.conf.auto_download = False
pytest.importorskip(
    "fyst_trajectories",
    reason="FYST contract tests require the optional fyst extra (unavailable or blocked)",
    exc_type=ImportError,
)

REPO = Path(__file__).resolve().parents[1]


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
def test_writer_schedule_contract(case, tmp_path):
    """FYST writes header-marked TOAST roll and the site the writer CLI reads."""
    import numpy as np
    from astropy import units as u
    from astropy.table import QTable
    from astropy.time import Time
    from fyst_trajectories.overhead import read_timeline

    from fyst_to_toast.generate import (
        DEFAULT_START,
        build_legacy_patch,
        generate_observing_timeline,
        write_timeline_ecsv,
    )

    start = Time(DEFAULT_START[case], scale="utc", precision=3)
    timeline = generate_observing_timeline(build_legacy_patch(case), start, 1800)
    write_timeline_ecsv(timeline, tmp_path)
    schedule = tmp_path / "full_timeline.ecsv"
    read_site_metadata = runpy.run_path(str(REPO / "scripts/save_trajectory_h5.py"))[
        "read_site_metadata"
    ]
    native = read_timeline(schedule).site
    assert read_site_metadata(schedule) == (
        native.name,
        native.latitude,
        native.longitude,
        native.elevation,
    )
    for name in ("full_timeline.ecsv", "schedule.ecsv"):
        table = QTable.read(tmp_path / name, format="ascii.ecsv")
        assert len(table) > 0
        assert table.meta["boresight_angle_convention"] == "toast_roll"
        assert "field_rotation" not in table.colnames
        np.testing.assert_array_equal(
            table["boresight_angle"].to_value(u.deg),
            -native.nasmyth_sign * table["el"].to_value(u.deg),
        )
