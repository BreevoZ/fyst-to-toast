"""Optional FYST generation -> ECSV -> writer input contract."""

import os

os.environ["MPI_DISABLE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import hashlib
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.table import QTable
from astropy.utils import iers

pytest.importorskip(
    "fyst_trajectories",
    reason="Generation requires the optional fyst extra (unavailable or blocked)",
    exc_type=ImportError,
)

from fyst_to_toast import trajectory_h5 as schema
from fyst_to_toast.generate import generate_trajectory


# Pong's documented night includes an initial planet calibration before science.
@pytest.mark.parametrize("case,duration_s", [("ces", 1800), ("pong", 2400)])
def test_ecsv_writer_contract(tmp_path, case, duration_s):
    iers_path = os.environ.get("FYST_TO_TOAST_TEST_IERS")
    with iers.conf.set_temp("auto_download", False):
        table = iers.IERS_A.open(iers_path) if iers_path else iers.IERS_Auto.open()
        with (
            iers.earth_orientation_table.set(table),
            solar_system_ephemeris.set("builtin"),
        ):
            report = generate_trajectory(
                case, tmp_path, duration_s=duration_s, iers_file=iers_path
            )
            assert report["status"] == "complete"
            source = tmp_path / "fyst_trajectory.ecsv"
            samples = QTable.read(source)
            assert samples.meta[schema.ECSV_CASE] == case
            assert tuple(samples.colnames) == schema.TRAJECTORY_COLUMNS
            assert set(samples.meta) == set(schema.TRAJECTORY_META_KEYS)
            roll = samples[schema.ECSV_BORESIGHT_ROLL_DEG]
            assert roll.unit == u.deg and roll.dtype == np.float64
            np.testing.assert_array_equal(roll.value, np.zeros(len(samples)))
            assert (
                samples.meta[schema.ROLL_SOURCE_KEY]
                == schema.ROLL_SOURCE_PLACEHOLDER_ZERO
            )
            assert samples.meta[schema.ROLL_CONVENTION_KEY] == schema.ROLL_CONVENTION
            blocks = samples.meta[schema.ECSV_BLOCKS]
            assert len(blocks) == len(report["science_blocks"]) == 1
            for block in blocks:
                assert set(block) == set(schema.TRAJECTORY_BLOCK_KEYS)
            assert len(samples) == report["total_science_samples"] > 0

            schedule = QTable.read(tmp_path / "schedule.ecsv")
            if case == "pong":
                assert set(schedule["scan_type"]) == {"pong"}
            site = schedule.meta
            folder = tmp_path / "hdf5"
            folder.mkdir()
            files = schema.write_trajectory_blocks(
                samples,
                site_name=str(site["site_name"]),
                latitude_deg=site["site_lat"].to_value(u.deg),
                longitude_deg=site["site_lon"].to_value(u.deg),
                height_m=site["site_alt"].to_value(u.m),
                output_dir=folder,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            assert len(files) == len(blocks)
            assert all((Path(folder) / name).is_file() for name in files)
