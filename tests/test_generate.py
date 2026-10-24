"""Optional FYST generation -> ECSV -> writer input contract."""

import os

os.environ["MPI_DISABLE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.table import QTable
from astropy.time import Time
from astropy.utils import iers

pytest.importorskip(
    "fyst_trajectories",
    reason="Generation requires the optional fyst extra (unavailable or blocked)",
    exc_type=ImportError,
)

from fyst_to_toast import trajectory_h5 as schema
from fyst_to_toast.generate import build_trajectory_table, generate_trajectory


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
            np.testing.assert_array_equal(roll.value, -samples[schema.ECSV_EL_DEG].value)
            if case == "ces":
                np.testing.assert_array_equal(roll.value, np.full(len(samples), -50.0))
            else:
                assert np.ptp(roll.value) > 0
            assert samples.meta[schema.ROLL_SOURCE_KEY] == schema.ROLL_SOURCE_FYST
            assert samples.meta[schema.ROLL_CONVENTION_KEY] == schema.ROLL_CONVENTION
            assert samples.meta[schema.ROLL_NASMYTH_PORT_KEY] == "right"
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


@pytest.mark.parametrize("port,sign", [("right", 1), ("left", -1), ("cassegrain", 0)])
def test_build_nasmyth_roll(tmp_path, port, sign):
    from fyst_trajectories import Trajectory, get_fyst_site

    start = Time("2026-03-15T04:00:00.000", scale="utc", precision=3)
    trajectory = Trajectory(
        times=np.array([0.0, 0.1, 0.2]),
        az=np.array([30.0, 31.0, 32.0]),
        el=np.array([40.0, 50.0, 60.0]),
        az_vel=np.full(3, 10.0),
        el_vel=np.full(3, 100.0),
        start_time=start,
    )
    timeline = SimpleNamespace(
        start_time=start, site=replace(get_fyst_site(), nasmyth_port=port)
    )
    block = SimpleNamespace(
        scan_index=0, subscan_index=0, t_start=start,
        t_stop=start + 0.3 * u.s, metadata={}, boresight_angle=123.0,
    )
    source = tmp_path / "schedule.ecsv"
    source.write_text("synthetic schedule hash input\n")
    samples = build_trajectory_table(
        "test", timeline, [(block, SimpleNamespace(trajectory=trajectory))],
        source,
    )
    roll = samples[schema.ECSV_BORESIGHT_ROLL_DEG]
    np.testing.assert_array_equal(roll.value, -sign * trajectory.el)
    assert roll.unit == u.deg and roll.dtype == np.float64
    assert samples.meta[schema.ROLL_NASMYTH_PORT_KEY] == port
