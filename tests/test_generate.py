"""Optional FYST generation -> ECSV -> writer input contract."""

import os

os.environ["MPI_DISABLE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import hashlib
from pathlib import Path

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


def test_ces_ecsv_writer_contract(tmp_path):
    iers_path = os.environ.get("FYST_TO_TOAST_TEST_IERS")
    with iers.conf.set_temp("auto_download", False):
        table = iers.IERS_A.open(iers_path) if iers_path else iers.IERS_Auto.open()
        with (
            iers.earth_orientation_table.set(table),
            solar_system_ephemeris.set("builtin"),
        ):
            report = generate_trajectory(
                "ces", tmp_path, duration_s=1800, iers_file=iers_path
            )
            assert report["status"] == "complete"
            source = tmp_path / "fyst_trajectory.ecsv"
            samples = QTable.read(source)
            assert tuple(samples.colnames) == schema.TRAJECTORY_COLUMNS
            assert set(samples.meta) == set(schema.TRAJECTORY_META_KEYS)
            blocks = samples.meta[schema.ECSV_BLOCKS]
            assert len(blocks) == len(report["science_blocks"]) == 1
            for block in blocks:
                assert set(block) == set(schema.TRAJECTORY_BLOCK_KEYS)
            assert len(samples) == report["total_science_samples"] > 0

            site = QTable.read(tmp_path / "schedule.ecsv").meta
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
