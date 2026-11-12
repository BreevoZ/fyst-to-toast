"""Fixed-window CES export preserves the planner's samples and IO contract."""

import os

for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

from dataclasses import asdict
import json
import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.table import QTable
from astropy.time import Time
from astropy.utils import iers

pytest.importorskip("fyst_trajectories", exc_type=ImportError)

from fyst_trajectories import get_primecam_offset
from fyst_trajectories.overhead import CalibrationPolicy, read_timeline
from fyst_to_toast import trajectory_h5 as schema
from fyst_to_toast.generate import generate_window_ces, science_only_overhead, sha256

START = "2026-08-13T03:48:45.147"
PLANNER = "fyst_trajectories.planning.plan_constant_el_window_scan"
OUTPUTS = {"schedule.ecsv", "fyst_trajectory.ecsv", "report.json"}


@pytest.fixture(autouse=True)
def offline_geometry():
    with (
        iers.conf.set_temp("auto_download", False),
        iers.conf.set_temp("auto_max_age", None),
        solar_system_ephemeris.set("builtin"),
    ):
        yield


def generate(folder, **kwargs):
    inputs = dict(az_window=(50, 76), elevation=50, velocity=1, start=START, duration_s=120)
    inputs.update(kwargs)
    return generate_window_ces(folder, **inputs)


@pytest.mark.parametrize("with_offset", [False, True])
def test_window_geometry_and_contract(tmp_path, with_offset):
    offset = get_primecam_offset("i6") if with_offset else None
    report = generate(tmp_path, detector_offset=offset)
    assert {path.name for path in tmp_path.iterdir()} == OUTPUTS
    samples = QTable.read(tmp_path / "fyst_trajectory.ecsv")
    assert tuple(samples.colnames) == schema.TRAJECTORY_COLUMNS
    assert set(samples.meta) == set(schema.TRAJECTORY_META_KEYS)
    assert samples.meta[schema.ECSV_GENERATOR] == PLANNER
    assert samples.meta[schema.ECSV_CASE] == "ces_window"
    assert len(samples.meta[schema.ECSV_BLOCKS]) == 1
    metadata = samples.meta[schema.ECSV_BLOCKS][0]
    assert set(metadata) == set(schema.TRAJECTORY_BLOCK_KEYS)
    assert metadata["start_s"] == metadata["first_sample_s"] == 0.0
    assert metadata["stop_s"] == metadata["last_sample_s"]
    assert samples["timestamp_unix_s"][0].to_value(u.s) == Time(START, scale="utc").unix
    np.testing.assert_allclose(np.diff(samples["block_time_s"].value), 0.1, atol=1e-9, rtol=0)
    assert report["first_sample_utc"] == START
    assert report["last_sample_utc"] == read_timeline(tmp_path / "schedule.ecsv").end_time.utc.isot
    assert report["total_science_samples"] == len(samples)
    assert sum(report["flag_value_counts"].values()) == len(samples)
    assert report["computed_params"]["end_time_iso"] == "2026-08-13 03:50:45.147"
    assert report["computed_params"]["duration"] == pytest.approx(metadata["stop_s"])
    assert json.loads((tmp_path / "report.json").read_text()) == report
    az = samples["az_deg"].value
    el = samples["el_deg"].value
    if with_offset:
        np.testing.assert_allclose(el, 51.762172213508194, rtol=0, atol=1e-9)
        np.testing.assert_allclose([az.min(), az.max()], [49.77168162606553, 77.02168162606553], atol=1e-9, rtol=0)
        assert az.min() - 49.375 == pytest.approx(0.40, abs=0.01)
        assert report["detector_offset"] == asdict(offset)
    else:
        np.testing.assert_array_equal(el, np.full(len(samples), 50.0))
        np.testing.assert_allclose([az.min(), az.max()], [49.375, 76.625], atol=1e-9, rtol=0)
        assert report["detector_offset"] is None
    timeline = read_timeline(tmp_path / "schedule.ecsv")
    block, = timeline.science_blocks
    assert block.rising
    assert (block.scan_index, block.subscan_index, block.scan_type, block.patch_name) == (0, 0, "constant_el", "window_ces")
    assert block.az_start == az.min() and block.az_end == az.max()
    assert block.elevation == el[0]
    assert block.az_final == az[-1]
    assert timeline.overhead_model == science_only_overhead()
    assert timeline.calibration_policy == CalibrationPolicy()
    assert timeline.metadata == {
        "planner": PLANNER, "requested_az_window": [50.0, 76.0], "requested_duration_s": 120.0,
        "detector_offset": {"dx": offset.dx, "dy": offset.dy} if offset else None,
    }


def test_setting_window(tmp_path):
    generate(tmp_path, az_window=(283, 309), start="2026-09-11T06:32:37.457")
    block, = read_timeline(tmp_path / "schedule.ecsv").science_blocks
    assert not block.rising
    assert block.az_start == 282.375 and block.az_end == 309.625


def test_window_h5_to_simfyst(tmp_path):
    from toast import Data
    from toast.instrument import GroundSite, Telescope
    from toast.instrument_sim import fake_boresight_focalplane
    from toast.mpi import Comm
    from fyst_to_toast.sim_fyst import SimFYST

    report = generate(tmp_path, detector_offset=get_primecam_offset("i6"), sample_rate_hz=10)
    source = tmp_path / "fyst_trajectory.ecsv"
    samples = QTable.read(source)
    meta = QTable.read(tmp_path / "schedule.ecsv").meta
    folder = tmp_path / "hdf5"
    folder.mkdir()
    names = schema.write_trajectory_blocks(
        samples, site_name=meta["site_name"], latitude_deg=meta["site_lat"].to_value(u.deg),
        longitude_deg=meta["site_lon"].to_value(u.deg), height_m=meta["site_alt"].to_value(u.m),
        output_dir=folder, source_sha256=sha256(source),
    )
    assert len(names) == 1
    telescope = Telescope(
        "window-test", site=GroundSite(meta["site_name"], meta["site_lat"], meta["site_lon"], meta["site_alt"]),
        focalplane=fake_boresight_focalplane(n_pix=1, sample_rate=10 * u.Hz),
    )
    data = Data(Comm(world=None))
    try:
        SimFYST(telescope=telescope, trajectory_file=str(folder / names[0])).apply(data)
        assert len(data.obs) == 1
        obs = data.obs[0]
        assert obs.n_all_samples == report["total_science_samples"]
        np.testing.assert_array_equal(obs.shared["times"].data, samples["timestamp_unix_s"].value)
        info = json.loads(obs["simfyst_info"])
        assert info["rate_check"]["rtol"] == info["stored_rate_check"]["rtol"] == 1e-3
    finally:
        data.clear()


@pytest.mark.parametrize("rate", [None, 10.0, 20.0])
def test_planner_inputs_warnings_and_iers(tmp_path, monkeypatch, rate):
    from fyst_trajectories import planning
    from fyst_trajectories.exceptions import PointingWarning

    original = planning.plan_constant_el_window_scan
    calls = []

    def capture(field, **kwargs):
        calls.append((field, kwargs))
        warnings.warn("window test advisory", PointingWarning)
        return original(field, **kwargs)

    monkeypatch.setattr(planning, "plan_constant_el_window_scan", capture)
    field = planning.FieldRegion(ra_center=20, dec_center=-3, width=60, height=14)
    iers_file = tmp_path / "iers-provenance.txt"
    iers_file.write_text("Provenance only: the caller keeps the bundled IERS table.\n")
    with pytest.warns(PointingWarning, match="window test advisory"):
        report = generate(tmp_path / "output", field=field, sample_rate_hz=rate, iers_file=iers_file)
    passed_field, kwargs = calls[0]
    assert passed_field is field
    assert set(kwargs) == {"az_window", "elevation", "velocity", "duration", "site", "start_time", "az_accel", "detector_offset"} | ({"timestep"} if rate is not None else set())
    assert kwargs["start_time"].precision == 3
    if rate is not None:
        assert kwargs["timestep"] == 1 / rate
    assert report["planner_inputs"]["field"] == asdict(field)
    assert report["iers_file"] == str(iers_file.resolve())
    assert report["iers_sha256"] == sha256(iers_file)
    assert report["iers_auto_download"] is False
    assert {"category": "PointingWarning", "message": "window test advisory", "count": 1} in report["fyst_warnings"]


@pytest.mark.parametrize("name", sorted(OUTPUTS))
def test_existing_output_refused(tmp_path, name):
    source = tmp_path / name
    source.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="Outputs exist"):
        generate(tmp_path)
    assert source.read_bytes() == b"keep"
    assert len(list(tmp_path.iterdir())) == 1


def test_overwrite_preserves_other_files_and_failed_plan(tmp_path):
    report = generate(tmp_path)
    before = {name: (tmp_path / name).read_bytes() for name in OUTPUTS}
    other = tmp_path / "unrelated.h5"
    other.write_bytes(b"keep")
    with pytest.raises(ValueError, match="az_window"):
        generate(tmp_path, az_window=(76, 50), overwrite=True)
    assert before == {name: (tmp_path / name).read_bytes() for name in OUTPUTS}
    assert generate(tmp_path, overwrite=True) == report
    assert before == {name: (tmp_path / name).read_bytes() for name in OUTPUTS}
    assert other.read_bytes() == b"keep"


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_rate_creates_no_output(tmp_path, rate):
    folder = tmp_path / "output"
    with pytest.raises(ValueError, match="sample_rate_hz must be finite and positive"):
        generate(folder, sample_rate_hz=rate)
    assert not folder.exists()


@pytest.mark.parametrize("name", sorted(OUTPUTS))
def test_iers_output_collision(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"keep")
    with pytest.raises(ValueError, match="Outputs must not overwrite the input IERS file"):
        generate(tmp_path, iers_file=path, overwrite=True)
    assert path.read_bytes() == b"keep"


def test_missing_iers_creates_no_output(tmp_path):
    folder = tmp_path / "output"
    with pytest.raises(FileNotFoundError, match="IERS file not found"):
        generate(folder, iers_file=tmp_path / "missing")
    assert not folder.exists()
