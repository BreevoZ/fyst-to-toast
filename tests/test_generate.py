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


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_requested_sample_rate(tmp_path, rate, quickstart):
    folder = tmp_path / "generation"
    patch = quickstart.example_patch("ces")
    with pytest.raises(ValueError, match="sample_rate_hz must be finite and positive"):
        generate_trajectory(
            patch, folder, start=quickstart.DEFAULT_START["ces"], duration_s=1800,
            sample_rate_hz=rate,
        )
    assert not folder.exists()


# Pong's documented night includes an initial planet calibration before science.
@pytest.mark.parametrize("case,duration_s", [("ces", 1800), ("pong", 2400)])
def test_ecsv_writer_contract(tmp_path, case, duration_s, quickstart):
    iers_path = os.environ.get("FYST_TO_TOAST_TEST_IERS")
    with iers.conf.set_temp("auto_download", False):
        table = iers.IERS_A.open(iers_path) if iers_path else iers.IERS_Auto.open()
        with (
            iers.earth_orientation_table.set(table),
            solar_system_ephemeris.set("builtin"),
        ):
            report = generate_trajectory(
                quickstart.example_patch(case),
                tmp_path, start=quickstart.DEFAULT_START[case],
                duration_s=duration_s, iers_file=iers_path,
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
        "test", timeline, [(block, SimpleNamespace(
            trajectory=trajectory, config=SimpleNamespace(timestep=0.1)
        ))],
        source,
    )
    roll = samples[schema.ECSV_BORESIGHT_ROLL_DEG]
    np.testing.assert_array_equal(roll.value, -sign * trajectory.el)
    assert roll.unit == u.deg and roll.dtype == np.float64
    assert samples.meta[schema.ROLL_NASMYTH_PORT_KEY] == port


@pytest.mark.parametrize("auto_download", [False, True])
def test_report_records_actual_iers_configuration(auto_download, quickstart):
    from fyst_trajectories.overhead import OverheadModel
    from fyst_to_toast.generate import _initial_report

    start = Time("2026-01-01", scale="utc")
    timeline = SimpleNamespace(
        science_blocks=[], blocks=[], overhead_model=OverheadModel(), end_time=start,
    )
    # Only report construction runs: enabling the setting never triggers a download.
    with iers.conf.set_temp("auto_download", auto_download):
        patch = quickstart.example_patch("ces")
        report = _initial_report(patch, "ces", timeline, start, 1.0, None)
        assert report["iers_auto_download"] is auto_download
        assert iers.conf.auto_download is auto_download


@pytest.mark.parametrize("timestep", [None, 0.2])
@pytest.mark.parametrize("rate", [None, 20.0])
def test_custom_patch_and_sampling_override(tmp_path, timestep, rate):
    """A caller-defined field keeps its identity and immutable sampling inputs."""
    from fyst_trajectories.overhead import ObservingPatch, read_timeline, schedule_to_trajectories

    params = {"spacing": 0.1, "num_terms": 4}
    if timestep is not None:
        params["timestep"] = timestep
    params["angle"] = 15.0  # Keep a key after timestep to check override ordering.
    patch = ObservingPatch(
        name="custom small Pong", ra_center=185, dec_center=-35,
        width=1.2, height=1.0, scan_type="pong", velocity=0.3,
        scan_params=params,
    )
    original = replace(patch)
    original_params = patch.scan_params
    with iers.conf.set_temp("auto_download", False), solar_system_ephemeris.set("builtin"):
        report = generate_trajectory(
            patch, tmp_path, start=Time("2026-03-15T04:00:00", scale="utc"),
            duration_s=1800, sample_rate_hz=rate,
        )
        assert report["status"] == "complete"
        timeline = read_timeline(tmp_path / "schedule.ecsv")
        pairs = schedule_to_trajectories(timeline)
    assert report["case"] == "pong"
    assert report["patch_name"] == patch.name
    assert report["scan_type"] == patch.scan_type
    expected_timestep = 1.0 / rate if rate is not None else timestep
    assert report["requested_sample_rate_hz"] == (
        None if expected_timestep is None else 1.0 / expected_timestep
    )
    samples = QTable.read(tmp_path / "fyst_trajectory.ecsv")
    assert samples.meta[schema.ECSV_CASE] == "pong"
    assert len(samples) == report["total_science_samples"] > 0
    expected_params = dict(params)
    if rate is not None:
        expected_params["timestep"] = 1.0 / rate
    for (block, planned), metadata in zip(pairs, samples.meta[schema.ECSV_BLOCKS], strict=True):
        # FYST appends its solved cycle count to the caller's ordered parameters.
        saved_params = dict(block.metadata["scan_params"])
        assert saved_params.pop("n_cycles") > 0
        assert list(saved_params.items()) == list(expected_params.items())
        assert metadata[schema.ECSV_REQUESTED_TIMESTEP_S] == planned.config.timestep
        if expected_timestep is not None:
            assert planned.config.timestep == expected_timestep
        np.testing.assert_allclose(np.diff(planned.trajectory.times), planned.config.timestep)
    assert patch == original
    assert patch.scan_params is original_params
    assert list(patch.scan_params.items()) == list(params.items())
    with pytest.raises(TypeError):
        patch.scan_params["timestep"] = 99


@pytest.mark.parametrize("value", ["pong", None, {}])
def test_generate_rejects_non_patch(tmp_path, value, quickstart):
    folder = tmp_path / "generation"
    with pytest.raises(TypeError, match="ObservingPatch"):
        generate_trajectory(
            value, folder, start=quickstart.DEFAULT_START["pong"], duration_s=1800,
        )
    assert not folder.exists()


def test_generate_rejects_unknown_scan_type(tmp_path, quickstart):
    patch = quickstart.example_patch("pong")
    # Emulate a future FYST pattern beyond the adapter's supported labels.
    object.__setattr__(patch, "scan_type", "future_pattern")
    folder = tmp_path / "generation"
    with pytest.raises(ValueError, match="Unknown scan_type: future_pattern"):
        generate_trajectory(
            patch, folder, start=quickstart.DEFAULT_START["pong"], duration_s=1800,
        )
    assert not folder.exists()
