"""Optional FYST generation -> ECSV -> writer input contract."""

import os

os.environ["MPI_DISABLE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import hashlib
from dataclasses import asdict, replace
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
from fyst_to_toast.generate import (
    build_trajectory_table,
    generate_observing_timeline,
    generate_trajectory,
    science_only_overhead,
)


def test_science_only_overhead_preserves_other_fyst_defaults():
    from fyst_trajectories.overhead import OverheadModel

    default = asdict(OverheadModel())
    science_only = asdict(science_only_overhead())
    calibration_fields = {
        "retune_duration", "pointing_cal_duration", "focus_duration",
        "skydip_duration", "planet_cal_duration", "beam_map_duration",
    }
    assert science_only.keys() == default.keys()
    assert {key for key in default if science_only[key] != default[key]} == calibration_fields
    for key, value in science_only.items():
        assert value == (0.0 if key in calibration_fields else default[key])


@pytest.mark.parametrize("value", ["science_only", {}, False])
def test_invalid_overhead_model_creates_no_output(tmp_path, value, quickstart):
    patch = quickstart.example_patch("ces")
    start = Time(quickstart.DEFAULT_START["ces"], scale="utc")
    with pytest.raises(TypeError, match="overhead_model must be.*OverheadModel"):
        generate_observing_timeline(patch, start, 1800, overhead_model=value)
    folder = tmp_path / "generation"
    with pytest.raises(TypeError, match="overhead_model must be.*OverheadModel"):
        generate_trajectory(
            patch, folder, start=start, duration_s=1800, overhead_model=value,
        )
    assert not folder.exists()


@pytest.mark.parametrize("selection,preset", [
    ("none", "fyst_default"), ("science_only", "science_only"),
    ("fyst_default", "fyst_default"), ("custom", "custom"),
])
def test_report_overhead_presets(tmp_path, quickstart, selection, preset):
    import json
    from fyst_trajectories.overhead import OverheadModel

    models = {
        "none": None,
        "science_only": science_only_overhead(),
        "fyst_default": OverheadModel(),
        "custom": replace(science_only_overhead(), settle_time=7.0),
    }
    model = models[selection]
    expected = OverheadModel() if model is None else model
    with iers.conf.set_temp("auto_download", False), solar_system_ephemeris.set("builtin"):
        report = generate_trajectory(
            quickstart.example_patch("ces"), tmp_path,
            start=quickstart.DEFAULT_START["ces"], duration_s=1, overhead_model=model,
        )
    assert report["status"] == "no_science"
    assert report["overhead_preset"] == preset
    assert report["overhead_model"] == asdict(expected)
    assert json.loads((tmp_path / "report.json").read_text()) == report


def test_explicit_science_only_generation(tmp_path, quickstart):
    with iers.conf.set_temp("auto_download", False), solar_system_ephemeris.set("builtin"):
        report = generate_trajectory(
            quickstart.example_patch("ces"), tmp_path,
            start=quickstart.DEFAULT_START["ces"], duration_s=1800,
            overhead_model=science_only_overhead(),
        )
    assert report["status"] == "complete"
    assert report["overhead_preset"] == "science_only"
    assert report["block_counts"].get("calibration", 0) == 0
    samples = QTable.read(tmp_path / "fyst_trajectory.ecsv")
    assert len(samples) == report["total_science_samples"] > 0


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


@pytest.mark.parametrize("timestep", [None, 0.25])
@pytest.mark.parametrize("rate", [None, 5.0])
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


@pytest.mark.parametrize("rate", [25, 488])
def test_scheduler_rejects_rate_above_go_tcs_limit(tmp_path, quickstart, rate):
    patch = quickstart.example_patch("ces")
    with pytest.raises(ValueError, match=r"Go TCS /path minimum of 0\.05 s: native CES"):
        generate_trajectory(
            patch, tmp_path, start=quickstart.DEFAULT_START["ces"], duration_s=1800,
            sample_rate_hz=rate,
        )


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


@pytest.fixture
def export_patch(tmp_path, quickstart):
    def run(case, duration_s, *, offset=None, sample_rate_hz=None):
        patch = quickstart.example_patch(case)
        if offset is not None:
            patch = replace(patch, detector_offset=offset)
        report = generate_trajectory(
            patch, tmp_path, start=quickstart.DEFAULT_START[case], duration_s=duration_s,
            sample_rate_hz=sample_rate_hz, overhead_model=science_only_overhead(),
        )
        assert report["status"] == "complete"
        return patch, QTable.read(tmp_path / "fyst_trajectory.ecsv"), report

    with iers.conf.set_temp("auto_download", False), solar_system_ephemeris.set("builtin"):
        yield run


@pytest.mark.parametrize("filename", ["full_timeline.ecsv", "schedule.ecsv"])
def test_crossing_ces_replays_exported_samples(tmp_path, export_patch, filename):
    from fyst_trajectories import get_absolute_times
    from fyst_trajectories.overhead import read_timeline, schedule_to_trajectories

    patch, samples, report = export_patch("ces", 1800, sample_rate_hz=5)
    timeline = read_timeline(tmp_path / filename)
    block, = timeline.science_blocks
    assert block.metadata["scan_params"] == {}
    window = block.metadata["window_scan"]
    assert window["az_window"][0] < window["az_window"][1]
    assert window["az_accel"] == patch.scan_params["az_accel"]
    exported_block, = samples.meta["blocks"]
    assert exported_block["requested_timestep_s"] == window["timestep"] == 0.2
    assert report["requested_sample_rate_hz"] == 5

    (_, scan), = schedule_to_trajectories(timeline)
    assert window["az_window"] == [scan.config.az_start, scan.config.az_stop]
    traj = scan.trajectory
    absolute = get_absolute_times(traj)
    np.testing.assert_allclose(np.diff(traj.times), 0.2, rtol=1e-3, atol=0)
    np.testing.assert_array_equal(
        samples["time_s"].to_value(u.s), (absolute - timeline.start_time).sec,
    )
    np.testing.assert_array_equal(samples["timestamp_unix_s"].to_value(u.s), absolute.unix)
    np.testing.assert_array_equal(samples["block_time_s"].to_value(u.s), traj.times)
    np.testing.assert_array_equal(samples["az_deg"].to_value(u.deg), traj.az)
    np.testing.assert_array_equal(samples["el_deg"].to_value(u.deg), traj.el)
    np.testing.assert_array_equal(samples["az_velocity"].to_value(u.deg / u.s), traj.az_vel)
    np.testing.assert_array_equal(samples["el_velocity"].to_value(u.deg / u.s), traj.el_vel)
    np.testing.assert_array_equal(samples["scan_flag"], traj.scan_flag)
    np.testing.assert_array_equal(samples["scan_index"], block.scan_index)
    np.testing.assert_array_equal(samples["subscan_index"], block.subscan_index)
    np.testing.assert_array_equal(
        samples["boresight_roll_deg"].to_value(u.deg), -timeline.site.nasmyth_sign * traj.el,
    )


@pytest.mark.parametrize("case,duration_s", [("pong", 1500), ("daisy", 600)])
def test_science_el_final_matches_last_sample(tmp_path, export_patch, case, duration_s):
    _, samples, _ = export_patch(case, duration_s)
    assert samples["el_deg"][0] != samples["el_deg"][-1]
    for filename in ("full_timeline.ecsv", "schedule.ecsv"):
        table = QTable.read(tmp_path / filename)
        row, = table[table["block_type"] == "science"]
        assert row["el_final"] == samples["el_deg"][-1]


def test_scheduler_im1_boresight_to_h5(tmp_path, export_patch):
    from fyst_trajectories import (
        boresight_to_detector,
        compute_focal_plane_rotation,
        get_primecam_offset,
    )
    from fyst_trajectories.overhead import read_timeline

    offset = get_primecam_offset("IM1")
    patch, samples, _ = export_patch("ces", 1800, offset=offset)
    timeline = read_timeline(tmp_path / "schedule.ecsv")
    block, = timeline.science_blocks
    assert block.metadata["detector_offset"] == asdict(offset)
    assert block.metadata["detector_elevation"] == patch.elevation
    az = samples["az_deg"].to_value(u.deg)
    el = samples["el_deg"].to_value(u.deg)
    assert block.elevation != patch.elevation
    np.testing.assert_array_equal(el, block.elevation)
    rotation = compute_focal_plane_rotation(el, site=timeline.site, offset=offset)
    _, detector_el = boresight_to_detector(az, el, offset, rotation)
    np.testing.assert_allclose(detector_el, patch.elevation, atol=1e-9, rtol=0)

    site = timeline.site
    folder = tmp_path / "hdf5"
    folder.mkdir()
    name, = schema.write_trajectory_blocks(
        samples, site_name=site.name, latitude_deg=site.latitude, longitude_deg=site.longitude,
        height_m=site.elevation, output_dir=folder,
        source_sha256=hashlib.sha256((tmp_path / "fyst_trajectory.ecsv").read_bytes()).hexdigest(),
    )
    loaded = schema._load_trajectory_block(folder / name)
    np.testing.assert_array_equal(loaded.arrays["azimuth"], samples["az_deg"].to_value(u.rad))
    np.testing.assert_array_equal(loaded.arrays["elevation"], samples["el_deg"].to_value(u.rad))
