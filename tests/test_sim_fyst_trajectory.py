"""B2.2/B2.3: native geometry, bytes and motion classification on fixture inputs."""

import os

for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import json
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.utils import iers
from toast import Data
from toast import qarray as qa
from toast.coordinates import azel_to_radec
from toast.instrument import Focalplane, GroundSite, Telescope
from toast.instrument_sim import fake_boresight_focalplane
from toast.mpi import Comm
from toast.observation import default_values as defaults
from toast.schedule import GroundSchedule
from toast.utils import name_UID

from fyst_to_toast import sim_fyst
from fyst_to_toast import trajectory_h5 as schema
from fyst_to_toast.fyst_flags import (
    SCAN_FLAG_RETUNE,
    SCAN_FLAG_SCIENCE,
    SCAN_FLAG_TURNAROUND,
    SCAN_FLAG_UNCLASSIFIED,
)
from fyst_to_toast.sim_fyst import SimFYST, _mask_to_sample_spans
from fyst_to_toast.trajectory_h5 import _TRAJECTORY_FIELDS, _load_trajectory_block

iers.conf.auto_download = False
SAMPLES = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(params=["fyst_ces_000_001.h5", "fyst_daisy_003_000.h5"])
def source(request, monkeypatch):
    path = SAMPLES / request.param
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    def no_network(*args, **kwargs):
        raise AssertionError("Network is forbidden in trajectory tests")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    with h5py.File(path, "r") as saved:
        # This oracle never uses the adapter loader to obtain expected arrays/site.
        arrays = {key: saved[f"shared/{key}"][()] for key in _TRAJECTORY_FIELDS}
        metadata = dict(saved["metadata/other"].attrs)
        site = dict(saved["instrument"].attrs)
        name = saved.attrs["observation_name"]
    with solar_system_ephemeris.set("builtin"):
        yield SimpleNamespace(
            path=path, arrays=arrays, metadata=metadata, site=site, name=name
        )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def caller(source, n_pix=1, rate=10, height_offset=0):
    full = fake_boresight_focalplane(n_pix=n_pix, sample_rate=rate * u.Hz)
    table = full.detector_data.copy()
    table["name"] = [f"caller-{i}" for i in range(len(table))]
    fp = Focalplane(
        detector_data=table, sample_rate=rate * u.Hz, field_of_view=full.field_of_view
    )
    site = GroundSite(
        "caller-site",
        source.site["site_lat_deg"] * u.deg,
        source.site["site_lon_deg"] * u.deg,
        (source.site["site_alt_m"] + height_offset) * u.m,
    )
    return Telescope("caller-telescope", focalplane=fp, site=site)


@pytest.fixture
def data():
    value = Data(Comm(world=None))
    yield value
    value.clear()


def quaternion_equal(actual, expected):
    assert np.isfinite(actual).all()
    assert np.max(np.abs(np.linalg.norm(actual, axis=1) - 1)) < 1e-12
    distance = np.minimum(
        np.linalg.norm(actual - expected, axis=1),
        np.linalg.norm(actual + expected, axis=1),
    )
    assert np.max(distance) < 1e-12


@pytest.mark.parametrize("n_pix,roll", [(1, None), (3, 10 * u.deg)])
def test_geometry_and_caller_instrument(source, data, monkeypatch, n_pix, roll):
    spies = []
    for module, name in (
        (sim_fyst, "simulate_ces_scan"),
        (sim_fyst, "simulate_elnod"),
        (GroundSchedule, "read"),
    ):
        spy = Mock(side_effect=AssertionError(f"{name} must never run"))
        monkeypatch.setattr(module, name, spy)
        spies.append(spy)
    telescope = caller(source, n_pix=n_pix)
    detector_table = telescope.focalplane.detector_data.copy()
    op = SimFYST(
        telescope=telescope,
        trajectory_file=str(source.path),
        trajectory_boresight_angle=roll,
    )
    op.apply(data)
    for spy in spies:
        spy.assert_not_called()
    assert len(data.obs) == 1
    obs = data.obs[0]
    t, az, el = (source.arrays[key] for key in ("times", "azimuth", "elevation"))
    assert obs.n_local_samples == len(t)
    assert obs.name == source.name and obs.uid == name_UID(source.name)
    assert obs.session.name == source.name and obs.session.uid == name_UID(source.name)
    assert obs.session.start == datetime.fromtimestamp(t[0], tz=timezone.utc)
    assert obs.session.end == datetime.fromtimestamp(t[-1], tz=timezone.utc)
    assert obs.telescope is telescope
    assert obs.telescope.site.name == "caller-site"
    assert obs.telescope.name == "caller-telescope"
    assert obs.local_detectors == [f"caller-{i}" for i in range(2 * n_pix)]
    assert obs.local_detectors == telescope.focalplane.detectors
    for key in detector_table.colnames:
        np.testing.assert_array_equal(
            telescope.focalplane.detector_data[key], detector_table[key]
        )
    assert telescope.focalplane.sample_rate == 10 * u.Hz
    for key, expected in source.arrays.items():
        actual = obs.shared[key].data
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, key
        assert actual.tobytes() == expected.tobytes(), key
    for key, dtype, units in (
        (op.det_data, np.float64, u.K),
        (op.det_flags, np.uint8, u.dimensionless_unscaled),
    ):
        actual = obs.detdata[key]
        assert actual.shape == (2 * n_pix, len(t))
        assert actual.dtype == dtype and actual.units == units
        assert np.count_nonzero(actual.data) == 0

    source_info = json.loads(source.metadata["trajectory_info"])
    roll_deg = source_info["roll_deg"] if roll is None else roll.to_value(u.deg)
    expected_bore = qa.from_lonlat_angles(-az, el, np.zeros_like(el))
    expected_bore = qa.mult(
        expected_bore, qa.rotation(np.array([0.0, 0.0, 1.0]), np.deg2rad(roll_deg))
    )
    quaternion_equal(obs.shared[op.boresight_azel].data, expected_bore)
    lon, lat, _ = qa.to_lonlat_angles(obs.shared[op.boresight_azel].data)
    delta_az = np.arctan2(np.sin(-lon - az), np.cos(-lon - az))
    assert np.max(np.abs(delta_az)) < 1e-12
    np.testing.assert_allclose(lat, el, atol=1e-12, rtol=0)
    expected_radec = azel_to_radec(
        telescope.site,
        t,
        expected_bore,
        use_ephem=True,
        use_qpoint=False,
        so3g_compat_mode=False,
    )
    quaternion_equal(obs.shared[op.boresight_radec].data, expected_radec)
    pos, vel = telescope.site.position_velocity(t)
    np.testing.assert_array_equal(obs.shared[op.position].data, pos)
    np.testing.assert_array_equal(obs.shared[op.velocity].data, vel)

    assert obs["trajectory_info"] == source.metadata["trajectory_info"]
    assert json.loads(obs["source_shared_array_sha256"]) == json.loads(
        source.metadata["shared_array_sha256"]
    )
    assert "shared_array_sha256" not in obs
    info = json.loads(obs["simfyst_info"])
    assert info["contract_version"] == "RL-A"
    assert info["source_file"] == source.path.name
    assert info["source_sha256"] == hashlib.sha256(source.path.read_bytes()).hexdigest()
    assert info["samples"] == len(t)
    assert info["first_time_unix_s"] == t[0] and info["last_time_unix_s"] == t[-1]
    actual_rate = (len(t) - 1) / (t[-1] - t[0])
    assert info["rate_check"] == {
        "nominal_rate_hz": 10.0,
        "actual_rate_hz": actual_rate,
        "relative_error": abs(10 - actual_rate) / actual_rate,
        "rtol": 1e-6,
    }
    assert info["roll_deg"] == roll_deg
    assert info[schema.ROLL_SOURCE_KEY] == (
        schema.ROLL_SOURCE_LEGACY_SCALAR
        if roll is None
        else schema.ROLL_SOURCE_OVERRIDE
    )
    assert info[schema.ROLL_CONVENTION_KEY] == schema.ROLL_CONVENTION
    np.testing.assert_array_equal(
        obs.shared[schema.H5_BORESIGHT_ROLL].data, np.deg2rad(roll_deg)
    )
    assert (
        info["caller_telescope"] == telescope.name
        and info["caller_site"] == telescope.site.name
    )
    assert info["coordinate_backend"] == {
        "use_ephem": True,
        "use_qpoint": False,
        "so3g_compat_mode": False,
    }
    assert info["max_phase_residual_s"] <= info["time_tolerance_s"]
    assert_motion_classification(obs, op, source.arrays["fyst_scan_flag"])
    assert set(op.provides()["shared"]) == set(obs.shared)
    assert set(op.provides()["detdata"]) == set(obs.detdata)
    assert set(op.provides()["intervals"]) == {
        op.scanning_interval,
        op.turnaround_interval,
        op.fyst_invalid_interval,
    }


UNSUPPORTED = [
    ("schedule", GroundSchedule()),
    ("schedule_file", "unused.ecsv"),
    ("telescope_file", "unused.h5"),
    ("distribute_time", True),
    ("detset_key", "band"),
    ("session_split_key", "band"),
    ("weather", "atacama"),
    ("max_pwv", 1 * u.mm),
    ("median_weather", True),
    ("hwp_step_time", 1 * u.s),
    ("elnod_start", True),
    ("elnod_end", True),
    ("elnod_every_scan", True),
    ("elnods", [1.0]),
    ("el_mod_step", 1 * u.deg),
    ("el_mod_rate", 1 * u.Hz),
    ("el_mod_amplitude", 2 * u.deg),
    ("el_mod_sine", True),
    ("el_mod_sine_phase", 1 * u.deg),
    ("randomize_phase", True),
    ("track_azimuth", True),
    ("scan_cosecant_modulation", True),
    ("fix_rate_on_sky", False),
    ("scan_rate_az", 2 * u.deg / u.s),
    ("scan_rate_el", 2 * u.deg / u.s),
    ("scan_accel_az", 2 * u.deg / u.s**2),
    ("scan_accel_el", 2 * u.deg / u.s**2),
    ("times", "fyst_time_s"),
    ("times", schema.H5_BORESIGHT_ROLL),
    ("position", "azimuth"),
    ("velocity", "position"),
    ("det_data", "flags"),
    ("trajectory_boresight_angle", np.nan * u.deg),
    ("trajectory_boresight_angle", np.inf * u.deg),
    ("shared_flags", None),
    ("shared_flags", "times"),
    ("shared_flags", schema.H5_BORESIGHT_ROLL),
    ("shared_flags", "boresight_azel"),
    ("scanning_interval", ""),
    ("turnaround_interval", "scanning"),
    ("fyst_invalid_interval", "ALL_OBSERVATION_SAMPLES"),
    ("turnaround_mask", -1),
    ("turnaround_mask", 256),
]


@pytest.mark.parametrize(
    "key,value", UNSUPPORTED, ids=[item[0] for item in UNSUPPORTED]
)
def test_unsupported_settings(source, data, key, value):
    match = {
        "times": "original fields",
        "position": "distinct",
        "velocity": "distinct",
        "det_data": "distinct",
        "trajectory_boresight_angle": "finite",
        "scanning_interval": "interval names",
        "turnaround_interval": "interval names",
        "fyst_invalid_interval": "interval names",
    }.get(key, key)
    with pytest.raises(ValueError, match=match):
        SimFYST(
            telescope=caller(source), trajectory_file=str(source.path), **{key: value}
        ).apply(data)
    assert data.obs == []


@pytest.mark.parametrize(
    "settings",
    [
        {"hwp_angle": "hwp", "hwp_rpm": 1.0},
        {"hwp_angle": "hwp", "hwp_step": 1 * u.deg, "hwp_step_time": 1 * u.s},
    ],
)
def test_hwp_settings(source, data, settings):
    with pytest.raises(ValueError, match="hwp_"):
        SimFYST(
            telescope=caller(source), trajectory_file=str(source.path), **settings
        ).apply(data)
    assert data.obs == []


@pytest.mark.parametrize(
    "mode", ["missing", "empty", "site", "weather", "rate", "detectors", "data"]
)
def test_caller_rejections(source, data, mode):
    telescope = caller(
        source,
        rate=20 if mode == "rate" else 10,
        height_offset=0.002 if mode == "site" else 0,
    )
    kwargs = {}
    match = {
        "missing": "caller.*Telescope",
        "empty": "non-empty.*focalplane",
        "site": "1 mm",
        "weather": "site weather",
        "rate": "Nominal sample rate 20 Hz does not match actual",
        "detectors": "filtering",
        "data": "empty data.obs",
    }[mode]
    if mode == "missing":
        telescope = None
    elif mode == "empty":
        telescope.focalplane = Focalplane(
            detector_data=telescope.focalplane.detector_data[:0],
            sample_rate=10 * u.Hz,
            field_of_view=1 * u.deg,
        )
    elif mode == "weather":
        telescope.site.weather = object()
    elif mode == "detectors":
        kwargs["detectors"] = []
    elif mode == "data":
        # Preserve pre-existing caller content instead of clearing it on rejection.
        data.obs.append(object())
    before = list(data.obs)
    try:
        with pytest.raises(ValueError, match=match):
            SimFYST(telescope=telescope, trajectory_file=str(source.path)).apply(
                data, **kwargs
            )
        assert data.obs == before
    finally:
        data.obs.clear()


@pytest.mark.parametrize(
    "key,value",
    [
        ("world_size", 2),
        ("group_size", 2),
        ("ngroups", 2),
        ("comm_world", object()),
        ("comm_group", object()),
    ],
)
def test_serial_rejection(source, key, value):
    comm = {
        "world_size": 1,
        "group_size": 1,
        "ngroups": 1,
        "comm_world": None,
        "comm_group": None,
    }
    comm[key] = value
    data = SimpleNamespace(comm=SimpleNamespace(**comm), obs=[])
    with pytest.raises(ValueError, match="serial.*MPI"):
        SimFYST(telescope=caller(source), trajectory_file=str(source.path))._exec(data)
    assert data.obs == []


@pytest.mark.parametrize("roll", [None, float("nan"), float("inf"), "invalid"])
def test_bad_source_roll(source, data, tmp_path, roll):
    path = tmp_path / source.path.name
    shutil.copyfile(source.path, path)
    with h5py.File(path, "r+") as saved:
        metadata = saved["metadata/other"].attrs
        info = json.loads(metadata["trajectory_info"])
        if roll is None:
            del info["roll_deg"]
        else:
            info["roll_deg"] = roll
        metadata["trajectory_info"] = json.dumps(info)
    # The loader now owns legacy scalar validation, including before an override.
    for override in (None, 10 * u.deg):
        op = SimFYST(
            telescope=caller(source),
            trajectory_file=str(path),
            trajectory_boresight_angle=override,
        )
        with pytest.raises(ValueError, match="roll_deg.*finite"):
            op.apply(data)
        assert data.obs == []


@pytest.mark.parametrize("override", [None, 0 * u.deg, 11 * u.deg])
def test_per_sample_roll_and_override(source, data, tmp_path, override):
    path = tmp_path / source.path.name
    shutil.copyfile(source.path, path)
    roll = np.linspace(-0.3, 0.5, len(source.arrays["times"]))
    with h5py.File(path, "r+") as saved:
        ds = saved["shared"].create_dataset(schema.H5_BORESIGHT_ROLL, data=roll)
        ds.attrs["comm_type"] = "column"
        metadata = saved["metadata/other"].attrs
        info = json.loads(metadata["trajectory_info"])
        info[schema.ROLL_SOURCE_KEY] = "test_ramp"
        info[schema.ROLL_CONVENTION_KEY] = schema.ROLL_CONVENTION
        metadata["trajectory_info"] = json.dumps(info)
        manifest = json.loads(metadata["shared_array_sha256"])
        manifest[schema.H5_BORESIGHT_ROLL] = hashlib.sha256(roll.tobytes()).hexdigest()
        metadata["shared_array_sha256"] = json.dumps(manifest)
    loaded = _load_trajectory_block(path)
    assert loaded.arrays[schema.H5_BORESIGHT_ROLL].tobytes() == roll.tobytes()
    op = SimFYST(
        telescope=caller(source),
        trajectory_file=str(path),
        trajectory_boresight_angle=override,
    )
    op.apply(data)
    obs = data.obs[0]
    effective = (
        roll if override is None else np.full(len(roll), override.to_value(u.rad))
    )
    np.testing.assert_array_equal(obs.shared[schema.H5_BORESIGHT_ROLL].data, effective)
    q0 = qa.from_lonlat_angles(
        -source.arrays["azimuth"], source.arrays["elevation"], np.zeros(len(roll))
    )
    expected = np.array(
        [qa.mult(q, qa.rotation([0.0, 0.0, 1.0], r)) for q, r in zip(q0, effective)]
    )
    np.testing.assert_allclose(
        obs.shared[op.boresight_azel].data, expected, rtol=0, atol=1e-15
    )
    if override is not None and override.value == 0:
        assert obs.shared[op.boresight_azel].data.tobytes() == q0.tobytes()
    for key, values in source.arrays.items():
        assert obs.shared[key].data.tobytes() == values.tobytes()
    assert obs["trajectory_info"] == loaded.trajectory_info_json
    info = json.loads(obs["simfyst_info"])
    assert info[schema.ROLL_SOURCE_KEY] == (
        "test_ramp" if override is None else schema.ROLL_SOURCE_OVERRIDE
    )
    assert info[schema.ROLL_CONVENTION_KEY] == schema.ROLL_CONVENTION
    if override is None:
        assert info["roll_deg"] is None
    assert_motion_classification(obs, op, source.arrays["fyst_scan_flag"])


@pytest.mark.parametrize(
    "det_data,det_flags", [(None, None), (None, "quality"), ("signal", None)]
)
def test_optional_detdata_aliases_and_site_tolerance(source, data, det_data, det_flags):
    op = SimFYST(
        telescope=caller(source, height_offset=0.0005),
        trajectory_file=str(source.path),
        det_data=det_data,
        det_flags=det_flags,
        det_data_units=u.mK,
        times="clock",
        azimuth="az",
        elevation="el",
        boresight_azel="horizontal",
        boresight_radec="equatorial",
        position="pos",
        velocity="vel",
    )
    op.apply(data)
    obs = data.obs[0]
    assert set(obs.detdata) == {key for key in (det_data, det_flags) if key is not None}
    assert set(op.provides()["detdata"]) == set(obs.detdata)
    if det_data is not None:
        assert obs.detdata[det_data].units == u.mK
        assert np.count_nonzero(obs.detdata[det_data].data) == 0
    assert set(op.provides()["shared"]) == set(obs.shared)
    for alias, native in (("clock", "times"), ("az", "azimuth"), ("el", "elevation")):
        assert obs.shared[alias].data.tobytes() == source.arrays[native].tobytes()
        assert obs.shared[native].data.tobytes() == source.arrays[native].tobytes()


def test_allocation_failure_cleans_unappended_observation(source, data, monkeypatch):
    original = sim_fyst.Observation
    created = []

    def failing_observation(*args, **kwargs):
        obs = original(*args, **kwargs)
        created.append(obs)
        monkeypatch.setattr(obs, "clear", Mock(wraps=obs.clear))
        monkeypatch.setattr(
            obs.detdata, "ensure", Mock(side_effect=RuntimeError("allocation failure"))
        )
        return obs

    monkeypatch.setattr(sim_fyst, "Observation", failing_observation)
    with pytest.raises(RuntimeError, match="allocation failure"):
        SimFYST(telescope=caller(source), trajectory_file=str(source.path)).apply(data)
    assert data.obs == [] and len(created) == 1
    created[0].clear.assert_called_once()


def test_missing_ground_site(source, data):
    telescope = caller(source)
    telescope.site = None
    with pytest.raises(TypeError, match="caller GroundSite"):
        SimFYST(telescope=telescope, trajectory_file=str(source.path)).apply(data)
    assert data.obs == []


def test_coordinate_backend_options(source, data, monkeypatch):
    native = Mock(wraps=azel_to_radec)
    monkeypatch.setattr(sim_fyst, "azel_to_radec", native)
    telescope = caller(source)
    op = SimFYST(telescope=telescope, trajectory_file=str(source.path), use_ephem=False)
    op.apply(data)
    expected_options = {
        "use_ephem": False,
        "use_qpoint": False,
        "so3g_compat_mode": False,
    }
    assert native.call_args.kwargs == expected_options
    obs = data.obs[0]
    expected = azel_to_radec(
        telescope.site,
        source.arrays["times"],
        obs.shared[op.boresight_azel].data,
        **expected_options,
    )
    quaternion_equal(obs.shared[op.boresight_radec].data, expected)
    assert json.loads(obs["simfyst_info"])["coordinate_backend"] == expected_options


def assert_motion_classification(obs, op, raw):
    """Independent per-sample oracle; validate intervals and native views separately."""
    expected_flags = np.zeros(len(raw), dtype=np.uint8)
    expected_flags[raw == SCAN_FLAG_TURNAROUND] = op.turnaround_mask
    expected_flags[np.isin(raw, [SCAN_FLAG_UNCLASSIFIED, SCAN_FLAG_RETUNE])] = (
        defaults.shared_mask_invalid
    )
    flags = obs.shared[op.shared_flags].data
    assert flags.dtype == np.uint8
    np.testing.assert_array_equal(flags, expected_flags)
    masks = {
        "science": (op.scanning_interval, raw == SCAN_FLAG_SCIENCE),
        "turnaround": (op.turnaround_interval, raw == SCAN_FLAG_TURNAROUND),
        "invalid": (
            op.fyst_invalid_interval,
            np.isin(raw, [SCAN_FLAG_UNCLASSIFIED, SCAN_FLAG_RETUNE]),
        ),
    }
    coverage = np.zeros(len(raw), dtype=int)
    for key, expected in masks.values():
        reconstructed = np.zeros(len(raw), dtype=bool)
        for interval in obs.intervals[key]:
            assert 0 <= interval.first < interval.last <= len(raw)
            reconstructed[interval.first : interval.last] = True
        np.testing.assert_array_equal(reconstructed, expected)
        coverage += reconstructed
        view_mask = np.zeros(len(raw), dtype=bool)
        for span in obs.view[key].slices:
            view_mask[span] = True
        np.testing.assert_array_equal(view_mask, expected)
        viewed_times = obs.view[key].shared[op.times]
        np.testing.assert_array_equal(
            np.concatenate(viewed_times) if viewed_times else np.array([]),
            obs.shared[op.times].data[expected],
        )
    np.testing.assert_array_equal(coverage, np.ones(len(raw), dtype=int))
    assert set(obs.intervals) == {
        "ALL_OBSERVATION_SAMPLES",
        *(key for key, _ in masks.values()),
    }
    info = json.loads(obs["simfyst_info"])
    assert (
        info["flags/intervals"]
        == "motion derived (B2.3); solar/directions not yet derived"
    )
    assert info["flag_mapping"] == {
        str(SCAN_FLAG_SCIENCE): 0,
        str(SCAN_FLAG_TURNAROUND): op.turnaround_mask,
        str(SCAN_FLAG_UNCLASSIFIED): defaults.shared_mask_invalid,
        str(SCAN_FLAG_RETUNE): defaults.shared_mask_invalid,
    }
    assert info["class_counts"] == {
        key: int(np.count_nonzero(mask)) for key, (_, mask) in masks.items()
    }


@pytest.mark.parametrize(
    "mask,spans",
    [
        ([True] * 6, [(0, 6)]),
        ([False] * 6, []),
        ([True, False, False], [(0, 1)]),
        ([False, False, True], [(2, 3)]),
        ([False, True, False], [(1, 2)]),
        ([True, True, False, True, True], [(0, 2), (3, 5)]),
    ],
)
def test_mask_to_sample_spans(mask, spans):
    assert _mask_to_sample_spans(np.array(mask, dtype=bool)) == spans


def test_adjacent_class_runs():
    raw = np.array(
        [
            SCAN_FLAG_SCIENCE,
            SCAN_FLAG_SCIENCE,
            SCAN_FLAG_TURNAROUND,
            SCAN_FLAG_UNCLASSIFIED,
            SCAN_FLAG_RETUNE,
        ]
    )
    assert _mask_to_sample_spans(raw == SCAN_FLAG_SCIENCE) == [(0, 2)]
    assert _mask_to_sample_spans(raw == SCAN_FLAG_TURNAROUND) == [(2, 3)]
    assert _mask_to_sample_spans(
        np.isin(raw, [SCAN_FLAG_UNCLASSIFIED, SCAN_FLAG_RETUNE])
    ) == [(3, 5)]


def test_fixture_motion_spans(source, data):
    op = SimFYST(telescope=caller(source), trajectory_file=str(source.path))
    op.apply(data)
    obs = data.obs[0]
    assert_motion_classification(obs, op, source.arrays["fyst_scan_flag"])
    expected = {
        "fyst_ces_000_001.h5": {
            "scanning": [
                (0, 477),
                (497, 1175),
                (1195, 1873),
                (1893, 2571),
                (2591, 2611),
            ],
            "turnaround": [(477, 497), (1175, 1195), (1873, 1893), (2571, 2591)],
        },
        "fyst_daisy_003_000.h5": {"scanning": [(5, 399)], "turnaround": [(0, 5)]},
    }[source.path.name]
    for key, spans in expected.items():
        actual = [(int(iv.first), int(iv.last)) for iv in obs.intervals[key]]
        assert actual == spans
        print(f"{source.path.name} {key}: {actual}")
    assert len(obs.intervals[op.fyst_invalid_interval]) == 0


def test_synthetic_invalid_endpoints(source, data, tmp_path):
    path = tmp_path / source.path.name
    shutil.copyfile(source.path, path)
    raw = source.arrays["fyst_scan_flag"].copy()
    raw[0], raw[-1] = SCAN_FLAG_UNCLASSIFIED, SCAN_FLAG_RETUNE
    with h5py.File(path, "r+") as saved:
        saved["shared/fyst_scan_flag"][:] = raw
        metadata = saved["metadata/other"].attrs
        manifest = json.loads(metadata["shared_array_sha256"])
        manifest["fyst_scan_flag"] = hashlib.sha256(raw.tobytes()).hexdigest()
        metadata["shared_array_sha256"] = json.dumps(manifest)
    op = SimFYST(telescope=caller(source), trajectory_file=str(path))
    op.apply(data)
    obs = data.obs[0]
    assert_motion_classification(obs, op, raw)
    assert [(iv.first, iv.last) for iv in obs.intervals[op.fyst_invalid_interval]] == [
        (0, 1),
        (len(raw) - 1, len(raw)),
    ]
    assert obs.shared[op.shared_flags].data[0] == defaults.shared_mask_invalid
    assert obs.shared[op.shared_flags].data[-1] == defaults.shared_mask_invalid
    assert obs.shared["fyst_scan_flag"].data.tobytes() == raw.tobytes()
    assert json.loads(obs["simfyst_info"])["class_counts"]["invalid"] == 2


def test_custom_motion_output_names_and_mask(source, data):
    op = SimFYST(
        telescope=caller(source),
        trajectory_file=str(source.path),
        shared_flags="motion_flags",
        scanning_interval="science_samples",
        turnaround_interval="turn_samples",
        fyst_invalid_interval="invalid_samples",
        turnaround_mask=64,
    )
    op.apply(data)
    obs = data.obs[0]
    assert_motion_classification(obs, op, source.arrays["fyst_scan_flag"])
    assert "flags" not in obs.shared
    assert set(op.provides()["shared"]) == set(obs.shared)
    assert set(op.provides()["intervals"]) == set(obs.intervals) - {
        "ALL_OBSERVATION_SAMPLES"
    }
