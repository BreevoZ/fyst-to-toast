"""Committed samples always run; full writer outputs add optional coverage."""

import os

# Set these before NumPy / TOAST imports, including in direct pytest collection.
os.environ["MPI_DISABLE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import hashlib
import json
import shutil
import socket
from itertools import pairwise
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.table import QTable
from astropy.utils import iers

from fyst_to_toast.trajectory_h5 import (
    _check_sample_rate,
    _load_trajectory_block,
    write_trajectory_blocks,
)

iers.conf.auto_download = False

REPO = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPO / "data"
# This override only controls optional writer outputs, never committed samples.
INPUT_ROOT = Path(
    os.environ.get("FYST_TO_TOAST_TEST_OUTPUT", REPO / "output")
).resolve()
CES_FILES = [f"ces/hdf5/fyst_ces_000_{i:03d}.h5" for i in range(3)]
DAISY_FILES = [f"daisy/hdf5/fyst_daisy_{i:03d}_000.h5" for i in range(4)]
SAMPLE_EXPECTATIONS = {
    "fyst_ces_000_001.h5": {
        "samples": 2611,
        "flags": {1: 2531, 2: 80},
        "rate_hz": 9.999997341770571,
    },
    "fyst_daisy_003_000.h5": {
        "samples": 399,
        "flags": {1: 394, 2: 5},
        "rate_hz": 10.000000011980832,
    },
}
FIELDS = (
    "times",
    "azimuth",
    "elevation",
    "fyst_time_s",
    "fyst_block_time_s",
    "fyst_az_deg",
    "fyst_el_deg",
    "fyst_az_velocity_deg_s",
    "fyst_el_velocity_deg_s",
    "fyst_scan_flag",
    "scan_index",
    "subscan_index",
)


def input_path(relative):
    """Only the optional full-output tests may skip for missing input."""
    path = INPUT_ROOT / relative
    if not path.is_file():
        pytest.skip(
            f"Writer H5 absent: {path}; set FYST_TO_TOAST_TEST_OUTPUT to supply it"
        )
    return path


@pytest.fixture(scope="session", autouse=True)
def protect_inputs_and_disable_network():
    paths = [SAMPLE_ROOT / name for name in SAMPLE_EXPECTATIONS]
    # Protect local outputs too, even while the override points at an empty dir.
    for root in (REPO / "output", INPUT_ROOT):
        paths.extend(root / p for p in CES_FILES + DAISY_FILES)
    before = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()
    }

    def no_network(*args, **kwargs):
        raise AssertionError("Network access is forbidden in trajectory input tests")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", no_network)
        patch.setattr(socket.socket, "connect_ex", no_network)
        patch.setattr(socket, "create_connection", no_network)
        yield
    after = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before}
    assert after == before, "Original input H5 bytes changed during the test session"
    print(f"Input SHA-256 unchanged ({len(before)} files): {json.dumps(after)}")


@pytest.fixture
def ces_path():
    path = SAMPLE_ROOT / "fyst_ces_000_001.h5"
    assert path.is_file(), f"Missing committed sample: {path}"
    return path


@pytest.fixture(params=list(SAMPLE_EXPECTATIONS))
def sample_path(request):
    path = SAMPLE_ROOT / request.param
    assert path.is_file(), f"Missing committed sample: {path}"
    return path


def refresh_manifest(saved):
    """Let semantic checks, not a stale hash, reject deliberately edited copies."""
    manifest = {
        key: hashlib.sha256(ds[()].tobytes()).hexdigest()
        for key, ds in saved["shared"].items()
    }
    saved["metadata/other"].attrs["shared_array_sha256"] = json.dumps(manifest)


def assert_original_arrays_and_metadata(path):
    """h5py is the independent reference for native LoadHDF5 output."""
    block = _load_trajectory_block(path)
    with h5py.File(path, "r") as saved:
        assert set(block.arrays) == set(FIELDS)
        for key in FIELDS:
            expected = saved[f"shared/{key}"][()]
            actual = block.arrays[key]
            assert actual.dtype == expected.dtype, key
            assert actual.shape == expected.shape, key
            assert actual.tobytes() == expected.tobytes(), key
            assert actual.flags.owndata, key
        metadata = saved["metadata/other"].attrs
        assert block.observation_name == saved.attrs["observation_name"]
        assert block.trajectory_info_json == metadata["trajectory_info"]
        assert block.trajectory_info == json.loads(metadata["trajectory_info"])
        assert block.source_shared_array_sha256 == json.loads(
            metadata["shared_array_sha256"]
        )
        site = saved["instrument"].attrs
        for key, attr, unit, atol in (
            ("lat", "site_lat_deg", u.deg, 1e-12),
            ("lon", "site_lon_deg", u.deg, 1e-12),
            ("height", "site_alt_m", u.m, 1e-6),
        ):
            assert isinstance(block.source_site[key], u.Quantity)
            assert block.source_site[key].to_value(unit) == pytest.approx(
                site[attr], rel=0, abs=atol
            )
    assert not hasattr(block, "telescope") and not hasattr(block, "focalplane")
    assert block.max_phase_residual_s <= block.time_tolerance_s
    print(
        f"{path.name}: N={len(block.arrays['times'])}, "
        f"actual_rate={block.actual_rate_hz:.12f} Hz, "
        f"residual={block.max_phase_residual_s:.12g} s, "
        f"tolerance={block.time_tolerance_s:.12g} s; 12 fields byte-identical"
    )


def test_committed_arrays_and_metadata(sample_path):
    assert_original_arrays_and_metadata(sample_path)


def test_committed_write_load_roundtrip(sample_path, tmp_path):
    """Rebuild writer input from committed original arrays, without local ECSV."""
    original = _load_trajectory_block(sample_path)
    arrays, info = original.arrays, original.trajectory_info
    samples = QTable()
    # Use original degrees, not a lossy inverse conversion from stored radians.
    for column, field, unit in (
        ("timestamp_unix_s", "times", u.s),
        ("time_s", "fyst_time_s", u.s),
        ("block_time_s", "fyst_block_time_s", u.s),
        ("az_deg", "fyst_az_deg", u.deg),
        ("el_deg", "fyst_el_deg", u.deg),
        ("az_velocity", "fyst_az_velocity_deg_s", u.deg / u.s),
        ("el_velocity", "fyst_el_velocity_deg_s", u.deg / u.s),
    ):
        samples[column] = arrays[field] * unit
    for column, field in (
        ("scan_flag", "fyst_scan_flag"),
        ("scan_index", "scan_index"),
        ("subscan_index", "subscan_index"),
    ):
        samples[column] = arrays[field]
    # The fixture contains one block, so only the table slice becomes local.
    block = {**info["block"], "first": 0, "last": len(samples)}
    samples.meta.update(case=info["case"], start_utc=info["start_utc"], blocks=[block])
    with h5py.File(sample_path, "r") as saved:
        site_name = saved["instrument"].attrs["site_name"]
    # Bundled offline IERS suffices for this input-array / manifest test. Pointing
    # is recomputed; full-file identity uses the original ECSV + IERS separately.
    with solar_system_ephemeris.set("builtin"):
        files = write_trajectory_blocks(
            samples,
            site_name=site_name,
            latitude_deg=original.source_site["lat"].to_value(u.deg),
            longitude_deg=original.source_site["lon"].to_value(u.deg),
            height_m=original.source_site["height"].to_value(u.m),
            output_dir=tmp_path,
            source_sha256=info["source_sha256"],
            roll_deg=info["roll_deg"],
        )
    assert files == [sample_path.name]
    path = tmp_path / files[0]
    loaded = _load_trajectory_block(path)
    assert loaded.observation_name == original.observation_name
    assert loaded.trajectory_info == {**info, "block": block}
    assert set(loaded.arrays) == set(FIELDS)
    for key in FIELDS:
        assert loaded.arrays[key].dtype == arrays[key].dtype, key
        assert loaded.arrays[key].shape == arrays[key].shape, key
        assert loaded.arrays[key].tobytes() == arrays[key].tobytes(), key
    with h5py.File(path, "r") as saved:
        manifest = loaded.source_shared_array_sha256
        assert manifest is not None
        assert len(manifest) == 17
        assert set(manifest) == set(saved["shared"])
        for key, dataset in saved["shared"].items():
            assert hashlib.sha256(dataset[()].tobytes()).hexdigest() == manifest[key]


@pytest.mark.parametrize("relative", CES_FILES + DAISY_FILES)
def test_optional_output_arrays_and_metadata(relative):
    assert_original_arrays_and_metadata(input_path(relative))


def test_committed_clock_and_nominal_rate(sample_path):
    expected = SAMPLE_EXPECTATIONS[sample_path.name]
    block = _load_trajectory_block(sample_path)
    times = block.arrays["times"]
    assert len(times) == expected["samples"]
    flags, counts = np.unique(block.arrays["fyst_scan_flag"], return_counts=True)
    assert dict(zip(flags.tolist(), counts.tolist())) == expected["flags"]
    assert np.all(np.diff(times) > 0)
    assert block.actual_rate_hz == pytest.approx(expected["rate_hz"], rel=0, abs=1e-12)
    assert block.time_tolerance_s == 2 * np.spacing(abs(times)).max()
    assert block.max_phase_residual_s <= block.time_tolerance_s
    numbers = _check_sample_rate(10, block.actual_rate_hz)
    assert numbers["nominal_rate_hz"] == 10
    assert numbers["actual_rate_hz"] == block.actual_rate_hz
    assert numbers["relative_error"] <= numbers["rtol"] == 1e-6
    with pytest.raises(ValueError, match="Nominal sample rate 20.*actual"):
        _check_sample_rate(20, block.actual_rate_hz)
    if sample_path.name == "fyst_ces_000_001.h5":
        with pytest.raises(ValueError, match="rtol 1e-08"):
            _check_sample_rate(10, block.actual_rate_hz, rtol=1e-8)
    assert _check_sample_rate(10, 10, rtol=0)["relative_error"] == 0
    print(f"{sample_path.name}: flags={expected['flags']}; 10 Hz PASS, 20 Hz rejected")


def test_optional_first_ces_clock_and_nominal_rate():
    # Retain the original 850-sample case alongside the new default CES block.
    block = _load_trajectory_block(input_path(CES_FILES[0]))
    assert len(block.arrays["times"]) == 850
    flags, counts = np.unique(block.arrays["fyst_scan_flag"], return_counts=True)
    assert dict(zip(flags.tolist(), counts.tolist())) == {1: 830, 2: 20}
    assert block.actual_rate_hz == pytest.approx(9.999997349033228, rel=0, abs=1e-12)
    assert block.time_tolerance_s == 2 * np.spacing(abs(block.arrays["times"])).max()
    numbers = _check_sample_rate(10, block.actual_rate_hz)
    assert numbers["nominal_rate_hz"] == 10
    assert numbers["actual_rate_hz"] == block.actual_rate_hz
    assert numbers["relative_error"] == pytest.approx(2.65096747e-7)
    assert numbers["rtol"] == 1e-6
    with pytest.raises(ValueError, match="Nominal sample rate 20.*actual"):
        _check_sample_rate(20, block.actual_rate_hz)
    with pytest.raises(ValueError, match="rtol 1e-08"):
        _check_sample_rate(10, block.actual_rate_hz, rtol=1e-8)
    assert _check_sample_rate(10, 10, rtol=0)["relative_error"] == 0
    print(f"CES flags={{1: 830, 2: 20}}; 10 Hz PASS, 20 Hz rejected; {numbers}")


def test_ces_sweeps_and_complete_turnarounds(ces_path):
    block = _load_trajectory_block(ces_path)
    flags = block.arrays["fyst_scan_flag"]
    velocity = block.arrays["fyst_az_velocity_deg_s"]
    # Runs use half-open sample spans [first, last), without changing the input.
    edges = np.r_[0, np.flatnonzero(flags[1:] != flags[:-1]) + 1, len(flags)]
    assert edges.tolist() == [0, 477, 497, 1175, 1195, 1873, 1893, 2571, 2591, 2611]
    assert flags[edges[:-1]].tolist() == [1, 2, 1, 2, 1, 2, 1, 2, 1]
    for run, (first, last) in enumerate(pairwise(edges)):
        if run % 2 == 0:
            sign = [-1, 1, -1, 1, -1][run // 2]
            assert np.all(np.sign(velocity[first:last]) == sign)
        else:
            assert last - first == 20
            assert np.sign(velocity[first]) == np.sign(velocity[first - 1])
            assert np.sign(velocity[last - 1]) == np.sign(velocity[last])
            assert np.sign(velocity[first]) == -np.sign(velocity[last - 1])
    assert np.all(block.arrays["fyst_el_deg"] == 50)


def test_daisy_starts_in_turnaround_and_varies_elevation():
    block = _load_trajectory_block(SAMPLE_ROOT / "fyst_daisy_003_000.h5")
    flags = block.arrays["fyst_scan_flag"]
    assert np.all(flags[:5] == 2)
    assert np.all(flags[5:] == 1)
    elevation = block.arrays["fyst_el_deg"]
    assert elevation.min() == pytest.approx(79.04457108202241, rel=0, abs=1e-12)
    assert elevation.max() == pytest.approx(80.51415846553718, rel=0, abs=1e-12)


@pytest.mark.parametrize(
    "damage, message",
    [
        ("missing_field", "Missing trajectory field: fyst_el_deg"),
        ("non_increasing", "times must be strictly increasing"),
        ("gap", "uniform clock.*gap or jitter"),
        ("unknown_flag", r"Unknown fyst_scan_flag values: \[7\]"),
        ("hash", "hash mismatch for fyst_az_deg"),
        ("derived_hash", "hash mismatch for boresight_azel"),
        ("samples", r"block.samples must equal N={samples}"),
        ("nonfinite", "fyst_el_deg: all values must be finite"),
        ("shape", "fyst_el_deg: expected a 1-D array"),
        ("json", "trajectory_info must be valid JSON"),
    ],
)
def test_reject_corrupted_copy(sample_path, tmp_path, damage, message):
    path = tmp_path / "corrupted.h5"
    shutil.copyfile(sample_path, path)
    message = message.replace(
        "{samples}", str(SAMPLE_EXPECTATIONS[sample_path.name]["samples"])
    )
    with h5py.File(path, "r+") as saved:
        shared = saved["shared"]
        metadata = saved["metadata/other"].attrs
        if damage == "missing_field":
            del shared["fyst_el_deg"]
        elif damage == "non_increasing":
            shared["times"][10] = shared["times"][9]
        elif damage == "gap":
            midpoint = len(shared["times"]) // 2
            shared["times"][midpoint:] += 0.1
        elif damage == "unknown_flag":
            shared["fyst_scan_flag"][10] = 7
        elif damage == "hash":
            shared["fyst_az_deg"][10] += 1
        elif damage == "derived_hash":
            shared["boresight_azel"][10, 0] += 1
        elif damage == "samples":
            info = json.loads(metadata["trajectory_info"])
            info["block"]["samples"] += 1
            metadata["trajectory_info"] = json.dumps(info)
        elif damage == "nonfinite":
            shared["fyst_el_deg"][10] = np.nan
        elif damage == "shape":
            values = shared["fyst_el_deg"][()]
            del shared["fyst_el_deg"]
            ds = shared.create_dataset("fyst_el_deg", data=values[:, None])
            ds.attrs["comm_type"] = "column"
        elif damage == "json":
            metadata["trajectory_info"] = "{invalid"
        if damage not in ("hash", "derived_hash"):
            refresh_manifest(saved)
    with pytest.raises(ValueError, match=message):
        _load_trajectory_block(path)


def test_optional_manifest_and_enum_zero_three(ces_path, tmp_path):
    path = tmp_path / "without_manifest.h5"
    shutil.copyfile(ces_path, path)
    with h5py.File(path, "r+") as saved:
        del saved["metadata/other"].attrs["shared_array_sha256"]
        saved["shared/fyst_scan_flag"][:2] = [0, 3]
    block = _load_trajectory_block(path)
    assert block.source_shared_array_sha256 is None
    assert block.arrays["fyst_scan_flag"][:2].tolist() == [0, 3]


@pytest.mark.parametrize(
    "mode", ["success", "validation_error", "load_error", "zero", "two"]
)
def test_temporary_data_cleanup(ces_path, monkeypatch, mode):
    from toast import Data, ops

    native_apply = ops.LoadHDF5.apply
    native_clear = Data.clear
    cleared = []

    def apply(loader, data):
        native_apply(loader, data)
        if mode == "validation_error":
            del data.obs[0].shared["fyst_el_deg"]
        elif mode == "load_error":
            raise RuntimeError("Synthetic failure after native load")
        elif mode == "zero":
            native_clear(data)
        elif mode == "two":
            native_apply(loader, data)

    def clear(data):
        native_clear(data)
        cleared.append(len(data.obs))

    monkeypatch.setattr(ops.LoadHDF5, "apply", apply)
    monkeypatch.setattr(Data, "clear", clear)
    if mode == "success":
        block = _load_trajectory_block(ces_path)
        assert block.arrays["times"].size == 2611
    elif mode == "load_error":
        with pytest.raises(RuntimeError, match="Synthetic failure"):
            _load_trajectory_block(ces_path)
    else:
        message = (
            "Missing trajectory field" if mode == "validation_error" else "exactly one"
        )
        with pytest.raises(ValueError, match=message):
            _load_trajectory_block(ces_path)
    assert cleared and all(count == 0 for count in cleared)


@pytest.mark.parametrize(
    "nominal, actual, rtol", [(0, 10, 1e-6), (10, np.nan, 1e-6), (10, 10, -1)]
)
def test_invalid_rate_arguments(nominal, actual, rtol):
    with pytest.raises(ValueError, match="finite"):
        _check_sample_rate(nominal, actual, rtol=rtol)
