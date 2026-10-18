"""B2.1: native IO, independent byte comparisons, and corrupted scratch copies."""

import os

# Set these before NumPy / TOAST imports, including in direct pytest collection.
os.environ["MPI_DISABLE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import hashlib
import json
import shutil
import socket
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy import units as u
from astropy.utils import iers

from fyst_toast.trajectory_input import _check_sample_rate, _load_trajectory_block

iers.conf.auto_download = False

REPO = Path(__file__).resolve().parents[1]
INPUT_ROOT = Path(os.environ.get("FYST_TOAST_TEST_OUTPUT", REPO / "output")).resolve()
CES_FILES = [f"ces/hdf5/fyst_ces_000_{i:03d}.h5" for i in range(3)]
DAISY_FILES = [f"daisy/hdf5/fyst_daisy_{i:03d}_000.h5" for i in range(4)]
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
    path = INPUT_ROOT / relative
    if not path.is_file():
        pytest.skip(
            f"Writer H5 absent: {path}; set FYST_TOAST_TEST_OUTPUT to supply it"
        )
    return path


@pytest.fixture(scope="session", autouse=True)
def protect_inputs_and_disable_network():
    paths = [INPUT_ROOT / p for p in CES_FILES + DAISY_FILES]
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
    return input_path(CES_FILES[0])


def refresh_manifest(saved):
    """Let semantic checks, not a stale hash, reject deliberately edited copies."""
    manifest = {
        key: hashlib.sha256(ds[()].tobytes()).hexdigest()
        for key, ds in saved["shared"].items()
    }
    saved["metadata/other"].attrs["shared_array_sha256"] = json.dumps(manifest)


@pytest.mark.parametrize("relative", [CES_FILES[0], *DAISY_FILES])
def test_original_arrays_and_metadata(relative):
    path = input_path(relative)
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


def test_ces_clock_and_nominal_rate(ces_path):
    block = _load_trajectory_block(ces_path)
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


@pytest.mark.parametrize(
    "damage, message",
    [
        ("missing_field", "Missing trajectory field: fyst_el_deg"),
        ("non_increasing", "times must be strictly increasing"),
        ("gap", "uniform clock.*gap or jitter"),
        ("unknown_flag", r"Unknown fyst_scan_flag values: \[7\]"),
        ("hash", "hash mismatch for fyst_az_deg"),
        ("derived_hash", "hash mismatch for boresight_azel"),
        ("samples", r"block.samples must equal N=850"),
        ("nonfinite", "fyst_el_deg: all values must be finite"),
        ("shape", "fyst_el_deg: expected a 1-D array"),
        ("json", "trajectory_info must be valid JSON"),
    ],
)
def test_reject_corrupted_copy(ces_path, tmp_path, damage, message):
    path = tmp_path / "corrupted.h5"
    shutil.copyfile(ces_path, path)
    with h5py.File(path, "r+") as saved:
        shared = saved["shared"]
        metadata = saved["metadata/other"].attrs
        if damage == "missing_field":
            del shared["fyst_el_deg"]
        elif damage == "non_increasing":
            shared["times"][10] = shared["times"][9]
        elif damage == "gap":
            shared["times"][425:] += 0.1
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
        assert block.arrays["times"].size == 850
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
