"""Writer preflight and the real round-trip CLI's reader / orientation checks."""

import os

for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import json
from pathlib import Path
import runpy
import sys

import h5py
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.table import QTable
from astropy.utils import iers
from toast import qarray as qa

from fyst_to_toast import trajectory_h5 as schema

ROOT = Path(__file__).resolve().parents[1]
ERRORS = {
    "one_sample": "Expected N >= 2 samples",
    "unknown_enum": "Unknown fyst_scan_flag",
    "jitter": "uniform clock",
    "bad_block_count": "block.samples must equal",
}


@pytest.fixture(autouse=True)
def offline_geometry():
    with iers.conf.set_temp("auto_download", False), solar_system_ephemeris.set("builtin"):
        yield


def input_table(tmp_path, damage=None, *, second_block=False, legacy=False):
    """Synthetic inputs with explicit physical units and a matching schedule hash."""
    count = 1 if damage == "one_sample" else 10
    first = 10 if second_block else 0
    n = first + count
    elapsed = np.arange(n, dtype=np.float64) * 0.1
    samples = QTable({
        "timestamp_unix_s": (1_700_000_000.0 + elapsed) * u.s,
        "time_s": elapsed * u.s,
        "block_time_s": elapsed * u.s,
        "az_deg": (100.0 + elapsed) * u.deg,
        "el_deg": np.full(n, 50.0) * u.deg,
        "az_velocity": np.ones(n) * u.deg / u.s,
        "el_velocity": np.zeros(n) * u.deg / u.s,
        "scan_flag": np.resize(np.array([1, 1, 2, 1], dtype=np.int8), n),
        "scan_index": np.zeros(n, dtype=np.int32),
        "subscan_index": (np.arange(n) >= first).astype(np.int32) if first else np.zeros(n, dtype=np.int32),
    })
    if not legacy:
        samples["boresight_roll_deg"] = np.full(n, -50.0) * u.deg
        samples.meta.update(roll_source="test", roll_convention=schema.ROLL_CONVENTION)
    blocks = []
    if first:
        blocks.append(dict(first=0, last=first, samples=first, scan_index=0,
                           subscan_index=0, native_timestep_s=0.1))
    block = dict(first=first, last=n, samples=count, scan_index=0,
                 subscan_index=1 if first else 0, native_timestep_s=0.1)
    blocks.append(block)
    if damage == "unknown_enum":
        samples["scan_flag"][first + 3] = 7
    elif damage == "jitter":
        samples["timestamp_unix_s"][first + 3] += 0.01 * u.s
    elif damage == "bad_block_count":
        block["samples"] = 999
    elif damage is not None and damage.startswith("nonfinite:"):
        key = damage.split(":", 1)[1]
        if getattr(samples[key], "unit", None) is None:
            samples[key] = samples[key].astype(float)
            samples[key][first + 3] = np.nan
        else:
            samples[key][first + 3] = np.nan * samples[key].unit
    samples.meta.update(case="test", start_utc="2023-11-14T22:13:20.000", blocks=blocks)
    schedule = QTable({"unused": [0]}, meta={
        "site_name": "test-site", "site_lat": -23.0 * u.deg,
        "site_lon": -68.0 * u.deg, "site_alt": 5600.0 * u.m,
    })
    schedule.write(tmp_path / "schedule.ecsv", format="ascii.ecsv")
    samples.meta["source_schedule_sha256"] = hashlib.sha256(
        (tmp_path / "schedule.ecsv").read_bytes()
    ).hexdigest()
    source = tmp_path / "fyst_trajectory.ecsv"
    samples.write(source, format="ascii.ecsv")
    return samples, source


def cli(monkeypatch, path, *args):
    monkeypatch.setattr(sys, "argv", [path, *map(str, args)])
    runpy.run_path(str(ROOT / path), run_name="__main__")


@pytest.mark.parametrize("damage", list(ERRORS))
@pytest.mark.parametrize("second_block", [False, True])
def test_writer_rejects_unloadable_input_before_any_output(tmp_path, monkeypatch, capsys,
                                                        damage, second_block):
    _, source = input_table(tmp_path, damage, second_block=second_block)
    with pytest.raises(SystemExit) as error:
        cli(monkeypatch, "scripts/save_trajectory_h5.py", source)
    assert error.value.code == 2
    assert ERRORS[damage] in capsys.readouterr().err
    assert not (tmp_path / "hdf5").exists()


@pytest.mark.parametrize("column", list(schema.TRAJECTORY_COLUMNS))
def test_writer_rejects_nonfinite_native_columns(tmp_path, monkeypatch, capsys, column):
    _, source = input_table(tmp_path, f"nonfinite:{column}", second_block=True)
    with pytest.raises(SystemExit) as error:
        cli(monkeypatch, "scripts/save_trajectory_h5.py", source)
    assert error.value.code == 2
    assert "finite" in capsys.readouterr().err
    assert not (tmp_path / "hdf5").exists()


@pytest.mark.parametrize("damage", list(ERRORS))
def test_checker_rejects_files_outside_reader_contract(tmp_path, monkeypatch, damage):
    samples, source = input_table(tmp_path, damage)
    folder = tmp_path / "hdf5"
    folder.mkdir()
    # Deliberately bypass CLI preflight to reproduce the old writer's bad files.
    # The library writer documents that its caller owns input validation.
    names = schema.write_trajectory_blocks(
        samples, site_name="test-site", latitude_deg=-23.0, longitude_deg=-68.0,
        height_m=5600.0, output_dir=folder,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    (folder / "files.json").write_text(json.dumps(names))
    with pytest.raises(ValueError, match=ERRORS[damage]):
        cli(monkeypatch, "tools/check_writer_roundtrip.py", source, folder)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("quaternion_change", ["none", "sign", "drop_roll"])
def test_checker_full_orientation(tmp_path, monkeypatch, capsys, legacy, quaternion_change):
    _, source = input_table(tmp_path, legacy=legacy)
    options = ["--roll-deg", "-50"] if legacy else []
    cli(monkeypatch, "scripts/save_trajectory_h5.py", source, *options)
    folder = tmp_path / "hdf5"
    names = json.loads((folder / "files.json").read_text())
    assert len(names) == 1
    path = folder / names[0]
    if quaternion_change != "none":
        with h5py.File(path, "r+") as saved:
            shared = saved["shared"]
            if quaternion_change == "sign":
                # q and -q must be accepted sample by sample, not just globally.
                shared["boresight_azel"][::2] *= -1
            else:
                shared["boresight_azel"][:] = qa.from_lonlat_angles(
                    -shared["azimuth"][()], shared["elevation"][()],
                    np.zeros(len(shared["times"])),
                )
            metadata = saved["metadata/other"].attrs
            hashes = json.loads(metadata["shared_array_sha256"])
            hashes["boresight_azel"] = hashlib.sha256(shared["boresight_azel"][()].tobytes()).hexdigest()
            metadata["shared_array_sha256"] = json.dumps(hashes)
    capsys.readouterr()
    if quaternion_change == "drop_roll":
        with pytest.raises(AssertionError, match="quaternion orientation"):
            cli(monkeypatch, "tools/check_writer_roundtrip.py", source, folder)
    else:
        cli(monkeypatch, "tools/check_writer_roundtrip.py", source, folder)
        assert '"status": "passed"' in capsys.readouterr().out
