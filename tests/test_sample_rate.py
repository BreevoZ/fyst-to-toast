"""Configurable generation and independent stored / measured SimFYST rate checks."""

import os

for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys

import h5py
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.table import QTable
from astropy.utils import iers
from toast import Data
from toast.instrument import GroundSite, Telescope
from toast.instrument_sim import fake_boresight_focalplane
from toast.mpi import Comm

from fyst_to_toast import trajectory_h5 as schema
from fyst_to_toast.sim_fyst import SimFYST

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def offline_geometry():
    with iers.conf.set_temp("auto_download", False), solar_system_ephemeris.set("builtin"):
        yield


@pytest.fixture(scope="module", params=[("ces", 1470), ("daisy", 1500), ("pong", 2280)])
def generated20(request, tmp_path_factory):
    pytest.importorskip("fyst_trajectories", exc_type=ImportError)
    from fyst_to_toast.generate import generate_trajectory

    case, duration = request.param
    folder = tmp_path_factory.mktemp(f"rate20_{case}")
    with iers.conf.set_temp("auto_download", False), solar_system_ephemeris.set("builtin"):
        report = generate_trajectory(case, folder, duration_s=duration, sample_rate_hz=20.0)
        assert report["status"] == "complete"
        assert report["requested_sample_rate_hz"] == 20.0
        source = folder / "fyst_trajectory.ecsv"
        samples = QTable.read(source)
        assert len(samples) > 0
        assert np.any(samples["scan_flag"] == 1)
        assert all(b["requested_timestep_s"] == 0.05 for b in samples.meta["blocks"])
        site = QTable.read(folder / "schedule.ecsv").meta
        hdf5 = folder / "hdf5"
        hdf5.mkdir()
        names = schema.write_trajectory_blocks(
            samples, site_name=site["site_name"], latitude_deg=site["site_lat"].to_value(u.deg),
            longitude_deg=site["site_lon"].to_value(u.deg), height_m=site["site_alt"].to_value(u.m),
            output_dir=hdf5, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )
    assert len(names) == 1
    return hdf5 / names[0]


def telescope_for(path, rate):
    # Read coordinates independently of the loader whose validation is under test.
    with h5py.File(path) as saved:
        site = dict(saved["instrument"].attrs)
    return Telescope(
        "rate-test", site=GroundSite("rate-test", site["site_lat_deg"] * u.deg,
                                    site["site_lon_deg"] * u.deg, site["site_alt_m"] * u.m),
        focalplane=fake_boresight_focalplane(n_pix=1, sample_rate=rate * u.Hz),
    )


def apply(path, rate, *, manifest=False):
    data = Data(Comm(world=None))
    try:
        if manifest:
            listing = path.parent / "files.json"
            listing.write_text(json.dumps([path.name]))
            inputs = {"trajectory_files": [listing]}
        else:
            inputs = {"trajectory_file": str(path)}
        SimFYST(telescope=telescope_for(path, rate), **inputs).apply(data)
        assert len(data.obs) == 1
        obs = data.obs[0]
        with h5py.File(path) as saved:
            np.testing.assert_array_equal(obs.shared["times"].data, saved["shared/times"][()])
        return json.loads(obs["simfyst_info"])
    except Exception:
        assert data.obs == []
        raise
    finally:
        data.clear()


def set_stored_rate(path, value):
    with h5py.File(path, "r+") as saved:
        attrs = saved["metadata/other"].attrs
        info = json.loads(attrs["trajectory_info"])
        info["sample_rate_hz"] = value
        attrs["trajectory_info"] = json.dumps(info)


def test_generated20_to_simfyst(generated20):
    info = apply(generated20, 20)
    for key in ("rate_check", "stored_rate_check"):
        assert info[key]["nominal_rate_hz"] == 20
        assert info[key]["actual_rate_hz"] == pytest.approx(20, rel=1e-6)
        assert info[key]["relative_error"] <= info[key]["rtol"] == 1e-6
    assert info["stored_rate_check"]["source"] == "trajectory_info.sample_rate_hz"


@pytest.mark.parametrize("manifest", [False, True])
def test_generated20_rejects_10hz_focalplane(generated20, tmp_path, manifest):
    path = tmp_path / generated20.name
    shutil.copyfile(generated20, path)
    with pytest.raises(RuntimeError) as error:
        apply(path, 10, manifest=manifest)
    message = str(error.value)
    assert str(path) in message and "Nominal sample rate 10 Hz" in message
    assert "trajectory_info.sample_rate_hz" in message and "actual 20" in message


@pytest.mark.parametrize("manifest", [False, True])
@pytest.mark.parametrize("nominal,comparison", [(20, "trajectory_info.sample_rate_hz"), (40, "timestamps")])
def test_stored_and_measured_rates_checked_independently(generated20, tmp_path, manifest, nominal, comparison):
    path = tmp_path / "tampered.h5"
    shutil.copyfile(generated20, path)
    set_stored_rate(path, 40.0)
    # At 40 Hz the stored check passes, but the original 20 Hz timestamps must fail.
    with pytest.raises(RuntimeError) as error:
        apply(path, nominal, manifest=manifest)
    message = str(error.value)
    assert str(path) in message and f"focalplane vs {comparison}:" in message
    assert f"Nominal sample rate {nominal} Hz" in message
    assert "actual 40 Hz" in message if nominal == 20 else "actual 20" in message


@pytest.mark.parametrize("name", ["fyst_ces_000_001.h5", "fyst_daisy_003_000.h5"])
def test_missing_legacy_rate_falls_back_to_timestamps(tmp_path, name):
    path = tmp_path / name
    shutil.copyfile(ROOT / "data" / name, path)
    with h5py.File(path, "r+") as saved:
        attrs = saved["metadata/other"].attrs
        info = json.loads(attrs["trajectory_info"])
        del info["sample_rate_hz"]
        attrs["trajectory_info"] = json.dumps(info)
    block = schema._load_trajectory_block(path)
    assert block.sample_rate_hz == block.actual_rate_hz
    info = apply(path, 10)
    stored = dict(info["stored_rate_check"])
    assert stored.pop("source") == "timestamps"
    assert stored == info["rate_check"]


@pytest.mark.parametrize("value", [None, True, "10", [], 0, -10, float("inf"), float("nan")])
def test_invalid_stored_rate_keeps_valueerror(tmp_path, value):
    path = tmp_path / "invalid.h5"
    shutil.copyfile(ROOT / "data/fyst_ces_000_001.h5", path)
    set_stored_rate(path, value)
    with pytest.raises(ValueError, match="sample_rate_hz must be finite and positive"):
        apply(path, 10)


@pytest.mark.parametrize("rate", ["0", "-10", "nan", "inf", "-inf", "bad"])
def test_invalid_sample_rate_cli(tmp_path, monkeypatch, capsys, rate):
    output = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", ["generate_trajectory.py", "ces", f"--sample-rate={rate}",
                                    "--output-dir", str(output)])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(ROOT / "scripts/generate_trajectory.py"), run_name="__main__")
    assert error.value.code == 2
    assert "--sample-rate" in capsys.readouterr().err
    assert not output.exists()
