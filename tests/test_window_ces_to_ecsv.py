"""Argument handling and offline configuration for the window CES CLI."""

import os

for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import importlib.util
import json
from pathlib import Path
import sys

import pytest
from astropy.coordinates import solar_system_ephemeris
from astropy.utils import iers

pytest.importorskip("fyst_trajectories", exc_type=ImportError)

from fyst_trajectories import get_primecam_offset
from fyst_trajectories.planning import FieldRegion


REQUIRED = [
    "--az-window", "50", "76", "--elevation", "50", "--velocity", "1",
    "--start", "2026-08-13T03:48:45.147", "--duration-s", "120",
]


@pytest.fixture(scope="module")
def window_cli():
    path = Path(__file__).resolve().parents[1] / "scripts/window_ces_to_ecsv.py"
    spec = importlib.util.spec_from_file_location("window_ces_to_ecsv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def restore_astropy_settings():
    with (
        iers.conf.set_temp("auto_download", False),
        iers.conf.set_temp("auto_max_age", 30),
        iers.conf.set_temp("iers_degraded_accuracy", "error"),
        iers.earth_orientation_table.set(iers.earth_orientation_table.get()),
        solar_system_ephemeris.set("builtin"),
    ):
        yield


@pytest.fixture
def captured(window_cli, monkeypatch):
    calls = []

    def capture(output_dir, **kwargs):
        assert iers.conf.auto_download is False
        assert iers.conf.auto_max_age is None
        assert iers.conf.iers_degraded_accuracy == "error"
        assert solar_system_ephemeris.get() == "builtin"
        calls.append((output_dir, kwargs))
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"status": "complete"}

    monkeypatch.setattr(window_cli, "generate_window_ces", capture)
    return calls


def test_cli_defaults(window_cli, captured, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(window_cli, "REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["window_ces_to_ecsv.py", *REQUIRED])
    window_cli.main()
    folder, kwargs = captured[0]
    assert folder == tmp_path / "output/generated/window_ces"
    assert kwargs["az_window"] == [50.0, 76.0]
    assert kwargs["elevation"] == 50.0 and kwargs["velocity"] == 1.0
    assert kwargs["duration_s"] == 120.0 and kwargs["az_accel"] == 1.0
    assert kwargs["start"].scale == "utc" and kwargs["start"].precision == 3
    assert kwargs["start"].isot == "2026-08-13T03:48:45.147"
    for key in ("sample_rate_hz", "detector_offset", "detector_label", "field", "iers_file"):
        assert kwargs[key] is None
    assert kwargs["overwrite"] is False
    report = json.loads((folder / "report.json").read_text())
    assert report["iers_auto_max_age"] is None
    stdout = capsys.readouterr().out
    assert '"iers_auto_max_age": null' in stdout
    assert f"Output: {folder}" in stdout


def test_cli_optional_inputs(window_cli, captured, tmp_path, monkeypatch):
    table_path = tmp_path / "offline-iers.txt"
    table_path.write_text("table supplied by test")
    opened = []
    table = iers.earth_orientation_table.get()

    def open_table(path):
        opened.append(path)
        return table

    monkeypatch.setattr(iers.IERS_A, "open", open_table)
    folder = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", [
        "window_ces_to_ecsv.py", *REQUIRED, "--output-dir", str(folder),
        "--sample-rate", "20", "--az-accel", "0.5", "--module", "i6",
        "--field", "20", "-3", "60", "14", "--iers-file", str(table_path), "--overwrite",
    ])
    window_cli.main()
    destination, kwargs = captured[0]
    assert destination == folder
    assert kwargs["sample_rate_hz"] == 20.0 and kwargs["az_accel"] == 0.5
    assert kwargs["detector_offset"] == get_primecam_offset("i6")
    assert kwargs["detector_label"] == "i6"
    assert kwargs["field"] == FieldRegion(20, -3, 60, 14)
    assert kwargs["iers_file"] == table_path and opened == [str(table_path)]
    assert kwargs["overwrite"] is True


def test_cli_bad_module(window_cli, tmp_path, monkeypatch, capsys):
    folder = tmp_path / "generated"
    with pytest.raises(KeyError) as native_error:
        get_primecam_offset("invalid-module")
    monkeypatch.setattr(sys, "argv", [
        "window_ces_to_ecsv.py", *REQUIRED, "--module", "invalid-module",
        "--output-dir", str(folder),
    ])
    with pytest.raises(SystemExit) as error:
        window_cli.main()
    assert error.value.code == 2
    assert native_error.value.args[0] in capsys.readouterr().err
    assert not folder.exists()


@pytest.mark.parametrize("option,count", [
    ("--az-window", 2), ("--elevation", 1), ("--velocity", 1),
    ("--start", 1), ("--duration-s", 1),
])
def test_cli_required_arguments(window_cli, tmp_path, monkeypatch, capsys, option, count):
    args = REQUIRED.copy()
    index = args.index(option)
    del args[index:index + count + 1]
    folder = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", ["window_ces_to_ecsv.py", *args, "--output-dir", str(folder)])
    with pytest.raises(SystemExit) as error:
        window_cli.main()
    assert error.value.code == 2
    assert option in capsys.readouterr().err
    assert not folder.exists()


@pytest.mark.parametrize("arguments,message", [
    (["--az-window", "50"], "expected 2 arguments"),
    (["--field", "20", "-3", "60"], "expected 4 arguments"),
    (["--az-window", "76", "50"], "az_hi > az_lo"),
    (["--elevation", "nan"], "must be finite"),
    (["--velocity", "0"], "velocity"),
    (["--duration-s", "-1"], "duration"),
    (["--sample-rate", "0"], "sample_rate_hz"),
    (["--sample-rate", "inf"], "must be finite"),
    (["--az-accel", "-1"], "az_accel"),
    (["--start", "invalid-time"], "Cannot parse UTC start"),
    (["--field", "20", "-3", "0", "14"], "width"),
    (["--field", "20", "nan", "60", "14"], "must be finite"),
])
def test_cli_bad_inputs(window_cli, tmp_path, monkeypatch, capsys, arguments, message):
    folder = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", [
        "window_ces_to_ecsv.py", *REQUIRED, "--output-dir", str(folder), *arguments,
    ])
    with pytest.raises(SystemExit) as error:
        window_cli.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not folder.exists()


@pytest.mark.parametrize("name", ["schedule.ecsv", "fyst_trajectory.ecsv", "report.json"])
def test_cli_existing_output(window_cli, captured, tmp_path, monkeypatch, capsys, name):
    path = tmp_path / name
    path.write_text("keep")
    monkeypatch.setattr(sys, "argv", [
        "window_ces_to_ecsv.py", *REQUIRED, "--output-dir", str(tmp_path),
    ])
    with pytest.raises(SystemExit) as error:
        window_cli.main()
    assert error.value.code == 2 and not captured
    assert "Outputs exist" in capsys.readouterr().err
    assert path.read_text() == "keep"


@pytest.mark.parametrize("kind", ["missing", "malformed", "collision"])
def test_cli_bad_iers_file(window_cli, tmp_path, monkeypatch, capsys, kind):
    folder = tmp_path / "generated"
    path = folder / "schedule.ecsv" if kind == "collision" else tmp_path / "iers.txt"
    if kind != "missing":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not an IERS table")
    monkeypatch.setattr(sys, "argv", [
        "window_ces_to_ecsv.py", *REQUIRED, "--output-dir", str(folder),
        "--iers-file", str(path), "--overwrite",
    ])
    with pytest.raises(SystemExit) as error:
        window_cli.main()
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
    if kind == "collision":
        assert list(folder.iterdir()) == [path]
        assert path.read_text() == "not an IERS table"
    else:
        assert not folder.exists()


def test_cli_september_offline(window_cli, tmp_path, monkeypatch):
    folder = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", [
        "window_ces_to_ecsv.py", *REQUIRED, "--az-window", "283", "309",
        "--start", "2026-09-11T06:32:37.457", "--output-dir", str(folder),
    ])
    window_cli.main()
    report = json.loads((folder / "report.json").read_text())
    assert report["status"] == "complete" and report["total_science_samples"] > 0
    assert report["first_sample_utc"] == "2026-09-11T06:32:37.457"
    assert report["iers_auto_download"] is False and report["iers_auto_max_age"] is None
