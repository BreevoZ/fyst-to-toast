"""YAML argument handling and offline configuration for the window CES CLI."""

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

import yaml
from fyst_trajectories import get_primecam_offset
from fyst_trajectories.planning import FieldRegion


REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "examples/windows/deep56_e50_rising.yaml"


@pytest.fixture(scope="module")
def window_cli():
    spec = importlib.util.spec_from_file_location(
        "window_ces_to_ecsv", REPO / "scripts/window_ces_to_ecsv.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def window_file(tmp_path):
    path = tmp_path / "custom_window.yaml"
    path.write_text("az_window: [50, 76]\nelevation: 50\nvelocity: 1\n")
    return path


@pytest.fixture
def required(window_file):
    return [
        "--window-file", str(window_file),
        "--start", "2026-08-13T03:48:45.147", "--duration-s", "120",
    ]


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


@pytest.fixture
def reject(window_cli, required, tmp_path, monkeypatch, capsys):
    def run(*arguments, messages=()):
        folder = tmp_path / "generated"
        monkeypatch.setattr(sys, "argv", [
            "window_ces_to_ecsv.py", *required, "--output-dir", str(folder), *arguments,
        ])
        with pytest.raises(SystemExit) as error:
            window_cli.main()
        assert error.value.code == 2
        stderr = capsys.readouterr().err
        for message in messages:
            assert message in stderr
        assert not folder.exists()
    return run


def test_cli_defaults(window_cli, required, window_file, captured, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(window_cli, "REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["window_ces_to_ecsv.py", *required])
    window_cli.main()
    folder, kwargs = captured[0]
    assert folder == tmp_path / "output/generated" / window_file.stem
    assert kwargs["az_window"] == [50, 76]
    assert kwargs["elevation"] == 50 and kwargs["velocity"] == 1
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


@pytest.mark.parametrize("rate", [None, 20.0])
def test_cli_template(window_cli, required, captured, tmp_path, monkeypatch, rate):
    original = TEMPLATE.read_bytes()
    folder = tmp_path / "generated"
    argv = ["window_ces_to_ecsv.py", *required, "--window-file", str(TEMPLATE),
            "--output-dir", str(folder)]
    if rate is not None:
        argv.extend(["--sample-rate", str(rate)])
    monkeypatch.setattr(sys, "argv", argv)
    window_cli.main()
    destination, kwargs = captured[0]
    assert destination == folder
    assert kwargs["az_window"] == [50, 76]
    assert kwargs["elevation"] == 50 and kwargs["velocity"] == 1
    assert kwargs["az_accel"] == 1.0
    assert kwargs["sample_rate_hz"] == (10.0 if rate is None else rate)
    assert kwargs["detector_offset"] == get_primecam_offset("i6")
    assert kwargs["detector_label"] == "i6"
    assert kwargs["field"] == FieldRegion(20, -3, 60, 14)
    assert TEMPLATE.read_bytes() == original


def test_cli_optional_inputs(window_cli, required, window_file, captured, tmp_path, monkeypatch):
    with window_file.open("a") as stream:
        stream.write("az_accel: 0.5\n")
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
        "window_ces_to_ecsv.py", *required, "--output-dir", str(folder),
        "--sample-rate", "20", "--iers-file", str(table_path), "--overwrite",
    ])
    window_cli.main()
    destination, kwargs = captured[0]
    assert destination == folder
    assert kwargs["sample_rate_hz"] == 20.0 and kwargs["az_accel"] == 0.5
    assert kwargs["iers_file"] == table_path and opened == [str(table_path)]
    assert kwargs["overwrite"] is True


def test_cli_bad_module(window_file, captured, reject):
    with window_file.open("a") as stream:
        stream.write("module: invalid-module\n")
    with pytest.raises(KeyError) as native_error:
        get_primecam_offset("invalid-module")
    reject(messages=(str(window_file), "module", native_error.value.args[0]))
    assert not captured


@pytest.mark.parametrize("option", ["--window-file", "--start", "--duration-s"])
def test_cli_required_arguments(required, reject, option):
    index = required.index(option)
    del required[index:index + 2]
    reject(messages=(option,))


@pytest.mark.parametrize("key,value,message", [
    ("az_window", "50, 76", "az_window"),
    ("az_window", [50], "az_window"),
    ("az_window", [50, 76, 80], "az_window"),
    ("az_window", [True, 76], "az_window[0]"),
    ("az_window", [50, "76"], "az_window[1]"),
    ("az_window", [50, float("inf")], "az_window[1]"),
    ("elevation", None, "elevation"),
    ("elevation", "50", "elevation"),
    ("elevation", float("nan"), "elevation"),
    ("velocity", True, "velocity"),
    ("velocity", [1], "velocity"),
    ("velocity", 0, "velocity"),
    ("az_accel", "1", "az_accel"),
    ("az_accel", -1, "az_accel"),
    ("timestep", False, "timestep"),
    ("timestep", None, "timestep"),
    ("timestep", "0.1", "timestep"),
    ("timestep", 0, "timestep"),
    ("timestep", float("inf"), "timestep"),
    ("module", 6, "module"),
    ("module", None, "module"),
    ("field", [20, -3, 60, 14], "field"),
    ("field", None, "field"),
    ("field", {"ra_center": 20, "dec_center": "-3", "width": 60, "height": 14}, "field.dec_center"),
    ("field", {"ra_center": 20, "dec_center": -3, "width": 0, "height": 14}, "width"),
    ("field", {"ra_center": 20, "dec_center": -3, "width": 60, "height": float("nan")}, "field.height"),
])
def test_cli_yaml_values(window_file, captured, reject, key, value, message):
    data = yaml.safe_load(window_file.read_text())
    data[key] = value
    window_file.write_text(yaml.safe_dump(data))
    reject(messages=(str(window_file), message))
    assert not captured


@pytest.mark.parametrize("scope,keys", [
    ("scan", ("az_window", "elevation", "velocity")),
    ("field", ("ra_center", "dec_center", "width", "height")),
])
@pytest.mark.parametrize("kind", ["unknown", "missing"])
def test_cli_yaml_keys(window_file, captured, reject, scope, keys, kind):
    data = yaml.safe_load(TEMPLATE.read_text())
    target = data if scope == "scan" else data["field"]
    if kind == "unknown":
        keys = ("typo", "other_typo")
        target.update(dict.fromkeys(keys, 1))
    else:
        for key in keys:
            del target[key]
    window_file.write_text(yaml.safe_dump(data))
    messages = (str(window_file), f"{kind} keys", *keys)
    if scope == "field":
        messages += ("field.",)
    reject(messages=messages)
    assert not captured


@pytest.mark.parametrize("content", [
    "", "- a\n- b\n", "a scalar", "az_window: [broken", "!!python/object:bad {}",
])
def test_cli_invalid_yaml(window_file, captured, reject, content):
    window_file.write_text(content)
    messages = (str(window_file), "Cannot load window")
    if content == "az_window: [broken":
        messages += (content, "line 1", "column",)
    reject(messages=messages)
    assert not captured


def test_cli_missing_yaml_file(window_file, captured, reject):
    window_file.unlink()
    reject(messages=(str(window_file), "Cannot load window"))
    assert not captured


@pytest.mark.parametrize("arguments,message", [
    (["--duration-s", "-1"], "--duration-s must be finite and positive"),
    (["--duration-s", "nan"], "--duration-s must be finite and positive"),
    (["--sample-rate", "0"], "--sample-rate must be finite and positive"),
    (["--sample-rate", "inf"], "--sample-rate must be finite and positive"),
    (["--start", "invalid-time"], "Cannot parse UTC start"),
])
def test_cli_bad_arguments(captured, reject, arguments, message):
    reject(*arguments, messages=(message,))
    assert not captured


@pytest.mark.parametrize("arguments", [
    ["--az-window", "50", "76"], ["--elevation", "50"], ["--velocity", "1"],
    ["--az-accel", "1"], ["--module", "i6"], ["--field", "20", "-3", "60", "14"],
])
def test_cli_old_flags_rejected(captured, reject, arguments):
    reject(*arguments, messages=(f"unrecognized arguments: {' '.join(arguments)}",))
    assert not captured


@pytest.mark.parametrize("name", ["schedule.ecsv", "fyst_trajectory.ecsv", "report.json"])
def test_cli_existing_output(window_cli, required, captured, tmp_path, monkeypatch, capsys, name):
    path = tmp_path / name
    path.write_text("keep")
    monkeypatch.setattr(sys, "argv", [
        "window_ces_to_ecsv.py", *required, "--output-dir", str(tmp_path),
    ])
    with pytest.raises(SystemExit) as error:
        window_cli.main()
    assert error.value.code == 2 and not captured
    assert "Outputs exist" in capsys.readouterr().err
    assert path.read_text() == "keep"


@pytest.mark.parametrize("kind", ["missing", "malformed", "collision"])
def test_cli_bad_iers_file(window_cli, required, captured, tmp_path, monkeypatch, capsys, kind):
    folder = tmp_path / "generated"
    path = folder / "schedule.ecsv" if kind == "collision" else tmp_path / "iers.txt"
    if kind != "missing":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not an IERS table")
    monkeypatch.setattr(sys, "argv", [
        "window_ces_to_ecsv.py", *required, "--output-dir", str(folder),
        "--iers-file", str(path), "--overwrite",
    ])
    with pytest.raises(SystemExit) as error:
        window_cli.main()
    assert error.value.code == 2 and not captured
    assert "error:" in capsys.readouterr().err
    if kind == "collision":
        assert list(folder.iterdir()) == [path]
        assert path.read_text() == "not an IERS table"
    else:
        assert not folder.exists()


def test_cli_reversed_window(window_file, reject):
    window_file.write_text("az_window: [76, 50]\nelevation: 50\nvelocity: 1\n")
    reject(messages=(str(window_file), "az_hi > az_lo"))


def test_cli_september_offline(window_cli, required, window_file, tmp_path, monkeypatch):
    window_file.write_text("az_window: [283, 309]\nelevation: 50\nvelocity: 1\ntimestep: 0.1\n")
    folder = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", [
        "window_ces_to_ecsv.py", *required, "--start", "2026-09-11T06:32:37.457",
        "--output-dir", str(folder),
    ])
    window_cli.main()
    report = json.loads((folder / "report.json").read_text())
    assert report["status"] == "complete" and report["total_science_samples"] > 0
    assert report["first_sample_utc"] == "2026-09-11T06:32:37.457"
    assert report["planner_inputs"]["timestep"] == 0.1
    assert report["iers_auto_download"] is False and report["iers_auto_max_age"] is None
