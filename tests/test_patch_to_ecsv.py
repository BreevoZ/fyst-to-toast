"""Optional FYST contracts for YAML templates and the patch CLI."""

import os

for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import json
import sys

import pytest

pytest.importorskip("fyst_trajectories", exc_type=ImportError)

import yaml
from astropy.table import QTable
from fyst_trajectories.overhead import ObservingPatch


@pytest.mark.parametrize("case,scan_type,timestep_index", [
    ("ces", "constant_el", 0), ("daisy", "daisy", -1), ("pong", "pong", -1),
])
def test_template_loads(quickstart, case, scan_type, timestep_index):
    path = quickstart.PATCH_DIR / f"{case}.yaml"
    patch = quickstart.example_patch(case)
    assert isinstance(patch, ObservingPatch)
    assert patch.scan_type == scan_type
    assert patch.scan_params["timestep"] == 0.1
    assert list(patch.scan_params)[timestep_index] == "timestep"
    assert patch == quickstart.load_patch(path)


@pytest.fixture
def custom_patch_file(tmp_path):
    path = tmp_path / "small_field.yaml"
    path.write_text(
        "name: custom YAML Pong\nra_center: 185\ndec_center: -35\n"
        "width: 1.2\nheight: 1.0\nscan_type: pong\nvelocity: 0.3\n"
        "scan_params:\n  spacing: 0.1\n  num_terms: 4\n"
        "  timestep: 0.2\n  angle: 15.0\n"
    )
    return path


@pytest.mark.parametrize("rate", [None, 20.0])
def test_cli_custom_yaml_completes(quickstart, custom_patch_file, tmp_path, monkeypatch, rate):
    folder = tmp_path / "generated"
    original = custom_patch_file.read_bytes()
    argv = [
        "patch_to_ecsv.py", "--patch-file", str(custom_patch_file),
        "--start", "2026-03-15T04:00:00", "--duration-s", "1800",
        "--output-dir", str(folder),
    ]
    if rate is not None:
        argv.extend(["--sample-rate", str(rate)])
    monkeypatch.setattr(sys, "argv", argv)
    quickstart.main()
    report = json.loads((folder / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["case"] == "pong"
    assert report["patch_name"] == "custom YAML Pong"
    assert report["requested_sample_rate_hz"] == (5.0 if rate is None else rate)
    samples = QTable.read(folder / "fyst_trajectory.ecsv")
    assert samples.meta["case"] == "pong"
    assert len(samples) == report["total_science_samples"] > 0
    expected_step = 0.2 if rate is None else 1.0 / rate
    assert all(block["requested_timestep_s"] == expected_step for block in samples.meta["blocks"])
    assert custom_patch_file.read_bytes() == original


@pytest.mark.parametrize("selection,message", [
    ([], "Exactly one of case or --patch-file"),
    (["ces", "--patch-file", "field.yaml"], "Exactly one of case or --patch-file"),
    (["--patch-file", "field.yaml"], "--start is required with --patch-file"),
])
def test_cli_requires_one_input_and_custom_start(quickstart, tmp_path, monkeypatch, capsys,
                                             selection, message):
    folder = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", ["patch_to_ecsv.py", *selection, "--output-dir", str(folder)])
    with pytest.raises(SystemExit) as error:
        quickstart.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not folder.exists()


@pytest.mark.parametrize("key,value", [
    ("unknown_field", 1), ("width", -1), ("scan_type", "unknown_pattern"),
    ("scan_params", {"bad_scan_key": 1}),
])
def test_cli_rejects_invalid_patch(quickstart, custom_patch_file, tmp_path, monkeypatch,
                                  capsys, key, value):
    data = yaml.safe_load(custom_patch_file.read_text())
    data[key] = value
    custom_patch_file.write_text(yaml.safe_dump(data, sort_keys=False))
    folder = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", [
        "patch_to_ecsv.py", "--patch-file", str(custom_patch_file),
        "--start", "2026-03-15T04:00:00", "--output-dir", str(folder),
    ])
    with pytest.raises(SystemExit) as error:
        quickstart.main()
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert str(custom_patch_file) in stderr
    assert ("bad_scan_key" if key == "scan_params" else key) in stderr
    assert not folder.exists()


@pytest.mark.parametrize("content", ["", "- a\n- b\n", "a scalar", "name: [broken"])
def test_cli_rejects_invalid_yaml(quickstart, tmp_path, monkeypatch, capsys, content):
    path = tmp_path / "invalid.yaml"
    path.write_text(content)
    folder = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", [
        "patch_to_ecsv.py", "--patch-file", str(path),
        "--start", "2026-03-15T04:00:00", "--output-dir", str(folder),
    ])
    with pytest.raises(SystemExit) as error:
        quickstart.main()
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert str(path) in stderr and "Cannot load patch" in stderr
    assert not folder.exists()


@pytest.mark.parametrize("custom", [False, True])
def test_cli_default_output_directory(quickstart, custom_patch_file, tmp_path, monkeypatch, custom):
    # Capture the destination without writing into the repository's default output/.
    calls = []

    def capture(patch, output_dir, **kwargs):
        calls.append((patch, output_dir, kwargs))
        return {"status": "complete"}

    monkeypatch.setattr(quickstart, "REPO", tmp_path)
    monkeypatch.setattr(quickstart, "generate_trajectory", capture)
    selection = (["--patch-file", str(custom_patch_file), "--start", "2026-03-15T04:00:00"]
                 if custom else ["ces"])
    monkeypatch.setattr(sys, "argv", ["patch_to_ecsv.py", *selection])
    quickstart.main()
    patch, folder, kwargs = calls[0]
    assert isinstance(patch, ObservingPatch)
    assert folder == tmp_path / "output/generated" / (custom_patch_file.stem if custom else "ces")
    assert kwargs["sample_rate_hz"] is None
    assert kwargs["duration_s"] == 10800
    if not custom:
        assert kwargs["start"].utc.isot == quickstart.DEFAULT_START["ces"] + ".000"
