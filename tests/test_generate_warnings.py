"""FYST warning reports preserve caller filters and warning display."""

import os

for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

from collections import Counter
import json
import warnings

import pytest
from astropy.coordinates import solar_system_ephemeris
from astropy.utils import iers

pytest.importorskip("fyst_trajectories", exc_type=ImportError)

from fyst_trajectories.exceptions import (
    AccelerationLimitWarning, PointingWarning, VelocityLimitWarning,
)
from fyst_trajectories.overhead import ObservingPatch

from fyst_to_toast import generate


@pytest.fixture(autouse=True)
def offline_geometry():
    with iers.conf.set_temp("auto_download", False), solar_system_ephemeris.set("builtin"):
        yield


@pytest.fixture
def fast_pong():
    return ObservingPatch(
        name="fast small Pong", ra_center=185, dec_center=-35,
        width=1, height=1, scan_type="pong", velocity=0.4,
        scan_params={"spacing": 0.1, "num_terms": 4, "timestep": 0.1},
    )


def test_fast_pong_records_and_forwards_warnings(tmp_path, fast_pong):
    with warnings.catch_warnings(record=True) as displayed:
        warnings.simplefilter("always")
        report = generate.generate_trajectory(
            fast_pong, tmp_path, start="2026-03-15T04:00:00", duration_s=1800,
        )
    assert report["status"] == "complete"
    counts = Counter(
        (item.category.__name__, str(item.message)) for item in displayed
        if issubclass(item.category, (AccelerationLimitWarning, VelocityLimitWarning, PointingWarning))
    )
    assert report["fyst_warnings"] == [
        {"category": category, "message": message, "count": count}
        for (category, message), count in counts.items()
    ]
    assert any(category == "AccelerationLimitWarning" and count > 0
               for (category, _), count in counts.items())
    assert json.loads((tmp_path / "report.json").read_text())["fyst_warnings"] == report["fyst_warnings"]


@pytest.mark.parametrize("duration,status", [(1, "no_science"), (1800, "complete")])
def test_no_fyst_warnings_records_empty_list(tmp_path, quickstart, duration, status):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        report = generate.generate_trajectory(
            quickstart.example_patch("ces"), tmp_path,
            start=quickstart.DEFAULT_START["ces"], duration_s=duration,
        )
    assert report["status"] == status
    assert report["fyst_warnings"] == []
    assert json.loads((tmp_path / "report.json").read_text())["fyst_warnings"] == []


def test_acceleration_error_filter_still_raises(tmp_path, fast_pong):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AccelerationLimitWarning)
        original_handler = warnings.showwarning
        original_filters = warnings.filters[:]
        with pytest.raises(AccelerationLimitWarning):
            generate.generate_trajectory(
                fast_pong, tmp_path, start="2026-03-15T04:00:00", duration_s=1800,
            )
        assert warnings.showwarning is original_handler
        assert warnings.filters == original_filters


def test_capture_preserves_custom_display_handler(monkeypatch):
    displayed = []

    def display(*args):
        displayed.append(args)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        monkeypatch.setattr(warnings, "showwarning", display)
        with generate._capture_fyst_warnings() as records:
            warnings.warn_explicit("limit", AccelerationLimitWarning, "source.py", 12)
            warnings.warn_explicit("other", UserWarning, "source.py", 34)
        assert warnings.showwarning is display
    assert records == [{"category": "AccelerationLimitWarning", "message": "limit", "count": 1}]
    assert [(str(message), *args) for message, *args in displayed] == [
        ("limit", AccelerationLimitWarning, "source.py", 12, None, None),
        ("other", UserWarning, "source.py", 34, None, None),
    ]


@pytest.mark.parametrize("action", ["always", "default", "ignore"])
@pytest.mark.parametrize("duration", [1, 1800])
def test_warning_counts_order_filters_and_non_fyst_passthrough(
    tmp_path, quickstart, monkeypatch, action, duration,
):
    schedule = generate.generate_observing_timeline
    reconstruct = generate.reconstruct_trajectories

    def schedule_with_warnings(*args):
        for _ in range(2):
            warnings.warn("repeated limit", AccelerationLimitWarning)
        warnings.warn("outside FYST", UserWarning)
        warnings.warn("second limit", VelocityLimitWarning)
        return schedule(*args)

    def reconstruct_with_warning(*args):
        warnings.warn("repeated limit", AccelerationLimitWarning)
        return reconstruct(*args)

    monkeypatch.setattr(generate, "generate_observing_timeline", schedule_with_warnings)
    monkeypatch.setattr(generate, "reconstruct_trajectories", reconstruct_with_warning)
    with warnings.catch_warnings(record=True) as displayed:
        warnings.simplefilter("always")
        warnings.simplefilter(action, AccelerationLimitWarning)
        original_handler = warnings.showwarning
        original_filters = warnings.filters[:]
        report = generate.generate_trajectory(
            quickstart.example_patch("ces"), tmp_path,
            start=quickstart.DEFAULT_START["ces"], duration_s=duration,
        )
        assert warnings.showwarning is original_handler
        assert warnings.filters == original_filters
    expected = []
    if action != "ignore":
        count = (2 if action == "always" else 1) + (duration == 1800)
        expected.append({"category": "AccelerationLimitWarning", "message": "repeated limit", "count": count})
    expected.append({"category": "VelocityLimitWarning", "message": "second limit", "count": 1})
    assert report["status"] == ("no_science" if duration == 1 else "complete")
    assert report["fyst_warnings"] == expected
    assert json.loads((tmp_path / "report.json").read_text())["fyst_warnings"] == expected
    assert len(displayed) == sum(item["count"] for item in expected) + 1
    assert sum(item.category is UserWarning and str(item.message) == "outside FYST"
               for item in displayed) == 1
