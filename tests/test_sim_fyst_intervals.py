"""IV1 compatibility names, angular metadata, and preserved motion intervals."""

import os

for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from astropy import units as u
from astropy.utils import iers
from toast.intervals import IntervalList
from toast.observation import default_values as defaults

from fyst_to_toast.sim_fyst import SimFYST, _scan_metadata, _throw_sample_spans
from test_sim_fyst_trajectory import SAMPLES, caller, data

iers.conf.auto_download = False
ROOT = Path(__file__).resolve().parents[1]
EMPTY_NAMES = {
    defaults.throw_leftright_interval, defaults.throw_rightleft_interval,
    defaults.scan_leftright_interval,
    defaults.scan_rightleft_interval, defaults.turn_leftright_interval,
    defaults.turn_rightleft_interval, defaults.elnod_interval,
    defaults.sun_up_interval, defaults.sun_close_interval,
}


@pytest.mark.parametrize("case", ["fixture_ces", "fixture_daisy", "ces", "daisy", "pong"])
def test_every_observation_intervals_and_metadata(case, data):
    if case.startswith("fixture_"):
        name = "fyst_ces_000_001.h5" if case == "fixture_ces" else "fyst_daisy_003_000.h5"
        paths = [SAMPLES / name]
    else:
        manifest = ROOT / "output/generated" / case / "hdf5/files.json"
        if not manifest.is_file():
            pytest.skip(f"Optional generated {case} inputs are unavailable")
        paths = [manifest.parent / name for name in json.loads(manifest.read_text())]
    with h5py.File(paths[0], "r") as saved:
        telescope = caller(SimpleNamespace(site=dict(saved["instrument"].attrs)))
    op = SimFYST(telescope=telescope, trajectory_files=paths)
    op.apply(data)
    assert len(data.obs) == len(paths)
    assert any(len(obs.intervals[defaults.throw_interval]) > 0 for obs in data.obs)
    by_name = {path.stem: path for path in paths}
    for obs in data.obs:
        with h5py.File(by_name[obs.name], "r") as saved:
            t, az, el, raw = [saved[f"shared/{key}"][()] for key in
                              ("times", "azimuth", "elevation", "fyst_scan_flag")]
        assert len(EMPTY_NAMES) == 9
        assert set(obs.intervals) == EMPTY_NAMES | {
            "ALL_OBSERVATION_SAMPLES", "scanning", "turnaround", "fyst_invalid", "throw",
        }
        for name in EMPTY_NAMES:
            assert isinstance(obs.intervals[name], IntervalList)
            assert len(obs.intervals[name]) == 0
            assert obs.intervals[name].data.size == 0
        assert set(op.provides()["intervals"]) == set(obs.intervals) - {"ALL_OBSERVATION_SAMPLES"}
        # Independent span oracle, including timestamps, exclusive ends and bytes.
        for name, states in (("scanning", {1}), ("turnaround", {2}), ("fyst_invalid", {0, 3})):
            spans, start = [], None
            for index, state in enumerate([*raw, -1]):
                if state in states and start is None:
                    start = index
                elif state not in states and start is not None:
                    spans.append((start, index))
                    start = None
            expected = IntervalList(t, samplespans=spans).data
            actual = obs.intervals[name].data
            assert actual.dtype == expected.dtype
            assert actual.tobytes() == expected.tobytes()
        expected_flags = np.zeros(len(raw), dtype=np.uint8)
        expected_flags[raw == 2] = defaults.shared_mask_unstable_scanrate
        expected_flags[np.isin(raw, [0, 3])] = defaults.shared_mask_invalid
        assert obs.shared["flags"].data.tobytes() == expected_flags.tobytes()
        # Check actual TOAST intervals, not just the helper's return value.
        throws = [(iv.first, iv.last) for iv in obs.intervals[defaults.throw_interval]]
        assert_throw_partition(raw, throws)
        # Ask TOAST to convert midpoint starts, independently of the helper.
        expected_cuts = set()
        for turn in obs.intervals["turnaround"]:
            midpoint = 0.5 * (turn.start + turn.stop)
            if midpoint == t[-1]:
                expected_cuts.add(len(t) - 1)
            else:
                converted = IntervalList(t, timespans=[(midpoint, t[-1])])
                expected_cuts.add(converted[0].first)
        valid = np.isin(raw, [1, 2])
        for index in range(1, len(raw)):
            if valid[index - 1] != valid[index]:
                expected_cuts.add(index)
        expected_cuts.update([0, len(raw)])
        edges = sorted(expected_cuts)
        expected = [(first, last) for first, last in zip(edges[:-1], edges[1:])
                    if valid[first]]
        assert throws == expected

        # Include turnaround extrema, as SimGround's final AzimuthRanges does.
        good = np.isin(raw, [1, 2])
        if not good.any():
            good[:] = True
        az_valid, el_valid = np.unwrap(az)[good], el[good]
        expected = {
            "scan_el": (np.rad2deg(el_valid.mean()), u.deg),
            "scan_min_el": (el_valid.min(), u.rad),
            "scan_max_el": (el_valid.max(), u.rad),
            "scan_min_az": (az_valid.min(), u.rad),
            "scan_max_az": (az_valid.max(), u.rad),
        }
        for key, (value, unit) in expected.items():
            assert type(obs[key]) is u.Quantity
            assert obs[key].isscalar and obs[key].unit == unit
            assert obs[key].dtype == np.dtype(np.float64)
            assert obs[key].value == value


def assert_throw_partition(raw, spans):
    coverage = np.zeros(len(raw), dtype=int)
    previous_end = 0
    for first, last in spans:
        assert previous_end <= first < last <= len(raw)
        coverage[first:last] += 1
        previous_end = last
        segment = raw[first:last]
        assert np.all(np.isin(segment, [1, 2]))
        # At most one scan, with optional partial turnarounds on either side.
        scanning = np.flatnonzero(segment == 1)
        if len(scanning):
            assert np.all(segment[scanning[0]:scanning[-1] + 1] == 1)
    np.testing.assert_array_equal(coverage, np.isin(raw, [1, 2]).astype(int))
    assert not coverage[np.isin(raw, [0, 3])].any()


@pytest.mark.parametrize("raw,expected", [
    ([], []),
    ([0, 3], []),
    ([1, 1, 1], [(0, 3)]),
    ([2], [(0, 1)]),
    ([2, 2], [(0, 1), (1, 2)]),
    ([2, 2, 2], [(0, 1), (1, 3)]),
    ([1, 1, 2, 2], [(0, 3), (3, 4)]),
    ([1, 1, 2, 2, 1, 2], [(0, 3), (3, 5), (5, 6)]),
    ([2, 2, 1, 1, 2, 1], [(0, 1), (1, 5), (5, 6)]),
    ([1, 0, 2, 1, 3, 2, 2, 1], [(0, 1), (2, 3), (3, 4), (5, 6), (6, 8)]),
    ([1, 2, 3, 1, 2], [(0, 2), (3, 4), (4, 5)]),
    ([1, 0, 1], [(0, 1), (2, 3)]),
    ([0, 2, 2, 3], [(1, 2), (2, 3)]),
    ([0, 2, 2, 2, 2, 3], [(1, 3), (3, 5)]),
])
def test_throw_edge_rules(raw, expected):
    raw = np.array(raw, dtype=np.uint8)
    spans = _throw_sample_spans(np.arange(len(raw), dtype=float), raw)
    assert spans == expected
    assert_throw_partition(raw, spans)


@pytest.mark.parametrize("times,raw", [
    ([0, 1, 2, 3, 4, 5, 6], [1, 2, 2, 2, 2, 1, 1]),  # On a sample.
    ([0, 1, 2, 3, 4, 5, 6], [1, 2, 2, 2, 1, 1, 1]),  # Between samples.
    ([0, 1, 2, 2.9, 4, 5, 6], [1, 2, 2, 2, 2, 1, 1]),  # Time, not index.
    ([0, 1, 2, 2.999999, 4, 5, 6], [1, 2, 2, 2, 2, 1, 1]),
])
@pytest.mark.parametrize("epoch", [0, 1_700_000_000])
def test_throw_midpoints_match_toast_timespans(times, raw, epoch):
    times = np.array(times) + epoch
    raw = np.array(raw)
    turn_indices = np.flatnonzero(raw == 2)
    midpoint = 0.5 * (times[turn_indices[0]] + times[turn_indices[-1] + 1])
    expected = IntervalList(times, timespans=[
        (times[0], midpoint), (midpoint, times[-1]),
    ])
    spans = _throw_sample_spans(times, raw)
    assert spans == [(iv.first, iv.last) for iv in expected]
    assert_throw_partition(raw, spans)


@pytest.mark.parametrize("az_deg,limits", [
    ([358, 359, 0, 1], (358, 361)),
    ([-2, -1, 0, 1], (-2, 1)),
    ([1, 0, 359, 358], (-2, 1)),
    ([178, 179, -180, -179], (178, 181)),
])
def test_metadata_wrap_keeps_native_filterbin_branch(az_deg, limits):
    az = np.deg2rad(az_deg)
    original = az.tobytes()
    meta = _scan_metadata(az, np.full(4, np.deg2rad(50)), np.ones(4, dtype=int))
    np.testing.assert_allclose(
        [meta["scan_min_az"].to_value(u.deg), meta["scan_max_az"].to_value(u.deg)],
        limits, atol=1e-12, rtol=0,
    )
    # FilterBin uses this expression on the native, unshifted shared array.
    lo, hi = [meta[key].to_value(u.rad) for key in ("scan_min_az", "scan_max_az")]
    phase = (np.unwrap(az) - lo) / (hi - lo) * 2 - 1
    np.testing.assert_allclose([phase.min(), phase.max()], [-1, 1])
    assert az.tobytes() == original


def test_metadata_includes_turnaround_and_excludes_invalid_samples():
    meta = _scan_metadata(
        np.deg2rad([-50, 0, 10, 12, 0, 80]),
        np.deg2rad([0, 40, 40, 48, 40, 90]),
        np.array([0, 1, 1, 2, 1, 3]),
    )
    for key, value in {"scan_el": 42, "scan_min_el": 40, "scan_max_el": 48,
                       "scan_min_az": 0, "scan_max_az": 12}.items():
        assert meta[key].to_value(u.deg) == pytest.approx(value)


def test_metadata_without_valid_motion_uses_all_samples():
    meta = _scan_metadata(np.deg2rad([10, 20]), np.deg2rad([40, 60]), np.array([0, 3]))
    for key, value in {"scan_el": 50, "scan_min_el": 40, "scan_max_el": 60,
                       "scan_min_az": 10, "scan_max_az": 20}.items():
        assert meta[key].to_value(u.deg) == pytest.approx(value)
