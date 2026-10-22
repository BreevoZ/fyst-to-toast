"""Roll convention checks; real FYST rotation is used only as a test oracle."""

import os

for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from toast import qarray as qa
from toast.instrument_coords import xieta_to_quat

from fyst_to_toast.trajectory_h5 import _boresight_with_roll


def test_zero_roll_skips_rotation(monkeypatch):
    az, el = np.deg2rad([0.0, 90.0, 270.0]), np.deg2rad([30.0, 50.0, 80.0])
    q0 = qa.from_lonlat_angles(-az, el, np.zeros(3))
    rotation = Mock(wraps=qa.rotation)
    mult = Mock(wraps=qa.mult)
    monkeypatch.setattr(qa, "rotation", rotation)
    monkeypatch.setattr(qa, "mult", mult)
    assert _boresight_with_roll(az, el, np.zeros(3)).tobytes() == q0.tobytes()
    rotation.assert_not_called()
    mult.assert_not_called()
    _boresight_with_roll(az, el, np.array([0.0, 0.2, 0.0]))
    rotation.assert_called_once()
    mult.assert_called_once()
    assert rotation.call_args.args[1].shape == (3,)


@pytest.mark.parametrize("port", ["right", "left"])
def test_fyst_off_axis_roll_contract(port):
    fyst = pytest.importorskip("fyst_trajectories", exc_type=ImportError)
    site = replace(fyst.get_fyst_site(), nasmyth_port=port)
    az_deg = np.array([0.0, 45.0, 180.0, 270.0, 359.0])
    el_deg = np.array([25.0, 45.0, 60.0, 80.0, 35.0])
    # Explicit test-only roll; RL-A generation must continue to supply zeros.
    roll = np.deg2rad(-site.nasmyth_sign * el_deg)
    bore = _boresight_with_roll(np.deg2rad(az_deg), np.deg2rad(el_deg), roll)
    detector = xieta_to_quat(np.sin(np.deg2rad(1.0)), 0.0, 0.0)
    actual = qa.rotate(qa.mult(bore, detector), [0.0, 0.0, 1.0])
    offset = fyst.InstrumentOffset(dx=60.0, dy=0.0)
    rotation = fyst.compute_focal_plane_rotation(el_deg, site=site, offset=offset)
    det_az, det_el = fyst.boresight_to_detector(az_deg, el_deg, offset, rotation)
    lon, lat = -np.deg2rad(det_az), np.deg2rad(det_el)
    expected = np.column_stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))
    )
    separation = np.arctan2(
        np.linalg.norm(np.cross(actual, expected), axis=1),
        np.sum(actual * expected, axis=1),
    )
    error_arcsec = np.rad2deg(separation) * 3600
    assert np.max(error_arcsec) < 1.0
    print(f"{port} Nasmyth: max off-axis error {np.max(error_arcsec):.6g} arcsec")
