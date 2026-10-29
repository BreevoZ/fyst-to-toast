"""DS1: two fixture Observations through native TOAST simulation and binning."""
import os

for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import solar_system_ephemeris
from toast import Data
from toast.mpi import Comm
from toast.ops import BinMap
from toast.ops.mapmaker_utils import CovarianceAndHits

from fyst_to_toast.sim_fyst import SimFYST

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("downstream_example", ROOT / "examples/downstream_maps.py")
walk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(walk)


@pytest.mark.parametrize("mode,noise", [("I", False), ("IQU", True)])
def test_multi_block_downstream(mode, noise, tmp_path):
    paths = [ROOT / "data/fyst_ces_000_001.h5", ROOT / "data/fyst_daisy_003_000.h5"]
    data = Data(Comm(world=None))
    nside = 32
    # Installed TOAST 3.0.5 observation.py:62-68,100-110 defines nonscience=15;
    # ops/mapmaker_binning.py:82-85 and mapmaker_utils/mapmaker_utils.py:1039-1041
    # use it by default. Thus turnaround=4 IS excluded. Detector masks default 7.
    # CovarianceAndHits:1060-1062 defaults rcond_threshold to 1e-8.
    assert BinMap().shared_flag_mask == CovarianceAndHits().shared_flag_mask == walk.SHARED_MASK == 15
    assert CovarianceAndHits().rcond_threshold == walk.RCOND == 1e-8
    try:
        with solar_system_ephemeris.set("builtin"):
            SimFYST(telescope=walk.caller_telescope(paths[0]), trajectory_files=paths).apply(data)
            assert len(data.obs) == 2
            sky_file = tmp_path / "input.fits"
            sky = walk.synthetic_sky(nside, sky_file, mode)
            result = walk.run_chain(data, sky_file, nside, mode=mode, noise=noise)
        assert np.isfinite(result["map"]).all() and np.isfinite(result["hits"]).all()
        # The chain independently checks EVERY pixel's hit histogram. Equality to
        # good samples * detector count assumes zero detector flags / cuts, all
        # detectors selected, full observation view and valid unmasked pixels.
        expected = sum(np.count_nonzero((obs.shared["flags"].data & 15) == 0)
                       * len(obs.local_detectors) for obs in data.obs)
        assert int(result["hits"].sum()) == expected
        assert any(np.any(obs.shared["flags"].data & 4) for obs in data.obs)
        if mode == "I":
            hit = result["hits"] > 0
            # ScanHealpixMap:167-177 stores input sky as float32; binning sums in
            # float64. 5e-7 relative + 1e-10 K covers float32 conversion and sums,
            # not a beam / interpolation error: the native scan is pixel-constant.
            np.testing.assert_allclose(result["map"][0, hit], sky[0, hit], rtol=5e-7, atol=1e-10)
        for row in result["per_observation"]:
            if "ces" in row["name"]:
                # Fixed-El CES sweeps a broad RA/Dec band; a subscan centroid
                # need not be the patch center. This fixture is also from a
                # legacy FYST 0.8.0 run, not today's generate.py configuration.
                continue
            # Daisy: half-diagonal of 2x2 deg patch + half focalplane width +
            # HEALPix pixel radius. The 39.8 s fixture is only part of a scan.
            tolerance = np.sqrt(2) + 0.2 + np.rad2deg(walk.hp.max_pixrad(nside))
            assert walk.separation(row["center_deg"], walk.PATCHES["daisy"]) < tolerance
    finally:
        data.clear()


@pytest.mark.parametrize("port,sign", [("right", 1), ("left", -1), ("cassegrain", 0)])
@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("mode", ["corrected", "schedule"])
def test_schedule_roll_copy(tmp_path, port, sign, modern, mode):
    from astropy import units as u
    from astropy.table import QTable

    roll = -sign * np.array([40., 60.])
    original_angles = roll if modern else np.array([12., 13.])
    table = QTable({"el": [40., 60.] * u.deg,
                    "boresight_angle": original_angles * u.deg},
                   meta={"site_nasmyth_port": port})
    if modern:
        table.meta["boresight_angle_convention"] = "toast_roll"
    source = tmp_path / "source.ecsv"
    table.write(source, format="ascii.ecsv")
    original = source.read_bytes()
    path, info = walk.prepare_simground_schedule(source, tmp_path / "out", mode)
    result = QTable.read(path, format="ascii.ecsv")
    unchanged = modern or mode == "schedule"
    expected = original_angles if unchanged else roll
    np.testing.assert_array_equal(result["boresight_angle"].to_value(u.deg), expected)
    assert info["boresight_angle_deg"] == list(expected)
    assert info["original_boresight_angle_deg"] == list(original_angles)
    assert (path == source) == unchanged
    assert source.read_bytes() == original
    if modern:
        assert info["action"] == (
            "boresight_angle is already the TOAST roll "
            "(boresight_angle_convention = toast_roll); unchanged"
        )
    elif mode == "corrected":
        assert info["action"] == "replaced boresight_angle with -nasmyth_sign * el"
    else:
        assert info["action"] == "original schedule"
    if unchanged:
        assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("port", [None, "unknown"])
def test_schedule_roll_requires_port(tmp_path, port):
    from astropy import units as u
    from astropy.table import QTable

    table = QTable({"el": [40.] * u.deg, "boresight_angle": [12.] * u.deg})
    if port is not None:
        table.meta["site_nasmyth_port"] = port
    source = tmp_path / "source.ecsv"
    table.write(source, format="ascii.ecsv")
    with pytest.raises(ValueError, match="site_nasmyth_port"):
        walk.prepare_simground_schedule(source, tmp_path / "out")


def test_comparison_common_conditioning_and_robust_residual():
    reference = np.zeros((3, 4))
    fyst = {"hits": np.array([1, 2, 3, 4]), "rcond": np.array([0.1, 0.01, 0.1, 2e-8]),
            "map": np.tile([2., 100., 200., 300.], (3, 1)), "center_deg": [0., 0.],
            "noiseless": {"map": np.tile([1e-12, 1., 2., 3.], (3, 1))}}
    ground = {**fyst, "rcond": np.array([0.1, 0.1, 0.001, 0.1]),
              "map": np.tile([3., 400., 500., 600.], (3, 1))}
    result = walk.compare_maps(fyst, ground, reference, True, compare_rcond=0.01)
    assert result["comparison_pixels"] == 1  # Exact threshold and either bad side excluded.
    assert result["noiseless_max_residual_simfyst_K"] == [1e-12] * 3
    assert result["noiseless_max_residual_simground_K"] == [1e-12] * 3
    assert result["noise_rms_simfyst_common_K"] == [2.] * 3
    assert result["noise_rms_simground_common_K"] == [3.] * 3
    assert result["noisy_difference_rms_common_K"] == [1.] * 3
    np.testing.assert_allclose(result["noise_robust_simfyst_common_K"], [1.4826 * 2] * 3)
    legacy = result["all_binned_pixels"]
    assert legacy["label"] == "all binned pixels"
    assert legacy["good_pixels_simfyst_simground_common"] == [4, 4, 4]
    assert legacy["noiseless_max_residual_simfyst_K"] == [3.] * 3
    np.testing.assert_allclose(legacy["noise_robust_simfyst_common_K"], [1.4826 * 150] * 3)
    with pytest.raises(ValueError, match="No common pixels"):
        walk.compare_maps(fyst, ground, reference, True, compare_rcond=0.5)
