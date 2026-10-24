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
spec = importlib.util.spec_from_file_location("downstream_example", ROOT / "examples/04_downstream_smoke.py")
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
