"""单个 native trajectory block → caller Telescope → TOAST Observation H5。"""

import argparse
import json
import os
from pathlib import Path
from tempfile import gettempdir

# 必须在 NumPy / TOAST 导入前设置；library 本身不修改进程环境。
for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
for key, folder in (("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "cache")):
    cache = Path(gettempdir()) / "fyst-to-toast" / folder
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(key, str(cache))

from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.utils import iers
from toast import Data
from toast.instrument import GroundSite, Telescope
from toast.instrument_sim import fake_boresight_focalplane
from toast.mpi import Comm
from toast.ops import SaveHDF5

from fyst_to_toast.sim_fyst import SimFYST
from fyst_to_toast.trajectory_h5 import _load_trajectory_block


def field_unit(key):
    """Shared arrays 没有单位属性；此处显示契约单位，detdata 则自带 units。"""
    if key == "times" or key.endswith("_time_s"):
        return "s"
    if key in ("azimuth", "elevation"):
        return "rad"
    if key.endswith("_deg_s"):
        return "deg/s"
    if key.endswith("_deg"):
        return "deg"
    if key == "position":
        return "km"
    if key == "velocity":
        return "km/s"
    return "1 (quaternion)" if key.startswith("boresight_") else "1 (enum/index)"


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory_file",
        type=Path,
        nargs="?",
        default=repo / "data/fyst_ces_000_001.h5",
    )
    parser.add_argument("--output-dir", type=Path, default=repo / "output/simfyst")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    iers.conf.auto_download = False
    with solar_system_ephemeris.set("builtin"):
        # 仅从来源取站点坐标；两个 detectors 和 10 Hz nominal rate 由 caller 定义。
        block = _load_trajectory_block(args.trajectory_file)
        site = GroundSite(
            "FYST-caller-site",
            block.source_site["lat"],
            block.source_site["lon"],
            block.source_site["height"],
        )
        telescope = Telescope(
            "FYST-caller-demo",
            site=site,
            focalplane=fake_boresight_focalplane(n_pix=1, sample_rate=10 * u.Hz),
        )
        folder = args.output_dir.expanduser().resolve() / block.observation_name
        if folder.exists() and not args.overwrite:
            parser.error(
                f"Output exists: {folder}; use --overwrite to replace the saved H5"
            )
        data = Data(Comm(world=None))
        try:
            SimFYST(
                telescope=telescope, trajectory_file=str(args.trajectory_file)
            ).apply(data)
            obs = data.obs[0]
            print(f"Observation: {obs.name}; N={obs.n_local_samples}")
            print(f"Telescope/site: {obs.telescope.name} / {obs.telescope.site.name}")
            print(
                f"Detectors ({len(obs.local_detectors)}): {', '.join(obs.local_detectors)}"
            )
            print("Shared fields (contract units):")
            for key in obs.shared:
                array = obs.shared[key].data
                print(
                    f"  {key:<25} {array.dtype!s:<8} {array.shape!s:<12} {field_unit(key)}"
                )
            for key in obs.detdata:
                array = obs.detdata[key]
                shape = tuple(int(size) for size in array.shape)
                print(
                    f"Detdata {key}: {array.dtype}, {shape}, "
                    f"units={str(array.units) or '1'}; zero initialized"
                )
            print(f"Metadata: {', '.join(sorted(obs))}")
            info = json.loads(obs["simfyst_info"])
            print(f"Rate check: {json.dumps(info['rate_check'], sort_keys=True)}")
            print(f"flags/intervals: {info['flags/intervals']}")
            # 禁用 volume index，重跑只替换这一 Observation H5，不追加重复索引。
            SaveHDF5(volume=str(folder), volume_index=None, force_serial=True).apply(
                data
            )
            print(f"Saved: {folder / (obs.name + '.h5')}")
        finally:
            data.clear()


if __name__ == "__main__":
    main()
