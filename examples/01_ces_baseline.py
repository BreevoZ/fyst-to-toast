"""第一步：同一个短 CES，比较 SimFYST baseline 与 TOAST SimGround。

运行前在本 package 安装一次：python -m pip install --no-deps --no-build-isolation -e .
本脚本只创建指向和初始 detector arrays，不运行 sky / noise simulation。
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import gettempdir

# 本机 MPI 的空闲线程曾造成高 CPU；只在这个串行 example 中设置运行环境。
# 要在 import numpy / toast 之前设置，不能依靠导入后的环境修改关闭 MPI。
os.environ["MPI_DISABLE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# TOAST import 会间接加载 Matplotlib；使用可写缓存，不在 terminal 留缓存警告。
for key, subdir in (("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "cache")):
    cache = Path(gettempdir()) / "fyst-toast" / subdir
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(key, str(cache))

import numpy as np
from astropy import units as u
from astropy.utils import iers

import toast
from toast.instrument import GroundSite, Telescope
from toast.instrument_sim import fake_boresight_focalplane
from toast.ops import SimGround
from toast.schedule import GroundScan, GroundSchedule

from fyst_toast.sim_fyst import SimFYST

iers.conf.auto_download = False


def make_inputs():
    """用 TOAST 原生类定义一个小 CES，避免依赖其他 repo 的路径或输出。"""
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    scan = GroundScan(
        name="ces_baseline",
        start=start,
        stop=start + timedelta(seconds=30),
        az_min=62.25 * u.deg,
        az_max=65.47 * u.deg,
        el=42.79 * u.deg,
    )
    schedule = GroundSchedule(scans=[scan])
    site = GroundSite(
        "Cerro-Chajnantor", -22.9855235 * u.deg, -67.7403087 * u.deg, 5610 * u.m
    )
    focalplane = fake_boresight_focalplane(
        n_pix=1, sample_rate=20 * u.Hz, random_seed=123456
    )
    telescope = Telescope("FYST-baseline", site=site, focalplane=focalplane)
    return schedule, telescope


def run_operator(operator_class):
    """各自建立输入与空 Data，防止两次 apply 共享可变状态。"""
    schedule, telescope = make_inputs()
    data = toast.Data(toast.Comm(world=None, groupsize=1))
    operator = operator_class(
        telescope=telescope,
        schedule=schedule,
        weather=None,
        scan_rate_az=1 * u.deg / u.s,
        scan_accel_az=2 * u.deg / u.s**2,
        fix_rate_on_sky=True,
        randomize_phase=False,
    )
    try:
        operator.apply(data)
    except Exception:
        data.clear()
        raise
    return data


def compare(reference, candidate):
    """逐项比较实际生成的 Observation，不只比较 az/el 曲线。"""
    assert len(reference.obs) == len(candidate.obs) == 1
    left, right = reference.obs[0], candidate.obs[0]
    for name in ("name", "uid", "n_local_samples", "local_detectors"):
        assert getattr(left, name) == getattr(right, name), name
    for name in ("start", "end"):
        assert getattr(left.session, name) == getattr(right.session, name), name
    for key in ("scan_el", "scan_min_el", "scan_max_el", "scan_min_az", "scan_max_az"):
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)

    shared = []
    assert set(left.shared) == set(right.shared)
    for key in left.shared:
        a, b = left.shared[key].data, right.shared[key].data
        assert a.dtype == b.dtype, key
        assert np.isfinite(a).all(), key
        np.testing.assert_array_equal(a, b, err_msg=key)
        shared.append({"name": key, "shape": list(a.shape), "dtype": str(a.dtype), "equal": True})

    intervals = {}
    assert set(left.intervals) == set(right.intervals)
    for key in left.intervals:
        a = [(iv.first, iv.last, iv.start, iv.stop) for iv in left.intervals[key]]
        b = [(iv.first, iv.last, iv.start, iv.stop) for iv in right.intervals[key]]
        assert a == b, key
        intervals[key] = len(a)
    assert intervals["turnaround"] > 0  # 确实覆盖到了换向，避免只检查单向片段。

    detdata = []
    assert set(left.detdata) == set(right.detdata)
    for key in left.detdata:
        a, b = left.detdata[key], right.detdata[key]
        assert a.units == b.units, key
        assert np.isfinite(a.data).all(), key
        np.testing.assert_array_equal(a.data, b.data, err_msg=key)
        detdata.append({"name": key, "shape": list(a.data.shape), "equal": True})
    return {
        "status": "passed",
        "stage": "CES baseline; no FYST trajectory or HDF5 reader yet",
        "toast_version": toast.__version__,
        "mpi_enabled": toast.mpi.use_mpi,
        "observations": 1,
        "samples": int(left.n_local_samples),
        "sample_rate_hz": 20,
        "detectors": list(left.local_detectors),
        "shared": shared,
        "interval_counts": intervals,
        "detdata": detdata,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / "ces_baseline",
        help="检查报告目录；默认相对于脚本位置，不依赖 terminal 的 cwd",
    )
    args = parser.parse_args()
    reference = candidate = None
    try:
        print("[1/3] 原生 SimGround：30 s CES，20 Hz，2 detectors")
        reference = run_operator(SimGround)
        print("[2/3] 本地 SimFYST：相同输入，独立 Data")
        candidate = run_operator(SimFYST)
        print("[3/3] 比较 Observation、全部 shared / intervals / detdata")
        report = compare(reference, candidate)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"\nPASS：1 observation，{report['samples']} samples，2 detectors")
        print(f"{'shared 字段':<24} {'shape':<16} {'dtype':<10} 结果")
        for row in report["shared"]:
            print(f"{row['name']:<24} {str(row['shape']):<16} {row['dtype']:<10} 完全一致")
        print(f"interval lists：{len(report['interval_counts'])} 个，全部一致")
        print(f"detdata：{', '.join(row['name'] for row in report['detdata'])}，全部一致")
        print(f"报告：{report_path.resolve()}")
    finally:
        for data in (reference, candidate):
            if data is not None:
                data.clear()


if __name__ == "__main__":
    main()
