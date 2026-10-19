"""把已有 FYST trajectory ECSV 写为 TOAST Observation HDF5。

复制并整理自用户本地的 legacy FYST example（原文件 SHA-256 ca2c9b3e…，详见本地 docs/UPSTREAM.json）。
保留原生 block、采样点、五个示意 detectors 和 TOAST SaveHDF5 schema。
此脚本独立于 SimFYST；不生成新运动，也不实现 SimFYST 的 H5 reader。
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import gettempdir


def read_site_metadata(schedule):
    """Read name, latitude (deg), longitude (deg), height (m) without snapping."""
    from astropy import units as u
    from astropy.table import Table

    meta = Table.read(schedule, format="ascii.ecsv").meta
    return (
        str(meta["site_name"]),
        float(meta["site_lat"].to_value(u.deg)),
        float(meta["site_lon"].to_value(u.deg)),
        float(meta["site_alt"].to_value(u.m)),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory", type=Path, help="旧 example 第 02 步生成的 fyst_trajectory.ecsv"
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        help="原生 timeline ECSV；默认 trajectory 同目录的 schedule.ecsv",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="默认写到本 repo 的 output/<case>/hdf5"
    )
    parser.add_argument(
        "--iers-file",
        type=Path,
        help="可选：离线 IERS-A 表；默认使用 Astropy 自带的数据",
    )
    parser.add_argument(
        "--roll-deg",
        type=float,
        default=0.0,
        help="诊断用固定 local boresight roll，默认 0 deg",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖本轮同名 H5 和清单；不删除其他文件",
    )
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()
    source = args.trajectory.expanduser().resolve()
    schedule = (
        args.schedule.expanduser().resolve()
        if args.schedule
        else source.with_name("schedule.ecsv")
    )
    for path in (source, schedule):
        if not path.is_file():
            parser.error(f"找不到输入文件：{path}")
    if args.iers_file is not None and not args.iers_file.expanduser().is_file():
        parser.error(f"找不到 IERS 文件：{args.iers_file}")

    # 只配置本 CLI 的串行执行；package 不修改用户的 MPI / 线程环境。
    os.environ["MPI_DISABLE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        cache = Path(gettempdir()) / "fyst-to-toast" / key.lower()
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(cache))

    import numpy as np
    from astropy import units as u
    from astropy.coordinates import solar_system_ephemeris
    from astropy.table import QTable
    from astropy.utils import iers

    from fyst_to_toast.trajectory_h5 import write_trajectory_blocks

    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "error"
    if args.iers_file:
        iers.earth_orientation_table.set(
            iers.IERS_A.open(str(args.iers_file.expanduser().resolve()))
        )
    solar_system_ephemeris.set("builtin")

    if not np.isfinite(args.roll_deg):
        parser.error("roll 必须是有限数值")
    samples = QTable.read(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    schedule_hash = hashlib.sha256(schedule.read_bytes()).hexdigest()
    if samples.meta.get("source_schedule_sha256") != schedule_hash:
        parser.error(
            "trajectory 与 schedule 的 SHA-256 不匹配；请使用同一轮生成的输入。"
        )
    case = samples.meta.get("case", "")
    if not isinstance(case, str) or re.fullmatch(r"[A-Za-z0-9_]+", case) is None:
        parser.error(
            "trajectory metadata.case 必须是可用于文件名的 pattern 名称，例如 ces / daisy"
        )
    if not samples.meta.get("blocks") or not len(samples):
        parser.error("trajectory 没有可写出的 blocks / samples")

    # 在写入前检查 block 清单完整覆盖输入，且不重新分段。
    offset, names = 0, []
    for block in samples.meta["blocks"]:
        if block["first"] != offset or not offset < block["last"] <= len(samples):
            parser.error("blocks 的 [first,last) 必须连续覆盖输入数组且不能为空")
        offset = block["last"]
        names.append(
            f"fyst_{case}_{block['scan_index']:03d}_{block['subscan_index']:03d}.h5"
        )
    if offset != len(samples) or len(set(names)) != len(names):
        parser.error("blocks 必须覆盖所有样本，scan/subscan 组合不能产生重复文件名")
    if not np.all(np.diff(samples["timestamp_unix_s"].to_value(u.s)) > 0):
        parser.error("timestamps 必须严格递增")

    folder = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else Path(__file__).resolve().parents[1] / "output" / case / "hdf5"
    )
    expected_paths = [
        folder / name for name in [*names, "files.json", "writer_report.json"]
    ]
    if not args.overwrite and any(path.exists() for path in expected_paths):
        parser.error(
            "输出已存在；请指定新的 --output-dir，或使用 --overwrite 明确覆盖。"
        )
    if source in expected_paths or schedule in expected_paths:
        parser.error("输出不能覆盖输入文件")
    folder.mkdir(parents=True, exist_ok=True)

    site_name, latitude, longitude, height = read_site_metadata(schedule)
    files = write_trajectory_blocks(
        samples,
        site_name=site_name,
        latitude_deg=latitude,
        longitude_deg=longitude,
        height_m=height,
        output_dir=folder,
        source_sha256=source_hash,
        roll_deg=args.roll_deg,
    )
    (folder / "files.json").write_text(json.dumps(files, indent=2) + "\n")
    report = {
        "status": "written",
        "schema": "TOAST Observation HDF5",
        "case": case,
        "observations": len(files),
        "samples": len(samples),
        "files": files,
        "source_sha256": source_hash,
        "schedule_sha256": schedule_hash,
        "roll_deg": args.roll_deg,
        "detectors_per_observation": 5,
        "detdata_saved": [],
        "interval_policy": "rebuild from original FYST flags",
        "iers_source": "explicit file" if args.iers_file else "Astropy bundled data",
        "iers_sha256": (
            hashlib.sha256(args.iers_file.expanduser().read_bytes()).hexdigest()
            if args.iers_file
            else None
        ),
    }
    (folder / "writer_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"写入完成：{len(files)} blocks → {len(files)} TOAST H5，{len(samples)} samples。"
    )
    print("仪器：5 个示意 detectors；没有写入 detector signal。")
    print(f"输出目录：{folder}")
    print(f"本轮文件清单：{folder / 'files.json'}")


if __name__ == "__main__":
    main()
