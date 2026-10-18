"""把已有 FYST trajectory ECSV 写为 TOAST Observation HDF5。

复制并整理自本地 FYST example；来源与 SHA-256 见 UPSTREAM.json。
保留原生 block、采样点、五个示意 detectors 和 TOAST SaveHDF5 schema。
此脚本独立于 SimFYST；不生成新运动，也不实现 SimFYST 的 H5 reader。
"""

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from tempfile import gettempdir


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

    # 只配置本 example 的串行执行；package 不修改用户的 MPI / 线程环境。
    os.environ["MPI_DISABLE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        cache = Path(gettempdir()) / "fyst-toast" / key.lower()
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(cache))

    import numpy as np
    from astropy import units as u
    from astropy.coordinates import solar_system_ephemeris
    from astropy.table import QTable
    from astropy.utils import iers
    from fyst_trajectories.overhead import read_timeline
    from fyst_trajectories.trajectory import SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND
    from toast import Data, Observation, ops
    from toast import qarray as qa
    from toast.coordinates import azel_to_radec
    from toast.instrument import Focalplane, GroundSite, Session, Telescope
    from toast.instrument_sim import fake_hexagon_focalplane
    from toast.mpi import Comm
    from toast.observation import default_values as defaults

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

    # 1. 站点来自原生 timeline。这里的五个 detectors 是 TOAST 自带的示意 focalplane。
    fyst_site = read_timeline(schedule).site
    site = GroundSite(
        fyst_site.name,
        fyst_site.latitude * u.deg,
        fyst_site.longitude * u.deg,
        fyst_site.elevation * u.m,
    )
    data = Data(Comm(world=None))
    try:
        files = []
        for block in samples.meta["blocks"]:
            # 这是读取第二步原封保存的一块数组，不是重新决定如何划分 block。
            part = samples[block["first"] : block["last"]]
            times = part["timestamp_unix_s"].to_value(u.s)
            az, el = part["az_deg"].to_value(u.rad), part["el_deg"].to_value(u.rad)
            dt = np.diff(part["block_time_s"].to_value(u.s))
            np.testing.assert_allclose(
                dt, block["native_timestep_s"], rtol=0, atol=1e-9
            )
            rate = (1 / block["native_timestep_s"]) * u.Hz
            full = fake_hexagon_focalplane(n_pix=7, width=0.1 * u.deg, sample_rate=rate)
            fp = Focalplane(
                detector_data=full.detector_data[:5],
                sample_rate=rate,
                field_of_view=full.field_of_view,
            )
            telescope = Telescope("FYST-trajectory-demo", site=site, focalplane=fp)
            name = f"fyst_{case}_{block['scan_index']:03d}_{block['subscan_index']:03d}"
            session = Session(
                name=name,
                start=datetime.fromtimestamp(times[0], tz=timezone.utc),
                end=datetime.fromtimestamp(times[-1], tz=timezone.utc),
            )
            obs = Observation(
                data.comm, telescope, len(part), name=name, session=session
            )
            data.obs.append(obs)
            files.append(f"{name}.h5")

            # 2. TOAST 的方向约定：Az 顺时针，而 longitude 逆时针。
            # 固定 roll 是 IO/pointing 诊断设置，不是已经验收的 FYST Nasmyth 姿态。
            bore = qa.from_lonlat_angles(-az, el, np.zeros_like(el))
            bore = qa.mult(
                bore, qa.rotation(np.array([0.0, 0.0, 1.0]), np.radians(args.roll_deg))
            )
            radec = azel_to_radec(site, times, bore, use_ephem=True)
            position, velocity = site.position_velocity(times)

            # 3. FYST enum → TOAST bitmask；原始枚举另存，不会丢掉换向点。
            original_flags = np.asarray(part["scan_flag"])
            flags = np.zeros(len(part), dtype=np.uint8)
            flags[original_flags == SCAN_FLAG_TURNAROUND] = (
                defaults.shared_mask_unstable_scanrate
            )
            flags[
                ~np.isin(original_flags, [SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND])
            ] = defaults.shared_mask_invalid
            arrays = {
                "times": times,
                "azimuth": az,
                "elevation": el,
                "flags": flags,
                "boresight_azel": bore,
                "boresight_radec": radec,
                "position": position,
                "velocity": velocity,
                "fyst_time_s": part["time_s"].to_value(u.s),
                "fyst_block_time_s": part["block_time_s"].to_value(u.s),
                "fyst_az_deg": part["az_deg"].to_value(u.deg),
                "fyst_el_deg": part["el_deg"].to_value(u.deg),
                "fyst_az_velocity_deg_s": part["az_velocity"].to_value(u.deg / u.s),
                "fyst_el_velocity_deg_s": part["el_velocity"].to_value(u.deg / u.s),
                "fyst_scan_flag": original_flags,
                "scan_index": np.asarray(part["scan_index"]),
                "subscan_index": np.asarray(part["subscan_index"]),
            }
            for key, values in arrays.items():
                assert np.isfinite(values).all()
                obs.shared.create_column(key, shape=values.shape, dtype=values.dtype)
                obs.shared[key].set(values, fromrank=0)

            # 4. 以下是适配说明与核对信息，不是 FYST/TOAST 的新接口。
            obs["trajectory_info"] = json.dumps(
                {
                    "case": case,
                    "source_sha256": source_hash,
                    "start_utc": samples.meta["start_utc"],
                    "block": block,
                    "roll_deg": args.roll_deg,
                    "sample_rate_hz": rate.to_value(u.Hz),
                    "fyst_flag_values": {
                        "scanning": SCAN_FLAG_SCIENCE,
                        "turnaround": SCAN_FLAG_TURNAROUND,
                    },
                    "adapter_note": "One native block per Observation; no resampling or motion generation",
                    "interval_policy": "Rebuild sample-index intervals from flags after loading",
                }
            )
            obs["shared_array_sha256"] = json.dumps(
                {
                    key: hashlib.sha256(
                        np.ascontiguousarray(value).tobytes()
                    ).hexdigest()
                    for key, value in arrays.items()
                }
            )

        # 5. 仍由原生 SaveHDF5 写 schema；一个 block 一文件，detdata=[]。
        ops.SaveHDF5(
            volume=str(folder), volume_index=None, detdata=[], force_serial=True
        ).apply(data)
        if not all((folder / name).is_file() for name in files):
            raise RuntimeError("SaveHDF5 没有产生全部预期文件")
        (folder / "files.json").write_text(json.dumps(files, indent=2) + "\n")
        report = {
            "status": "written",
            "schema": "TOAST Observation HDF5",
            "case": case,
            "observations": len(data.obs),
            "samples": len(samples),
            "files": files,
            "source_sha256": source_hash,
            "schedule_sha256": schedule_hash,
            "roll_deg": args.roll_deg,
            "detectors_per_observation": 5,
            "detdata_saved": [],
            "interval_policy": "rebuild from original FYST flags",
            "iers_source": "explicit file"
            if args.iers_file
            else "Astropy bundled data",
            "iers_sha256": (
                hashlib.sha256(args.iers_file.expanduser().read_bytes()).hexdigest()
                if args.iers_file
                else None
            ),
        }
        (folder / "writer_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"写入完成：{len(data.obs)} blocks → {len(files)} TOAST H5，{len(samples)} samples。"
        )
        print("仪器：5 个示意 detectors；没有写入 detector signal。")
        print(f"输出目录：{folder}")
        print(f"本轮文件清单：{folder / 'files.json'}")
    finally:
        data.clear()


if __name__ == "__main__":
    main()
