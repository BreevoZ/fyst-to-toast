"""只读验收：用原生 LoadHDF5 读回 writer 输出，并与输入 trajectory 逐点核对。

这不是 SimFYST reader，也不生成新运动、intervals 或 detector pointing。
成功时向 stdout 输出 JSON；失败时退出并显示不一致的字段。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import gettempdir


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory", type=Path, help="writer 使用的原始 trajectory ECSV"
    )
    parser.add_argument(
        "output_dir", type=Path, help="包含 files.json 的 writer 输出目录"
    )
    args = parser.parse_args()
    source = args.trajectory.expanduser().resolve()
    folder = args.output_dir.expanduser().resolve()
    for path in (source, folder / "files.json"):
        if not path.is_file():
            parser.error(f"找不到输入文件：{path}")

    # 必须在数值库及 TOAST 导入前配置；不启动 MPI 或额外数值线程。
    for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["TOAST_LOGLEVEL"] = "WARNING"
    for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        cache = Path(gettempdir()) / "fyst-toast" / key.lower()
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(cache))

    import h5py
    import numpy as np
    from astropy import units as u
    from astropy.table import QTable
    from toast import Data, ops
    from toast import qarray as qa
    from toast.mpi import Comm
    from toast.observation import default_values as defaults

    from fyst_toast.fyst_flags import SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND

    samples = QTable.read(source)
    blocks = samples.meta["blocks"]
    case = samples.meta["case"]
    require(bool(blocks) and len(samples) > 0, "输入必须含非空 blocks / samples")
    expected_names = [
        f"fyst_{case}_{b['scan_index']:03d}_{b['subscan_index']:03d}" for b in blocks
    ]
    names = json.loads((folder / "files.json").read_text())
    require(
        names == [f"{name}.h5" for name in expected_names],
        "文件清单与 native blocks 不符",
    )
    require(len(set(names)) == len(names), "文件清单含重复项")
    paths = [folder / name for name in names]
    source_hash = file_sha256(source)
    original_hashes = {path: file_sha256(path) for path in paths}

    # LoadHDF5(detdata=[]) 不载入 detector 数据；另查文件本身确实没有此数据。
    for path in paths:
        with h5py.File(path, "r") as saved:
            require(len(saved["detdata"]) == 0, f"{path.name}: 文件中不应包含 detdata")

    data = Data(Comm(world=None))
    try:
        ops.LoadHDF5(
            files=[str(p) for p in paths], detdata=[], force_serial=True
        ).apply(data)
        observations = {obs.name: obs for obs in data.obs}
        require(len(data.obs) == len(blocks), "Observation 数量与 native blocks 不符")
        require(
            set(observations) == set(expected_names),
            "Observation 名称与 native blocks 不符",
        )
        reports = []
        offset = 0
        for block, name in zip(blocks, expected_names):
            require(block["first"] == offset, f"{name}: 输入 block 切片不连续")
            offset = block["last"]
            part = samples[block["first"] : block["last"]]
            obs = observations[name]
            require(obs.n_all_samples == len(part) > 0, f"{name}: 样本数不符")
            require(len(obs.all_detectors) == 5, f"{name}: 应有五个示意 detectors")
            require(len(obs.detdata) == 0, f"{name}: 读回后不应有 detdata")
            info = json.loads(obs["trajectory_info"])
            require(info["block"] == block, f"{name}: block metadata 不符")
            require(info["source_sha256"] == source_hash, f"{name}: 输入文件 hash 不符")
            require(info["case"] == case, f"{name}: pattern 名称不符")
            require(
                info["start_utc"] == samples.meta["start_utc"],
                f"{name}: start_utc 不符",
            )
            np.testing.assert_equal(
                obs.telescope.focalplane.sample_rate.to_value(u.Hz),
                1 / block["native_timestep_s"],
            )

            arrays = {key: np.asarray(obs.shared[key].data) for key in obs.shared}
            recorded = json.loads(obs["shared_array_sha256"])
            require(
                len(arrays) == 17 and set(arrays) == set(recorded),
                f"{name}: shared 字段不符",
            )
            for key, values in arrays.items():
                actual = hashlib.sha256(
                    np.ascontiguousarray(values).tobytes()
                ).hexdigest()
                require(actual == recorded[key], f"{name}/{key}: 保存前后 hash 不符")
                require(np.isfinite(values).all(), f"{name}/{key}: 含非有限值")

            # 独立比较原始 ECSV，避免仅相信 writer 自己保存的 hash。
            expected_arrays = {
                "times": part["timestamp_unix_s"].to_value(u.s),
                "azimuth": part["az_deg"].to_value(u.rad),
                "elevation": part["el_deg"].to_value(u.rad),
                "fyst_time_s": part["time_s"].to_value(u.s),
                "fyst_block_time_s": part["block_time_s"].to_value(u.s),
                "fyst_az_deg": part["az_deg"].to_value(u.deg),
                "fyst_el_deg": part["el_deg"].to_value(u.deg),
                "fyst_az_velocity_deg_s": part["az_velocity"].to_value(u.deg / u.s),
                "fyst_el_velocity_deg_s": part["el_velocity"].to_value(u.deg / u.s),
                "fyst_scan_flag": np.asarray(part["scan_flag"]),
                "scan_index": np.asarray(part["scan_index"]),
                "subscan_index": np.asarray(part["subscan_index"]),
            }
            for key, expected in expected_arrays.items():
                np.testing.assert_array_equal(
                    arrays[key], expected, err_msg=f"{name}/{key}"
                )
            require(np.all(np.diff(arrays["times"]) > 0), f"{name}: 时间不严格递增")
            np.testing.assert_array_equal(arrays["scan_index"], block["scan_index"])
            np.testing.assert_array_equal(
                arrays["subscan_index"], block["subscan_index"]
            )

            # FYST enum 的各类别单独验收；TOAST 是 bitmask，不直接复制 enum。
            raw = np.asarray(part["scan_flag"])
            science = raw == SCAN_FLAG_SCIENCE
            turnaround = raw == SCAN_FLAG_TURNAROUND
            for mask, value in (
                (science, 0),
                (turnaround, defaults.shared_mask_unstable_scanrate),
                (~(science | turnaround), defaults.shared_mask_invalid),
            ):
                np.testing.assert_array_equal(arrays["flags"][mask], value)

            for key in ("boresight_azel", "boresight_radec"):
                require(
                    arrays[key].shape == (len(part), 4),
                    f"{name}/{key}: quaternion shape 不符",
                )
                np.testing.assert_allclose(
                    np.linalg.norm(arrays[key], axis=1), 1, rtol=0, atol=1e-12
                )
            lon, lat, _ = qa.to_lonlat_angles(arrays["boresight_azel"])
            az_error = (
                np.degrees(-lon) - part["az_deg"].to_value(u.deg) + 180
            ) % 360 - 180
            el_error = np.degrees(lat) - part["el_deg"].to_value(u.deg)
            max_bore_error = float(
                max(np.max(np.abs(az_error)), np.max(np.abs(el_error)))
            )
            require(max_bore_error < 1e-9, f"{name}: boresight Az/El 与输入不符")
            reports.append(
                {
                    "name": name,
                    "samples": len(part),
                    "boresight_max_error_deg": max_bore_error,
                    "fyst_flag_values_observed": np.unique(raw).tolist(),
                }
            )

        require(offset == len(samples), "native blocks 没有覆盖全部输入")
        require(
            all(file_sha256(p) == original_hashes[p] for p in paths),
            "H5 在验证期间发生变化",
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "case": case,
                    "observations": reports,
                    "samples": len(samples),
                    "shared_arrays_per_observation": 17,
                    "input_arrays_compared_exactly": list(expected_arrays),
                    "flags_mapping_checked": True,
                    "detectors_per_observation": 5,
                    "detdata": [],
                    "h5_unchanged": True,
                },
                indent=2,
            )
        )
    finally:
        data.clear()


if __name__ == "__main__":
    main()
