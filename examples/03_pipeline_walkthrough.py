"""FYST → ECSV → TOAST H5 → SimFYST 的逐步教学与 roll 对照。

本 example 可以 import fyst_trajectories：它是教学对照，不是 library；
只有 src/ 受运行库依赖边界约束。复用现有生成器、writer、loader 和 SimFYST，
不重写它们的调度、运动、flags 或 interval 逻辑。所有产物写在 repo 的 output/。
"""

import os

# 在 NumPy / TOAST / FYST 导入之前设置；只影响本教学进程及子进程。
for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DURATION_S = 6000.0


class Tee:
    """在终端和本次运行日志中同时保留教学输出。"""

    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def step(number, owner, files, content):
    print(f"\n第 {number} 步：{owner} · {files} · {content}")


def unit_for(key):
    """Shared arrays 不自带单位；这里明确打印 adapter 契约单位。"""
    if key == "times" or key.endswith("_time_s"):
        return "s"
    if key in ("azimuth", "elevation", "boresight_roll"):
        return "rad"
    if key.endswith("_deg_s"):
        return "deg/s"
    if key.endswith("_deg"):
        return "deg"
    if key == "position":
        return "km"
    if key == "velocity":
        return "km/s"
    if key.startswith("boresight_"):
        return "1 (quaternion)"
    return "1 (bitmask)" if key == "flags" else "1 (enum/index)"


def describe_arrays(arrays):
    descriptions = {}
    for key, value in arrays.items():
        descriptions[key] = {
            "dtype": str(value.dtype), "shape": list(value.shape),
            "unit_by_contract": unit_for(key),
        }
        print(f"  {key:<25} {str(value.dtype):<8} {str(value.shape):<13} {unit_for(key)}")
    return descriptions


def utc_ms(value):
    """Time 或 Unix 数值 → 毫秒 UTC；不截断存储的浮点样本时钟。"""
    from astropy.time import Time

    stamp = value.utc.replicate() if isinstance(value, Time) else Time(value, format="unix", scale="utc")
    stamp.precision = 3
    return stamp.isot + "+00:00"


def counts(array):
    import numpy as np

    keys, values = np.unique(array, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(keys, values, strict=True)}


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def roll_comparison(block, bore_quat, fp_quat, site, rotation_deg, schedule_angle):
    """完整 block 的 horizon 几何对照；正确路径来自实际 SimFYST Observation。"""
    import numpy as np
    from fyst_trajectories import InstrumentOffset, boresight_to_detector, compute_focal_plane_rotation
    from toast import qarray as qa
    from fyst_to_toast.trajectory_h5 import _boresight_with_roll

    arrays = block.arrays
    az, el = arrays["fyst_az_deg"], arrays["fyst_el_deg"]
    offset = InstrumentOffset(dx=60.0, dy=0.0, instrument_rotation=rotation_deg)
    theta = compute_focal_plane_rotation(el, site=site, offset=offset)
    ref_az, ref_el = boresight_to_detector(az, el, offset, theta)

    def vector(az_deg, el_deg):
        lon, lat = -np.deg2rad(az_deg), np.deg2rad(el_deg)
        return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))

    poses = {"Boresight": (az, el), "FYST reference": (ref_az, ref_el)}
    reference = vector(ref_az, ref_el)
    errors = {}
    rotations = {
        "Correct: -theta": None,
        "Zero roll": np.zeros_like(theta),
        "Wrong: +theta": theta,
        "Schedule angle": np.full_like(theta, schedule_angle),
    }
    for label, roll in rotations.items():
        bore = bore_quat if roll is None else _boresight_with_roll(
            arrays["azimuth"], arrays["elevation"], np.deg2rad(roll)
        )
        lon, lat, _ = qa.to_lonlat_angles(qa.mult(bore, fp_quat))
        det_az, det_el = -np.rad2deg(lon), np.rad2deg(lat)
        actual = vector(det_az, det_el)
        separation = np.arctan2(
            np.linalg.norm(np.cross(actual, reference), axis=1),
            np.sum(actual * reference, axis=1),
        )
        errors[label] = float(np.max(np.rad2deg(separation) * 3600))
        poses[label] = (det_az, det_el)
    assert errors["Correct: -theta"] < 1.0, errors
    return theta, poses, errors


COLORS = {
    "Boresight": "#303b4a", "FYST reference": "#13966a",
    "Correct: -theta": "#e4aa12", "Zero roll": "#d95056",
    "Wrong: +theta": "#8267bc", "Schedule angle": "#287fc1",
}


def draw_figures(folder, full, timeline, samples, blocks, obs_arrays, comparison, no_gif):
    """仅作图抽样；误差、存储、flags 和 intervals 始终使用完整样本。"""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    from PIL import Image
    from fyst_trajectories import compute_focal_plane_rotation, InstrumentOffset, sky_to_focal_plane

    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure_paths = []
    origin = timeline.start_time
    categories = [str(row["scan_type"]) if str(row["block_type"]) == "calibration"
                  else str(row["block_type"]) for row in full]
    kinds = list(dict.fromkeys(categories))
    palette = {kind: plt.get_cmap("tab20")(i) for i, kind in enumerate(kinds)}
    fig, ax = plt.subplots(figsize=(11, 3.8), layout="constrained")
    for row, native, kind in zip(full, timeline.blocks, categories, strict=True):
        start = float((native.t_start - origin).sec / 60)
        width = float(native.duration / 60)
        y = kinds.index(kind)
        ax.barh(y, width, left=start, height=0.65, color=palette[kind])
        if kind == "science":
            ax.text(start + width / 2, y, f"{row['scan_index']}/{row['subscan_index']}",
                    ha="center", va="center", weight="bold", color="white")
    ax.set(yticks=range(len(kinds)), yticklabels=kinds, xlabel="Minutes since schedule start",
           title="1 | Native FYST timeline - science labels are scan/subscan")
    ax.grid(axis="x", alpha=0.2)
    path = folder / "01_schedule.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    figure_paths.append(path)

    fig, axes = plt.subplots(5, 1, figsize=(11, 8), sharex=True, layout="constrained")
    rotation = samples.meta["roll_instrument_rotation_deg"]
    for index, (block, arrays) in enumerate(zip(blocks, obs_arrays, strict=True)):
        a = block.arrays
        # 不跨 block 连线，也不连接 overhead 空档。
        take = np.unique(np.linspace(0, len(a["times"]) - 1, min(3500, len(a["times"]))).astype(int))
        elapsed = (a["times"] - origin.unix) / 60
        theta = compute_focal_plane_rotation(a["fyst_el_deg"], site=timeline.site,
                                             offset=InstrumentOffset(dx=0, dy=0, instrument_rotation=rotation))
        axes[0].plot(elapsed[take], a["fyst_az_deg"][take], color=COLORS["Boresight"])
        axes[1].plot(elapsed[take], a["fyst_el_deg"][take], color=COLORS["Boresight"])
        axes[2].plot(elapsed[take], theta[take], color="#13966a", label="FYST theta" if index == 0 else None)
        axes[2].plot(elapsed[take], np.rad2deg(a["boresight_roll"])[take], color="#287fc1", label="TOAST rho" if index == 0 else None)
        # Flags 是离散状态，保留全部点，防止短 turnaround 被抽样跳过。
        axes[3].step(elapsed, a["fyst_scan_flag"], where="post", color="#8267bc")
        axes[4].step(elapsed, arrays["flags"], where="post", color="#d95056")
        for ax in axes:
            ax.axvspan(elapsed[0], elapsed[-1], color="black", alpha=0.035)
    for ax, label in zip(axes, ("Az [deg]", "El [deg]", "Rotation [deg]", "FYST enum", "TOAST mask"), strict=True):
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[2].legend(loc="upper right")
    axes[3].set_yticks([0, 1, 2, 3])
    axes[4].set_yticks([0, 1, 4])
    axes[-1].set_xlabel("Minutes since schedule start (gaps are overhead)")
    axes[0].set_title("2 | Stored trajectories and flags - all science blocks")
    path = folder / "02_trajectory.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    figure_paths.append(path)

    theta, poses, errors = comparison
    az, el = poses["Boresight"]
    center = (float(az[len(az) // 2]), float(el[len(el) // 2]))
    xy = {key: np.column_stack(sky_to_focal_plane(*center, *value, focal_plane_rotation=0))
          for key, value in poses.items()}
    n = len(az)
    thin = np.unique(np.linspace(0, n - 1, min(n, 2500)).astype(int))
    detail = np.flatnonzero(blocks[0].arrays["times"] - blocks[0].arrays["times"][0] <= 120)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    for ax, take, title in zip(axes, (thin, detail), ("Full block overview", "First 120 s - matching curves overlap"), strict=True):
        for label, values in xy.items():
            style = "--" if label == "Correct: -theta" else "-"
            ax.plot(*values[take].T, style, color=COLORS[label], lw=2.5 if label == "FYST reference" else 1.1,
                    label=label, alpha=0.5 if ax is axes[0] else 0.95)
        ax.set(xlabel="Local cross-elevation [deg]", ylabel="Local elevation [deg]", title=title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("3 | Off-axis detector: four roll choices in a fixed horizon tangent plane")
    path = folder / "03_offaxis.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    figure_paths.append(path)

    gif_path = folder / "walkthrough.gif"
    frames_dir = folder / "frames"
    if no_gif:
        # --overwrite --no-gif 不应留下前一轮动画作为本轮结果。
        gif_path.unlink(missing_ok=True)
        for name in ("first.png", "middle.png", "last.png"):
            (frames_dir / name).unlink(missing_ok=True)
        return figure_paths, None

    frames = np.unique(np.linspace(0, n - 1, min(64, n)).astype(int))
    fig = plt.figure(figsize=(10, 6), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.65, 1), height_ratios=(1.1, 1))
    track = fig.add_subplot(grid[:, 0])
    angle_ax = fig.add_subplot(grid[0, 1])
    status_ax = fig.add_subplot(grid[1, 1])
    fig.suptitle("FYST vs TOAST | One off-axis detector, four roll choices", fontsize=12)
    lines, markers = {}, {}
    for label, values in xy.items():
        track.plot(*values[thin].T, color=COLORS[label], lw=0.6, alpha=0.12)
        lines[label], = track.plot([], [], "--" if label == "Correct: -theta" else "-",
                                   color=COLORS[label], lw=3 if label == "FYST reference" else 1.5, label=label)
        markers[label], = track.plot([], [], "o", color=COLORS[label], ms=3)
    track.set(xlabel="Local cross-elevation [deg]", ylabel="Local elevation [deg]",
              title="Fixed horizon plane | 120 s trails")
    track.set_aspect("equal", adjustable="datalim")
    track.grid(alpha=0.2)
    track.legend(loc="upper left", fontsize=7)
    circle = np.linspace(0, 2 * np.pi, 120)
    angle_ax.plot(np.cos(circle), np.sin(circle), color="#ccd2d8", lw=1)
    angle_ax.axhline(0, color="#ddd", lw=0.7)
    angle_ax.axvline(0, color="#ddd", lw=0.7)
    angle_ax.plot([0, 0], [0, -1], "--", color="#aaa", label="Zero: FYST +x = -Y")
    cross_x, = angle_ax.plot([], [], color="#287fc1", lw=2)
    cross_y, = angle_ax.plot([], [], color="#287fc1", lw=2)
    arrow = angle_ax.annotate("", xy=(0, -1), xytext=(0, 0),
                              arrowprops={"arrowstyle": "->", "color": "#13966a", "lw": 3})
    angle_ax.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35), xlabel="TOAST local X", ylabel="TOAST local Y")
    angle_ax.set_aspect("equal")
    angle_ax.legend(loc="upper left", fontsize=7)
    status_ax.axis("off")
    status_text = status_ax.text(0.02, 0.95, "", transform=status_ax.transAxes,
                                va="top", family="monospace", fontsize=9)
    a = blocks[0].arrays
    times = a["times"]
    raw = a["fyst_scan_flag"]
    flags = obs_arrays[0]["flags"]
    state_names = {0: "UNCLASSIFIED", 1: "SCIENCE", 2: "TURNAROUND", 3: "RETUNE"}
    writer = PillowWriter(fps=10)
    with writer.saving(fig, str(gif_path), dpi=80):
        for index in frames:
            first = int(np.searchsorted(times, times[index] - 120))
            trail = np.unique(np.linspace(first, index, min(400, index - first + 1)).astype(int))
            for label, values in xy.items():
                lines[label].set_data(*values[trail].T)
                markers[label].set_data([values[index, 0]], [values[index, 1]])
            rho = a["boresight_roll"][index]
            x, y = np.array([np.cos(rho), np.sin(rho)]), np.array([-np.sin(rho), np.cos(rho)])
            cross_x.set_data([-x[0], x[0]], [-x[1], x[1]])
            cross_y.set_data([-y[0], y[0]], [-y[1], y[1]])
            arrow.xy = (-y[0], -y[1])
            angle_ax.set_title(f"Mechanical roll | El={el[index]:.2f} deg\nrho={np.rad2deg(rho):.2f} deg")
            status_text.set_text(
                f"{utc_ms(times[index])}\n\n"
                f"Sample {index + 1:,} / {n:,}\n"
                f"FYST: {int(raw[index])} {state_names[int(raw[index])]}\n"
                f"TOAST flags: {int(flags[index])}\n\n"
                f"theta = {theta[index]:.3f} deg\n"
                f"rho   = {np.rad2deg(rho):.3f} deg\n\n"
                "Gold dashed overlaps green.\n"
                "Display subsampling only."
            )
            writer.grab_frame()
    plt.close(fig)
    frames_dir.mkdir(exist_ok=True)
    with Image.open(gif_path) as gif:
        frame_count = gif.n_frames
        for name, index in (("first", 0), ("middle", frame_count // 2), ("last", frame_count - 1)):
            gif.seek(index)
            gif.convert("RGB").save(frames_dir / f"{name}.png")
    byte_count = gif_path.stat().st_size
    if byte_count > 5_000_000:
        raise RuntimeError(f"GIF 大于 5 MB：{byte_count} bytes")
    figure_paths.extend([gif_path, *(frames_dir / f"{name}.png" for name in ("first", "middle", "last"))])
    return figure_paths, {"frames": frame_count, "bytes": byte_count, "fps": 10,
                          "sample_indices": frames.tolist(), "size_pixels": [800, 480]}


def run(args):
    import h5py
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import solar_system_ephemeris
    from astropy.table import QTable
    from astropy.time import Time, TimeDelta
    from astropy.utils import iers

    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "error"
    # 显式 bundled IERS-A，writer 子进程也收到同一文件，避免使用用户缓存的表。
    iers.earth_orientation_table.set(iers.IERS_A.open(iers.IERS_A_FILE))
    solar_system_ephemeris.set("builtin")

    from fyst_trajectories.overhead import read_timeline
    from toast import Data, qarray as qa
    from toast.instrument import Focalplane, GroundSite, Telescope
    from toast.instrument_coords import xieta_to_quat
    from toast.instrument_sim import fake_boresight_focalplane
    from toast.mpi import Comm
    from toast.ops import LoadHDF5, SaveHDF5
    from fyst_to_toast.generate import generate_trajectory
    from fyst_to_toast.trajectory_h5 import _load_trajectory_block
    from fyst_to_toast.sim_fyst import SimFYST

    out = args.output_dir
    generation = out / "01_fyst"
    writer_dir, loader_dir, sim_dir = (out / name for name in ("03_writer", "04_loader", "05_simfyst"))
    for path in (writer_dir, loader_dir, sim_dir):
        path.mkdir(exist_ok=True)
    files = []

    def record(path, owner, viewer=None):
        command = viewer or f"cat {shlex.quote(str(path))}"
        files.append({"file": str(path.relative_to(out)), "producer": owner, "view": command})

    step(1, "FYST 原生（本 repo 调用）", f"case={args.case} → {generation}/schedule.ecsv", "调度、overhead、science blocks")
    print("generate_trajectory 一次完成第 1、2 步；下面按文件逐层展开，不重复生成轨迹。")
    generated = generate_trajectory(args.case, generation, duration_s=args.duration_s,
                                    iers_file=iers.IERS_A_FILE, overwrite=args.overwrite)
    if generated["status"] != "complete":
        raise RuntimeError(f"窗口没有 science；检查 {generation}/full_timeline.ecsv，增大 --duration-s。")
    schedule = QTable.read(generation / "schedule.ecsv")
    full = QTable.read(generation / "full_timeline.ecsv")
    timeline = read_timeline(generation / "full_timeline.ecsv")
    science = read_timeline(generation / "schedule.ecsv")
    print(f"【执行验证】science blocks={len(schedule)}；window={args.duration_s:g} s")
    print("scan/subscan  start UTC                         stop UTC                          Az min/max [deg]       El [deg]  sky boresight_angle [deg]")
    for row in schedule:
        print(f"{row['scan_index']:3}/{row['subscan_index']:<3}  {row['start_time']}  {row['stop_time']}  "
              f"{row['azmin'].value:9.3f}/{row['azmax'].value:9.3f}  {row['el'].value:8.3f}  {row['boresight_angle'].value:10.6f}")
    if len(schedule) < 2:
        print("本次自选窗口只有一个 science block；可增大 --duration-s 查看多个 blocks。")
    for name in ("full_timeline.ecsv", "schedule.ecsv"):
        path = generation / name
        record(path, "第1步 / FYST write_timeline", f"python tools/ecsv_reader.py {shlex.quote(str(path))} --view text --pager never --rows 8")

    source = generation / "fyst_trajectory.ecsv"
    step(2, "FYST 原生重建 + 本 repo 保存", f"{generation}/schedule.ecsv → {source}", "原始逐样本 arrays、roll、metadata")
    samples = QTable.read(source)
    for key in samples.colnames:
        print(f"  {key:<25} {str(samples[key].dtype):<8} {str(getattr(samples[key], 'unit', '1'))}")
    print("\n前 3 行（保留原始数值；未重采样）：")
    print("\n".join(samples[:3].pformat(max_lines=-1, max_width=-1)))
    print("FYST scan_flag 计数：", counts(samples["scan_flag"]))
    roll_meta = {k: v for k, v in samples.meta.items() if k.startswith("roll_")}
    print("roll metadata：", json.dumps(roll_meta, ensure_ascii=False))
    for b in samples.meta["blocks"]:
        roll = samples["boresight_roll_deg"].value[b["first"]:b["last"]]
        print(f"  block {b['scan_index']}/{b['subscan_index']}: N={len(roll)}, "
              f"roll min/max/drift={roll.min():.6f}/{roll.max():.6f}/{np.ptp(roll):.6f} deg")
    record(source, "第2步 / FYST reconstruction + 本 repo QTable", f"python tools/ecsv_reader.py {shlex.quote(str(source))} --view text --pager never --rows 5")
    record(generation / "report.json", "第1–2步 / generate_trajectory")

    step(3, "本 repo writer → TOAST 原生 SaveHDF5", f"{source} → {writer_dir}/*.h5", "每 native block 一个文件，5 个示意 detectors")
    command = [sys.executable, str(REPO / "scripts/save_trajectory_h5.py"), str(source),
               "--output-dir", str(writer_dir), "--iers-file", str(iers.IERS_A_FILE)]
    if args.overwrite:
        command.append("--overwrite")
    print("复用 CLI 的站点读取/校验，使用 subprocess：", shlex.join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    (writer_dir / "writer_cli.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"writer 失败，查看 {writer_dir}/writer_cli.log")
    h5_paths = [writer_dir / name for name in json.loads((writer_dir / "files.json").read_text())]
    print("Shared 单位是文件契约说明，H5 shared dataset 本身不带 unit 属性。")
    for path in h5_paths:
        print(f"\n{path}")
        with h5py.File(path) as saved:
            describe_arrays(dict(saved["shared"].items()))
            info = json.loads(saved["metadata/other"].attrs["trajectory_info"])
            print("  trajectory_info：", json.dumps({k: info[k] for k in
                  ("case", "sample_rate_hz", "roll_source", "roll_convention", "block")}, ensure_ascii=False))
        record(path, "第3步 / writer + SaveHDF5", f"python tools/h5reader.py {shlex.quote(str(path))} --tree")
    for name in ("files.json", "writer_report.json", "writer_cli.log"):
        record(writer_dir / name, "第3步 / writer CLI")

    step(4, "TOAST 原生 LoadHDF5 + 本 repo loader", f"{writer_dir}/*.h5 → {loader_dir}/loader_summary.json", "临时 Observation vs 13-array _TrajectoryBlock")
    blocks, loader_summary = [], []
    for path in h5_paths:
        temporary = Data(Comm(world=None))
        try:
            LoadHDF5(files=[str(path)], detdata=[], force_serial=True).apply(temporary)
            obs = temporary.obs[0]
            print(f"\n临时 Observation {obs.name}: detectors={len(obs.all_detectors)} {obs.all_detectors}; "
                  f"shared={len(obs.shared)}, detdata={list(obs.detdata)}, intervals={list(obs.intervals)}")
            print("  含示意 instrument、geometry、flags；下面的 loader 只返回输入 arrays 与 metadata。")
            native_count = len(obs.all_detectors)
            assert native_count == 5
        finally:
            temporary.clear()
        block = _load_trajectory_block(path)
        blocks.append(block)
        assert len(block.arrays) == 13
        print(f"_TrajectoryBlock {block.observation_name}: 不携带 focalplane / detector / Observation")
        description = describe_arrays(block.arrays)
        print("  metadata: trajectory_info, source_shared_array_sha256, source_site, actual_rate_hz, clock residual, roll convention/source")
        loader_summary.append({"name": block.observation_name, "temporary_detectors": native_count,
                               "arrays": description, "trajectory_info": block.trajectory_info,
                               "actual_rate_hz": block.actual_rate_hz,
                               "max_phase_residual_s": block.max_phase_residual_s})
    json_file(loader_dir / "loader_summary.json", loader_summary)
    record(loader_dir / "loader_summary.json", "第4步 / 本例检查 native LoadHDF5 与 adapter loader；arrays 在内存")

    step(5, "本 repo SimFYST → TOAST 原生 Observation/SaveHDF5", f"{writer_dir}/*.h5 + caller Telescope → {sim_dir}/*.h5", "2 个 caller detectors、flags、intervals、rate/roll")
    obs_arrays, sim_summary = [], []
    fp_quat = xieta_to_quat(np.sin(np.deg2rad(1.0)), 0.0, 0.0)
    for path, block in zip(h5_paths, blocks, strict=True):
        full_fp = fake_boresight_focalplane(n_pix=1, sample_rate=10 * u.Hz)
        table = full_fp.detector_data[:2].copy()
        table["name"] = ["caller-boresight", "caller-fyst-x-1deg"]
        table["quat"][0] = xieta_to_quat(0.0, 0.0, 0.0)
        table["quat"][1] = fp_quat
        fp = Focalplane(detector_data=table, sample_rate=10 * u.Hz, field_of_view=3 * u.deg)
        site = GroundSite("caller-site", block.source_site["lat"], block.source_site["lon"], block.source_site["height"])
        telescope = Telescope("walkthrough-caller", site=site, focalplane=fp)
        data = Data(Comm(world=None))
        try:
            op = SimFYST(telescope=telescope, trajectory_file=str(path))
            op.apply(data)
            obs = data.obs[0]
            arrays = {key: obs.shared[key].data for key in obs.shared}
            print(f"\n{obs.name}: caller detectors={[str(d) for d in obs.local_detectors]}")
            describe_arrays(arrays)
            info = json.loads(obs["simfyst_info"])
            spans = {name: [[int(iv.first), int(iv.last)] for iv in obs.intervals[name]]
                     for name in (op.scanning_interval, op.turnaround_interval, op.fyst_invalid_interval)}
            print("  flag_mapping / class_counts：", info["flag_mapping"], info["class_counts"])
            print("  intervals 数量：", {key: len(value) for key, value in spans.items()})
            print("  rate_check：", info["rate_check"])
            print("  roll 来源：", info["roll_source"], info["roll_convention"])
            print("  detdata 为零初始化 signal/flags，本例没有 sky/noise simulation。")
            np.testing.assert_array_equal(arrays["boresight_roll"], block.arrays["boresight_roll"])
            obs_arrays.append({key: arrays[key].copy() for key in ("boresight_azel", "flags")})
            sim_summary.append({"name": obs.name, "simfyst_info": info, "intervals_samples_half_open": spans})
            SaveHDF5(volume=str(sim_dir), volume_index=None, force_serial=True).apply(data)
            saved_path = sim_dir / f"{obs.name}.h5"
            record(saved_path, "第5步 / SimFYST caller Observation + SaveHDF5", f"python tools/h5reader.py {shlex.quote(str(saved_path))} --tree")
        finally:
            data.clear()
    json_file(sim_dir / "simfyst_summary.json", sim_summary)
    record(sim_dir / "simfyst_summary.json", "第5步 / 本例导出实际 SimFYST metadata/intervals")

    step(6, "FYST 原生 vs 本 repo/TOAST 原生", f"同一 block 的实际 arrays → {out}/report.json + figures/", "时间、Az、flags、roll、intervals 数值对照")
    block, arrays = blocks[0], blocks[0].arrays
    t = arrays["times"]
    j = min(123, len(t) - 1)
    relative = arrays["fyst_block_time_s"]
    restored = Time(t[0], format="unix", scale="utc") + TimeDelta(relative[j] - relative[0], format="sec")
    print(f"a. 时间：样本 {j}，FYST block_time_s={relative[j]:.9f} s，首样本 block_time_s={relative[0]:.9f} s")
    print(f"   由保存值还原 Astropy Time={utc_ms(restored)}；TOAST Unix float64={t[j]:.9f} s → {utc_ms(t[j])}")
    print(f"   schedule start={schedule[0]['start_time']}；Astropy/Unix 残差={restored.unix - t[j]:.3g} s；UTC 文本只显示毫秒。")
    lon, lat, _ = qa.to_lonlat_angles(obs_arrays[0]["boresight_azel"])
    delta = np.arctan2(np.sin(lon + arrays["azimuth"]), np.cos(lon + arrays["azimuth"]))
    print(f"b. Az：原 Az={arrays['fyst_az_deg'][j]:.6f} deg，quaternion longitude={np.rad2deg(lon[j]):.6f} deg，"
          f"回投 -longitude={-np.rad2deg(lon[j]):.6f} deg；wrap 后 max|lon+Az|={np.rad2deg(abs(delta)).max():.3g} deg")
    print("   【仅读源码】Q0=qa.from_lonlat_angles(-az, el, 0)，随后右乘 Rz(roll)：trajectory_h5.py:_boresight_with_roll。")
    print("c. Flags：FYST enum → TOAST bitmask（只统计所选 block；缺席状态计数为 0）")
    raw, flags = arrays["fyst_scan_flag"], obs_arrays[0]["flags"]
    flag_rows = []
    for code, mask in ((0, 1), (1, 0), (2, 4), (3, 1)):
        selected = raw == code
        row = {"fyst_enum": code, "toast_mask": mask, "samples": int(selected.sum()),
               "matched": int(np.count_nonzero(selected & (flags == mask)))}
        assert row["samples"] == row["matched"]
        flag_rows.append(row)
        print(f"   {code} → {mask}: FYST={row['samples']}, TOAST 匹配={row['matched']}")
    schedule_angle = float(schedule[0]["boresight_angle"].to_value(u.deg))
    comparison = roll_comparison(block, obs_arrays[0]["boresight_azel"], fp_quat, science.site,
                                 samples.meta["roll_instrument_rotation_deg"], schedule_angle)
    theta, poses, errors = comparison
    rho = np.rad2deg(arrays["boresight_roll"])
    print(f"d. Roll：schedule boresight_angle={schedule_angle:.9f} deg（天空坐标系，每 block 一个值，含视差角）")
    print(f"   FYST 机械 theta 范围={theta.min():.9f}..{theta.max():.9f} deg；TOAST rho 范围={rho.min():.9f}..{rho.max():.9f} deg")
    print(f"   同一样本 {j}: El={arrays['fyst_el_deg'][j]:.9f}，theta={theta[j]:.9f}，rho={rho[j]:.9f} deg")
    for label, error in errors.items():
        print(f"   【执行验证】{label:<20} max detector separation={error:.12g} arcsec（全部 {len(t)} 点）")
    print("   【仅读源码】FYST +x = TOAST local -Y，所以 rho=-theta；视差旋转由 TOAST azel_to_radec 处理。")
    print("   依据：research_notes/roll_investigation_2026-10-19/README.md:9–20、:56–62。此处只验 horizon 几何。")
    print(f"e. 名义 block=[{schedule[0]['start_time']}, {schedule[0]['stop_time']})")
    print(f"   实际首/末样本={utc_ms(t[0])} / {utc_ms(t[-1])}，N={len(t)}；名义 stop 不保证等于最后一个 sample。")
    for key, spans in sim_summary[0]["intervals_samples_half_open"].items():
        print(f"   TOAST {key}: {len(spans)} 个样本区间 [first,last)，前3个={spans[:3]}")
        if spans:
            first, last = spans[0]
            print(f"     首段包含样本 {first}..{last - 1}：{utc_ms(t[first])} → {utc_ms(t[last - 1])}")
    print("【执行验证】只在作图时抽样：PNG 每条曲线最多 3500 点，GIF 64 帧/120 s 拖尾；flags 图保留全部点。")
    figure_paths, gif = draw_figures(out / "figures", full, timeline, samples, blocks, obs_arrays, comparison, args.no_gif)
    for path in figure_paths:
        record(path, "第6步 / matplotlib Agg + PillowWriter（图像子采样）", f"open {shlex.quote(str(path))}")
    report = {
        "evidence": "执行验证", "case": args.case, "duration_s": args.duration_s,
        "science_blocks": len(blocks), "samples": len(samples), "selected_block": block.observation_name,
        "selected_samples": len(t), "roll_metadata": roll_meta,
        "schedule_boresight_angle_deg": schedule_angle,
        "theta_min_max_deg": [float(theta.min()), float(theta.max())],
        "rho_min_max_deg": [float(rho.min()), float(rho.max())],
        "roll_max_error_arcsec": errors, "flag_rows": flag_rows,
        "time_example": {"sample": j, "block_time_s": float(relative[j]), "unix_s": float(t[j]), "utc": utc_ms(t[j])},
        "longitude_max_error_deg": float(np.rad2deg(abs(delta)).max()),
        "gif": gif, "iers_file": str(iers.IERS_A_FILE), "iers_auto_download": False,
        "unrun": ["独立 sky astrometry", "真实仪器/偏振标定", "MPI", "sky/noise simulation"],
    }
    record(out / "report.json", "第6步 / 本例完整数值报告")
    record(out / "walkthrough.log", "第1–6步 / 终端输出")
    record(out / "file_map.json", "收尾 / 本例文件地图")
    json_file(out / "file_map.json", files)
    print("\n文件地图（命令从 repo 根目录执行；JSON/log 用 cat，PNG/GIF 用图像查看器）：")
    for item in files:
        print(f"  {item['file']} → {item['producer']}\n    {item['view']}")
    print("【未运行】独立 sky astrometry、真实仪器/偏振标定、MPI、sky/noise；本例不验收扫描动力学。")
    return report


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", choices=("ces", "daisy", "pong"), default="daisy")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S,
                        help="含原生 overhead 的调度窗口；默认 6000 s，至少两个 science blocks")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error("duration-s 必须是有限正数")
    args.output_dir = (args.output_dir or REPO / "output/walkthrough" / args.case).expanduser().resolve()
    output_root = (REPO / "output").resolve()
    if not args.output_dir.is_relative_to(output_root) or args.output_dir == output_root:
        parser.error("所有产物必须放在本 repo output/ 的子目录中")
    if args.output_dir.exists() and not args.overwrite:
        parser.error("输出目录已存在；请换目录或显式 --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        cache = args.output_dir / ".cache" / key.lower()
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(cache)
    with (args.output_dir / "walkthrough.log").open("w") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            report = run(args)
            report["elapsed_s"] = time.perf_counter() - started
            json_file(args.output_dir / "report.json", report)
            print(f"\n【执行验证】完成，用时 {report['elapsed_s']:.2f} s；报告：{args.output_dir / 'report.json'}")
            if report["gif"]:
                print(f"GIF：{report['gif']['frames']} 帧，{report['gif']['bytes'] / 1_000_000:.3f} MB；首/中/末帧已保存。")


if __name__ == "__main__":
    main()
