"""Step-by-step FYST → ECSV → TOAST H5 → SimFYST walkthrough and roll comparison.

This teaching example imports fyst_trajectories directly;
the library dependency boundary applies to src/. It reuses the generator, writer, loader, and SimFYST
without reimplementing scheduling, motion, flags, or intervals. All outputs go under output/.
"""

import os

# Set before NumPy / TOAST / FYST imports; affects only this example and its subprocesses.
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
    """Write walkthrough output to both the terminal and the run log."""

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
    print(f"\nStep {number}: {owner} · {files} · {content}")


def unit_for(key):
    """Shared arrays carry no units; print the adapter's contract units explicitly."""
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
    """Time or Unix values → millisecond UTC; preserve stored floating-point sample times."""
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
    """Compare horizon geometry over a full block using the actual SimFYST Observation."""
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
    """Subsample only for plots; errors, storage, flags, and intervals use all samples."""
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
    rotation = 0.0  # Clocking is applied to focal-plane detectors, not the roll; none here.
    for index, (block, arrays) in enumerate(zip(blocks, obs_arrays, strict=True)):
        a = block.arrays
        # Do not connect across blocks or overhead gaps.
        take = np.unique(np.linspace(0, len(a["times"]) - 1, min(3500, len(a["times"]))).astype(int))
        elapsed = (a["times"] - origin.unix) / 60
        theta = compute_focal_plane_rotation(a["fyst_el_deg"], site=timeline.site,
                                             offset=InstrumentOffset(dx=0, dy=0, instrument_rotation=rotation))
        axes[0].plot(elapsed[take], a["fyst_az_deg"][take], color=COLORS["Boresight"])
        axes[1].plot(elapsed[take], a["fyst_el_deg"][take], color=COLORS["Boresight"])
        axes[2].plot(elapsed[take], theta[take], color="#13966a", label="FYST theta" if index == 0 else None)
        axes[2].plot(elapsed[take], np.rad2deg(a["boresight_roll"])[take], color="#287fc1", label="TOAST rho" if index == 0 else None)
        # Keep all flag samples so subsampling cannot hide short turnarounds.
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
        # --overwrite --no-gif must not leave a previous animation as a current result.
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
        raise RuntimeError(f"GIF exceeds 5 MB: {byte_count} bytes")
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
    # Use explicit bundled IERS-A in both this process and the writer, avoiding cached user tables.
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

    step(1, "Native FYST (called by this repository)", f"case={args.case} → {generation}/schedule.ecsv", "Scheduling, overhead, science blocks")
    print("generate_trajectory performs Steps 1 and 2 together; inspect each file below without regenerating trajectories.")
    generated = generate_trajectory(args.case, generation, duration_s=args.duration_s,
                                    iers_file=iers.IERS_A_FILE, overwrite=args.overwrite)
    if generated["status"] != "complete":
        raise RuntimeError(f"No science in this window; inspect {generation}/full_timeline.ecsv and increase --duration-s.")
    schedule = QTable.read(generation / "schedule.ecsv")
    full = QTable.read(generation / "full_timeline.ecsv")
    timeline = read_timeline(generation / "full_timeline.ecsv")
    science = read_timeline(generation / "schedule.ecsv")
    print(f"[Verified] science blocks={len(schedule)}; window={args.duration_s:g} s")
    print("scan/subscan  start UTC                         stop UTC                          Az min/max [deg]       El [deg]  sky boresight_angle [deg]")
    for row in schedule:
        print(f"{row['scan_index']:3}/{row['subscan_index']:<3}  {row['start_time']}  {row['stop_time']}  "
              f"{row['azmin'].value:9.3f}/{row['azmax'].value:9.3f}  {row['el'].value:8.3f}  {row['boresight_angle'].value:10.6f}")
    if len(schedule) < 2:
        print("This window has only one science block; increase --duration-s to see multiple blocks.")
    for name in ("full_timeline.ecsv", "schedule.ecsv"):
        path = generation / name
        record(path, "Step 1 / FYST write_timeline", f"python tools/ecsv_reader.py {shlex.quote(str(path))} --view text --pager never --rows 8")

    source = generation / "fyst_trajectory.ecsv"
    step(2, "Native FYST reconstruction + adapter storage", f"{generation}/schedule.ecsv → {source}", "Raw per-sample arrays, roll, metadata")
    samples = QTable.read(source)
    for key in samples.colnames:
        print(f"  {key:<25} {str(samples[key].dtype):<8} {str(getattr(samples[key], 'unit', '1'))}")
    print("\nFirst 3 rows (original values; no resampling):")
    print("\n".join(samples[:3].pformat(max_lines=-1, max_width=-1)))
    print("FYST scan_flag counts:", counts(samples["scan_flag"]))
    roll_meta = {k: v for k, v in samples.meta.items() if k.startswith("roll_")}
    print("roll metadata: ", json.dumps(roll_meta, ensure_ascii=False))
    for b in samples.meta["blocks"]:
        roll = samples["boresight_roll_deg"].value[b["first"]:b["last"]]
        print(f"  block {b['scan_index']}/{b['subscan_index']}: N={len(roll)}, "
              f"roll min/max/drift={roll.min():.6f}/{roll.max():.6f}/{np.ptp(roll):.6f} deg")
    record(source, "Step 2 / FYST reconstruction + adapter QTable", f"python tools/ecsv_reader.py {shlex.quote(str(source))} --view text --pager never --rows 5")
    record(generation / "report.json", "Steps 1–2 / generate_trajectory")

    step(3, "Adapter writer → native TOAST SaveHDF5", f"{source} → {writer_dir}/*.h5", "One file per native block, 5 illustrative detectors")
    command = [sys.executable, str(REPO / "scripts/save_trajectory_h5.py"), str(source),
               "--output-dir", str(writer_dir), "--iers-file", str(iers.IERS_A_FILE)]
    if args.overwrite:
        command.append("--overwrite")
    print("Reuse CLI site loading / validation via subprocess:", shlex.join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    (writer_dir / "writer_cli.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Writer failed; see {writer_dir}/writer_cli.log")
    h5_paths = [writer_dir / name for name in json.loads((writer_dir / "files.json").read_text())]
    print("Shared units describe the file contract; H5 shared datasets have no unit attribute.")
    for path in h5_paths:
        print(f"\n{path}")
        with h5py.File(path) as saved:
            describe_arrays(dict(saved["shared"].items()))
            info = json.loads(saved["metadata/other"].attrs["trajectory_info"])
            print("  trajectory_info: ", json.dumps({k: info[k] for k in
                  ("case", "sample_rate_hz", "roll_source", "roll_convention", "block")}, ensure_ascii=False))
        record(path, "Step 3 / writer + SaveHDF5", f"python tools/h5reader.py {shlex.quote(str(path))} --tree")
    for name in ("files.json", "writer_report.json", "writer_cli.log"):
        record(writer_dir / name, "Step 3 / writer CLI")

    step(4, "Native TOAST LoadHDF5 + adapter loader", f"{writer_dir}/*.h5 → {loader_dir}/loader_summary.json", "Temporary Observation vs 13-array _TrajectoryBlock")
    blocks, loader_summary = [], []
    for path in h5_paths:
        temporary = Data(Comm(world=None))
        try:
            LoadHDF5(files=[str(path)], detdata=[], force_serial=True).apply(temporary)
            obs = temporary.obs[0]
            print(f"\nTemporary Observation {obs.name}: detectors={len(obs.all_detectors)} {obs.all_detectors}; "
                  f"shared={len(obs.shared)}, detdata={list(obs.detdata)}, intervals={list(obs.intervals)}")
            print("  Includes an illustrative instrument, geometry, and flags; the loader below returns only input arrays and metadata.")
            native_count = len(obs.all_detectors)
            assert native_count == 5
        finally:
            temporary.clear()
        block = _load_trajectory_block(path)
        blocks.append(block)
        assert len(block.arrays) == 13
        print(f"_TrajectoryBlock {block.observation_name}: Contains no focalplane / detector / Observation")
        description = describe_arrays(block.arrays)
        print("  metadata: trajectory_info, source_shared_array_sha256, source_site, actual_rate_hz, clock residual, roll convention/source")
        loader_summary.append({"name": block.observation_name, "temporary_detectors": native_count,
                               "arrays": description, "trajectory_info": block.trajectory_info,
                               "actual_rate_hz": block.actual_rate_hz,
                               "max_phase_residual_s": block.max_phase_residual_s})
    json_file(loader_dir / "loader_summary.json", loader_summary)
    record(loader_dir / "loader_summary.json", "Step 4 / Example checks native LoadHDF5 and adapter loader; arrays remain in memory")

    step(5, "SimFYST → native TOAST Observation/SaveHDF5", f"{writer_dir}/*.h5 + caller Telescope → {sim_dir}/*.h5", "2 caller detectors, flags, intervals, rate/roll")
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
            print("  flag_mapping / class_counts: ", info["flag_mapping"], info["class_counts"])
            print("  interval counts: ", {key: len(value) for key, value in spans.items()})
            print("  rate_check: ", info["rate_check"])
            print("  roll source: ", info["roll_source"], info["roll_convention"])
            print("  detdata contains zero-initialized signal/flags; this example runs no sky/noise simulation.")
            np.testing.assert_array_equal(arrays["boresight_roll"], block.arrays["boresight_roll"])
            obs_arrays.append({key: arrays[key].copy() for key in ("boresight_azel", "flags")})
            sim_summary.append({"name": obs.name, "simfyst_info": info, "intervals_samples_half_open": spans})
            SaveHDF5(volume=str(sim_dir), volume_index=None, force_serial=True).apply(data)
            saved_path = sim_dir / f"{obs.name}.h5"
            record(saved_path, "Step 5 / SimFYST caller Observation + SaveHDF5", f"python tools/h5reader.py {shlex.quote(str(saved_path))} --tree")
        finally:
            data.clear()
    json_file(sim_dir / "simfyst_summary.json", sim_summary)
    record(sim_dir / "simfyst_summary.json", "Step 5 / Example exports actual SimFYST metadata/intervals")

    step(6, "Native FYST vs adapter/native TOAST", f"Actual arrays from the same block → {out}/report.json + figures/", "Numerical comparisons of time, Az, flags, roll, intervals")
    block, arrays = blocks[0], blocks[0].arrays
    t = arrays["times"]
    j = min(123, len(t) - 1)
    relative = arrays["fyst_block_time_s"]
    restored = Time(t[0], format="unix", scale="utc") + TimeDelta(relative[j] - relative[0], format="sec")
    print(f"a. Time: sample {j}, FYST block_time_s={relative[j]:.9f} s, first sample block_time_s={relative[0]:.9f} s")
    print(f"   Astropy Time reconstructed from saved values={utc_ms(restored)}; TOAST Unix float64={t[j]:.9f} s → {utc_ms(t[j])}")
    print(f"   schedule start={schedule[0]['start_time']}; Astropy/Unix residual={restored.unix - t[j]:.3g} s; UTC text shows milliseconds only.")
    lon, lat, _ = qa.to_lonlat_angles(obs_arrays[0]["boresight_azel"])
    delta = np.arctan2(np.sin(lon + arrays["azimuth"]), np.cos(lon + arrays["azimuth"]))
    print(f"b. Az: original Az={arrays['fyst_az_deg'][j]:.6f} deg, quaternion longitude={np.rad2deg(lon[j]):.6f} deg, "
          f"Projected back -longitude={-np.rad2deg(lon[j]):.6f} deg; After wrapping, max|lon+Az|={np.rad2deg(abs(delta)).max():.3g} deg")
    print("   [Source inspection] Q0=qa.from_lonlat_angles(-az, el, 0), then right-multiply by Rz(roll): trajectory_h5.py:_boresight_with_roll.")
    print("c. Flags: FYST enum → TOAST bitmask (selected block only; absent states have zero counts)")
    raw, flags = arrays["fyst_scan_flag"], obs_arrays[0]["flags"]
    flag_rows = []
    for code, mask in ((0, 1), (1, 0), (2, 4), (3, 1)):
        selected = raw == code
        row = {"fyst_enum": code, "toast_mask": mask, "samples": int(selected.sum()),
               "matched": int(np.count_nonzero(selected & (flags == mask)))}
        assert row["samples"] == row["matched"]
        flag_rows.append(row)
        print(f"   {code} → {mask}: FYST={row['samples']}, TOAST matches={row['matched']}")
    schedule_angle = float(schedule[0]["boresight_angle"].to_value(u.deg))
    comparison = roll_comparison(block, obs_arrays[0]["boresight_azel"], fp_quat, science.site,
                                 0.0, schedule_angle)
    theta, poses, errors = comparison
    rho = np.rad2deg(arrays["boresight_roll"])
    print(f"d. Roll: schedule boresight_angle={schedule_angle:.9f} deg (sky frame, one value per block, includes parallactic angle)")
    print(f"   FYST mechanical theta range={theta.min():.9f}..{theta.max():.9f} deg; TOAST rho range={rho.min():.9f}..{rho.max():.9f} deg")
    print(f"   Same sample {j}: El={arrays['fyst_el_deg'][j]:.9f}, theta={theta[j]:.9f}, rho={rho[j]:.9f} deg")
    for label, error in errors.items():
        print(f"   [Verified] {label:<20} max detector separation={error:.12g} arcsec (all {len(t)} samples)")
    print("   [Source inspection] FYST +x = TOAST local -Y, so rho=-theta; TOAST azel_to_radec handles parallactic rotation.")
    print("   This comparison validates horizon geometry only.")
    print(f"e. Nominal block=[{schedule[0]['start_time']}, {schedule[0]['stop_time']})")
    print(f"   Actual first/last samples={utc_ms(t[0])} / {utc_ms(t[-1])}, N={len(t)}; nominal stop need not equal the last sample.")
    for key, spans in sim_summary[0]["intervals_samples_half_open"].items():
        print(f"   TOAST {key}: {len(spans)} sample intervals [first,last), first 3={spans[:3]}")
        if spans:
            first, last = spans[0]
            print(f"     First interval contains samples {first}..{last - 1}: {utc_ms(t[first])} → {utc_ms(t[last - 1])}")
    print("[Verified] Subsample only for plotting: at most 3500 points per PNG curve, 64 GIF frames with a 120 s trail; flag plots keep all samples.")
    figure_paths, gif = draw_figures(out / "figures", full, timeline, samples, blocks, obs_arrays, comparison, args.no_gif)
    for path in figure_paths:
        record(path, "Step 6 / matplotlib Agg + PillowWriter (plot subsampling)", f"open {shlex.quote(str(path))}")
    report = {
        "evidence": "Verified by execution", "case": args.case, "duration_s": args.duration_s,
        "science_blocks": len(blocks), "samples": len(samples), "selected_block": block.observation_name,
        "selected_samples": len(t), "roll_metadata": roll_meta,
        "schedule_boresight_angle_deg": schedule_angle,
        "theta_min_max_deg": [float(theta.min()), float(theta.max())],
        "rho_min_max_deg": [float(rho.min()), float(rho.max())],
        "roll_max_error_arcsec": errors, "flag_rows": flag_rows,
        "time_example": {"sample": j, "block_time_s": float(relative[j]), "unix_s": float(t[j]), "utc": utc_ms(t[j])},
        "longitude_max_error_deg": float(np.rad2deg(abs(delta)).max()),
        "gif": gif, "iers_file": str(iers.IERS_A_FILE), "iers_auto_download": False,
        "unrun": ["independent sky astrometry", "real instrument / polarization calibration", "MPI", "sky/noise simulation"],
    }
    record(out / "report.json", "Step 6 / Full numerical report from this example")
    record(out / "walkthrough.log", "Steps 1–6 / Terminal output")
    record(out / "file_map.json", "Summary / example file map")
    json_file(out / "file_map.json", files)
    print("\nFile map (run commands from the repository root; use cat for JSON/log and an image viewer for PNG/GIF):")
    for item in files:
        print(f"  {item['file']} → {item['producer']}\n    {item['view']}")
    print("[Not run] independent sky astrometry, real instrument / polarization calibration, MPI, sky/noise; this example does not validate scan dynamics.")
    return report


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", choices=("ces", "daisy", "pong"), default="daisy")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S,
                        help="Scheduling window including native overhead; default 6000 s, at least two science blocks")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error("duration-s must be finite and positive")
    args.output_dir = (args.output_dir or REPO / "output/walkthrough" / args.case).expanduser().resolve()
    output_root = (REPO / "output").resolve()
    if not args.output_dir.is_relative_to(output_root) or args.output_dir == output_root:
        parser.error("All outputs must be in a subdirectory of this repository's output/")
    if args.output_dir.exists() and not args.overwrite:
        parser.error("Output directory exists; choose another directory or specify --overwrite")
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
            print(f"\n[Verified] Completed in {report['elapsed_s']:.2f} s; Report: {args.output_dir / 'report.json'}")
            if report["gif"]:
                print(f"GIF: {report['gif']['frames']} frames, {report['gif']['bytes'] / 1_000_000:.3f} MB; First/middle/last frames saved.")


if __name__ == "__main__":
    main()
