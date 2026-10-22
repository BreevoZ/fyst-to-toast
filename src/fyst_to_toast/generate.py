"""Native FYST scheduling and per-sample ECSV generation.

Ported from the user's local legacy examples (examples/fyst_to_toast/01-02 of the
old fyst-trajectories sandbox).
Only this module imports FYST, lazily. Callers configure IERS / ephemeris and
MPI / thread settings; importing the package never configures the process.
"""

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from fyst_to_toast import trajectory_h5 as schema

DEFAULT_DURATION_S = 10800
DEFAULT_START = {
    "ces": "2026-08-02T01:25:00",
    "daisy": "2026-03-15T04:00:00",
    # FYST docs/trajectory_examples.rst:143 (f095c51).
    "pong": "2026-03-15T01:00:00",
}
OUTPUT_NAMES = (
    "full_timeline.ecsv",
    "schedule.ecsv",
    "fyst_trajectory.ecsv",
    "report.json",
)


def sha256(path):
    """Return a file's byte hash for provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_legacy_patch(case):
    """Build legacy CES / Daisy or FYST's documented Pong as an ObservingPatch."""
    from fyst_trajectories.overhead import ObservingPatch

    if case not in DEFAULT_START:
        raise ValueError(f"Unknown case: {case}")
    if case == "ces":
        patch = ObservingPatch(
            name="native CES",
            ra_center=330,
            dec_center=-50,
            width=30,
            height=8,
            scan_type="constant_el",
            velocity=1,
            elevation=50,
            scan_params={"timestep": 0.1, "az_accel": 1.0},
        )
    elif case == "daisy":
        patch = ObservingPatch(
            name="native Daisy",
            ra_center=180,
            dec_center=-30,
            width=2,
            height=2,
            scan_type="daisy",
            velocity=0.3,
            scan_params={"radius": 1, "turn_radius": 0.3, "timestep": 0.1},
        )
    else:
        # Field and pattern: FYST docs/trajectory_examples.rst:145-160 (f095c51).
        # Native period is 240 s, within the default 3600 - 300 s subscan budget.
        patch = ObservingPatch(
            name="native Pong",
            ra_center=180,
            dec_center=-30,
            width=2,
            height=2,
            scan_type="pong",
            velocity=0.4,
            scan_params={
                "spacing": 0.1,
                "num_terms": 4,
                "angle": 0.0,
                "timestep": 0.1,
            },
        )
    return patch


def generate_observing_timeline(patch, start, duration_s):
    """Run FYST with OverheadModel() defaults and the legacy 30 s scheduler step."""
    from astropy.time import TimeDelta
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.overhead import OverheadModel, generate_timeline

    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration-s 必须是有限正数")
    return generate_timeline(
        patches=[patch],
        site=get_fyst_site(),
        start_time=start,
        end_time=start + TimeDelta(duration_s, format="sec"),
        overhead_model=OverheadModel(),
        time_step=30,
    )


def write_timeline_ecsv(timeline, output_dir):
    """Write the full timeline and science-only schedule with native FYST IO.

    Return the schedule path, or None when FYST scheduled no science.
    """
    from fyst_trajectories.overhead import write_timeline

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    write_timeline(timeline, folder / "full_timeline.ecsv")
    if not timeline.science_blocks:
        return None
    source = folder / "schedule.ecsv"
    write_timeline(replace(timeline, blocks=timeline.science_blocks), source)
    return source


def reconstruct_trajectories(source, science):
    """Re-read the written schedule, preserving native scan/subscan reconstruction."""
    from fyst_trajectories.overhead import read_timeline, schedule_to_trajectories

    science_timeline = read_timeline(source)
    pairs = schedule_to_trajectories(science_timeline)
    expected = [(b.scan_index, b.subscan_index) for b in science]
    actual = [(b.scan_index, b.subscan_index) for b, _ in pairs]
    if actual != expected:
        raise RuntimeError(f"原生 reconstruction 跳过 blocks：{expected=}, {actual=}")

    return science_timeline, pairs


def toast_boresight_roll_deg(el_deg, site, instrument_rotation_deg=0.0):
    """TOAST roll from FYST's mechanical rotation: -(s*El + instrument_rotation)."""
    from fyst_trajectories import InstrumentOffset, compute_focal_plane_rotation

    if not math.isfinite(instrument_rotation_deg):
        raise ValueError("instrument_rotation_deg must be finite")
    offset = InstrumentOffset(
        dx=0.0, dy=0.0, instrument_rotation=instrument_rotation_deg
    )
    # FYST +x is TOAST local -Y, so the same physical rotation has opposite sign.
    # Keep parallactic_angle=0: TOAST's azel_to_radec handles the sky rotation.
    return -compute_focal_plane_rotation(el_deg, site=site, offset=offset)


def build_trajectory_table(
    case, science_timeline, pairs, source, instrument_rotation_deg=0.0
):
    """Keep native arrays, units and half-open block slices; do not resample."""
    import numpy as np
    from astropy import units as u
    from astropy.table import QTable, vstack
    from fyst_trajectories import get_absolute_times

    origin = science_timeline.start_time
    chunks, blocks = [], []
    offset = 0
    for block, planned in pairs:
        trajectory = planned.trajectory
        absolute = get_absolute_times(trajectory)
        chunk = QTable()
        chunk[schema.ECSV_TIME_S] = (absolute - origin).sec * u.s
        chunk[schema.ECSV_BLOCK_TIME_S] = trajectory.times * u.s
        chunk[schema.ECSV_TIMESTAMP_UNIX_S] = absolute.unix * u.s
        chunk[schema.ECSV_AZ_DEG] = trajectory.az * u.deg
        chunk[schema.ECSV_EL_DEG] = trajectory.el * u.deg
        chunk[schema.ECSV_AZ_VELOCITY] = trajectory.az_vel * u.deg / u.s
        chunk[schema.ECSV_EL_VELOCITY] = trajectory.el_vel * u.deg / u.s
        chunk[schema.ECSV_SCAN_FLAG] = trajectory.scan_flag
        chunk[schema.ECSV_SCAN_INDEX] = np.full(
            len(chunk), block.scan_index, dtype=np.int32
        )
        chunk[schema.ECSV_SUBSCAN_INDEX] = np.full(
            len(chunk), block.subscan_index, dtype=np.int32
        )
        chunk[schema.ECSV_BORESIGHT_ROLL_DEG] = (
            np.asarray(
                toast_boresight_roll_deg(
                    trajectory.el, science_timeline.site, instrument_rotation_deg
                ),
                dtype=np.float64,
            )
            * u.deg
        )
        if len(chunk) < 2:
            raise RuntimeError(
                "science block 少于两个 samples，无法计算 native timestep"
            )
        chunks.append(chunk)
        blocks.append(
            {
                schema.ECSV_SCAN_INDEX: block.scan_index,
                schema.ECSV_SUBSCAN_INDEX: block.subscan_index,
                schema.ECSV_FIRST: offset,
                schema.ECSV_LAST: offset + len(chunk),
                schema.ECSV_START_S: float((block.t_start - origin).sec),
                schema.ECSV_STOP_S: float((block.t_stop - origin).sec),
                schema.ECSV_FIRST_SAMPLE_S: float(chunk[schema.ECSV_TIME_S][0].value),
                schema.ECSV_LAST_SAMPLE_S: float(chunk[schema.ECSV_TIME_S][-1].value),
                schema.ECSV_SAMPLES: len(chunk),
                schema.ECSV_NATIVE_TIMESTEP_S: float(
                    np.median(np.diff(trajectory.times))
                ),
                schema.ECSV_REQUESTED_TIMESTEP_S: float(
                    block.metadata.get("scan_params", {}).get("timestep", 0.1)
                ),
            }
        )
        offset += len(chunk)
    samples = vstack(chunks)
    if not np.all(np.diff(samples[schema.ECSV_TIMESTAMP_UNIX_S].value) > 0):
        raise RuntimeError("原生 timestamps 必须严格递增")
    samples.meta = {
        schema.ECSV_CASE: case,
        schema.ECSV_COORDINATE_FRAME: "mount AltAz",
        schema.ECSV_START_UTC: origin.utc.isot,
        schema.ECSV_GENERATOR: "fyst_trajectories.overhead.schedule_to_trajectories",
        schema.ECSV_SOURCE_SCHEDULE_SHA256: sha256(source),
        schema.ECSV_BLOCKS: blocks,
        schema.ECSV_TIME_NOTE: "time_s since timeline start; block_time_s is unchanged Trajectory.times",
        schema.ROLL_SOURCE_KEY: schema.ROLL_SOURCE_FYST,
        schema.ROLL_CONVENTION_KEY: schema.ROLL_CONVENTION,
        schema.ROLL_NASMYTH_PORT_KEY: science_timeline.site.nasmyth_port,
        schema.ROLL_INSTRUMENT_ROTATION_DEG_KEY: float(instrument_rotation_deg),
    }
    return samples


def _initial_report(case, timeline, start, duration_s, iers_path):
    import fyst_trajectories
    from fyst_trajectories.overhead import io as overhead_io

    science = timeline.science_blocks
    report = {
        "case": case,
        "status": "scheduled",
        "fyst": {
            "__version__": fyst_trajectories.__version__,
            "__file__": fyst_trajectories.__file__,
            "overhead_io_file": overhead_io.__file__,
            "overhead_io_sha256": sha256(Path(overhead_io.__file__)),
        },
        "overhead_model": asdict(timeline.overhead_model),
        "window_start_utc": start.utc.isot,
        "window_stop_utc": timeline.end_time.utc.isot,
        "window_s": float(duration_s),
        "scheduler_time_step_s": 30,
        "iers_file": str(iers_path) if iers_path else None,
        "iers_sha256": sha256(iers_path) if iers_path else None,
        "iers_auto_download": False,
        "block_counts": dict(Counter(str(b.block_type) for b in timeline.blocks)),
        "science_blocks": [
            {
                "scan_index": b.scan_index,
                "subscan_index": b.subscan_index,
                "start_utc": b.t_start.utc.isot,
                "stop_utc": b.t_stop.utc.isot,
                "duration_s": b.duration,
                "samples": None,
            }
            for b in science
        ],
        "total_science_samples": 0,
        "flag_value_counts": {},
    }

    return report


def generate_trajectory(
    case,
    output_dir,
    *,
    start=None,
    duration_s=DEFAULT_DURATION_S,
    instrument_rotation_deg=0.0,
    iers_file=None,
    overwrite=False,
):
    """Write the four generation artifacts and return their report.

    IERS / ephemeris configuration belongs to the caller. ``iers_file`` records
    the caller's selected table and hash; it does not install that table.
    With no science, only the full timeline and a no_science report are written.
    ``instrument_rotation_deg`` is the finite mechanical mounting zero point;
    the Nasmyth port comes from the written schedule.
    """
    import numpy as np
    from astropy.time import Time

    if not math.isfinite(instrument_rotation_deg):
        raise ValueError("instrument_rotation_deg must be finite")
    patch = build_legacy_patch(case)
    start = Time(start or DEFAULT_START[case], scale="utc", precision=3)
    folder = Path(output_dir).expanduser().resolve()
    outputs = [folder / name for name in OUTPUT_NAMES]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("输出已存在；使用新的 output_dir 或显式 overwrite。")
    iers_path = Path(iers_file).expanduser().resolve() if iers_file else None
    if iers_path is not None:
        if not iers_path.is_file():
            raise FileNotFoundError(f"找不到 IERS 文件：{iers_path}")
        if iers_path in outputs:
            raise ValueError("输出不能覆盖 IERS 输入文件")

    timeline = generate_observing_timeline(patch, start, duration_s)
    report = _initial_report(case, timeline, start, duration_s, iers_path)
    if overwrite:
        # A no-science run must not leave a stale trajectory; never touch H5.
        for path in outputs:
            path.unlink(missing_ok=True)
    source = write_timeline_ecsv(timeline, folder)
    report_path = folder / "report.json"
    if source is None:
        report["status"] = "no_science"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        return report
    report["source_schedule_sha256"] = sha256(source)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    science_timeline, pairs = reconstruct_trajectories(source, timeline.science_blocks)
    samples = build_trajectory_table(
        case, science_timeline, pairs, source, instrument_rotation_deg
    )
    samples.write(folder / "fyst_trajectory.ecsv", format="ascii.ecsv")
    for summary, block in zip(
        report["science_blocks"], samples.meta[schema.ECSV_BLOCKS], strict=True
    ):
        summary.update(block)
    values, counts = np.unique(samples[schema.ECSV_SCAN_FLAG], return_counts=True)
    report["flag_value_counts"] = {
        str(int(value)): int(count) for value, count in zip(values, counts, strict=True)
    }
    report["total_science_samples"] = len(samples)
    report["status"] = "complete"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
