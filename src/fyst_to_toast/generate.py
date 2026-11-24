"""Native FYST scheduling and per-sample ECSV generation.

Adapted from the author's earlier local scheduling and trajectory examples.
Only this module imports FYST, lazily. Callers configure IERS / ephemeris and
MPI / thread settings; importing the package never configures the process.
"""

import hashlib
import json
import math
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

from fyst_to_toast import trajectory_h5 as schema

SCAN_TYPE_LABELS = {"constant_el": "ces", "daisy": "daisy", "pong": "pong"}
OUTPUT_NAMES = (
    "full_timeline.ecsv",
    "schedule.ecsv",
    "fyst_trajectory.ecsv",
    "report.json",
)


def sha256(path):
    """Return a file's byte hash for provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def _capture_fyst_warnings():
    """Record warnings that pass caller filters, forwarding their normal display."""
    records, entries = [], {}
    original_showwarning = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        if category.__module__.split(".", 1)[0] == "fyst_trajectories":
            key = (category.__name__, str(message))
            if key not in entries:
                entry = {"category": key[0], "message": key[1], "count": 0}
                entries[key] = entry
                records.append(entry)
            entries[key]["count"] += 1
        original_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = showwarning
    try:
        yield records
    finally:
        warnings.showwarning = original_showwarning


def science_only_overhead():
    """Disable calibrations while retaining FYST's settling and scan limits."""
    from fyst_trajectories.overhead import OverheadModel

    return replace(
        OverheadModel(),
        retune_duration=0.0,
        pointing_cal_duration=0.0,
        focus_duration=0.0,
        skydip_duration=0.0,
        planet_cal_duration=0.0,
        beam_map_duration=0.0,
    )


def generate_observing_timeline(patch, start, duration_s, overhead_model=None):
    """Run the FYST scheduler with the given overhead and a 30 s scheduler step.

    ``overhead_model=None`` means FYST's default calibrations, as in FYST.
    Pass ``science_only_overhead()`` for science-only scheduling.
    """
    from astropy.time import TimeDelta
    from fyst_trajectories import get_fyst_site
    from fyst_trajectories.overhead import OverheadModel, generate_timeline

    if overhead_model is not None and not isinstance(overhead_model, OverheadModel):
        raise TypeError("overhead_model must be a fyst_trajectories.overhead.OverheadModel")
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration-s must be finite and positive")
    return generate_timeline(
        patches=[patch],
        site=get_fyst_site(),
        start_time=start,
        end_time=start + TimeDelta(duration_s, format="sec"),
        overhead_model=overhead_model,
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
        raise RuntimeError(f"Native reconstruction skipped blocks: {expected=}, {actual=}")

    return science_timeline, pairs


def toast_boresight_roll_deg(el_deg, site):
    """TOAST roll from FYST's Nasmyth rotation: -nasmyth_sign*El."""
    from fyst_trajectories import InstrumentOffset, compute_focal_plane_rotation

    # FYST +x is TOAST local -Y, so the same physical rotation has opposite sign.
    # Keep parallactic_angle=0: TOAST's azel_to_radec handles the sky rotation.
    # Module clocking belongs in focalplane detector quaternions, not roll.
    return -compute_focal_plane_rotation(
        el_deg, site=site, offset=InstrumentOffset(dx=0.0, dy=0.0)
    )


def build_trajectory_table(
    label, science_timeline, pairs, source, *,
    generator="fyst_trajectories.overhead.schedule_to_trajectories",
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
                    trajectory.el, science_timeline.site
                ),
                dtype=np.float64,
            )
            * u.deg
        )
        if len(chunk) < 2:
            raise RuntimeError(
                "Science block has fewer than two samples; cannot compute native timestep"
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
                    planned.config.timestep
                ),
            }
        )
        offset += len(chunk)
    samples = vstack(chunks)
    if not np.all(np.diff(samples[schema.ECSV_TIMESTAMP_UNIX_S].value) > 0):
        raise RuntimeError("Native timestamps must be strictly increasing")
    samples.meta = {
        schema.ECSV_CASE: label,
        schema.ECSV_COORDINATE_FRAME: "mount AltAz",
        schema.ECSV_START_UTC: origin.utc.isot,
        schema.ECSV_GENERATOR: generator,
        schema.ECSV_SOURCE_SCHEDULE_SHA256: sha256(source),
        schema.ECSV_BLOCKS: blocks,
        schema.ECSV_TIME_NOTE: "time_s since timeline start; block_time_s is unchanged Trajectory.times",
        schema.ROLL_SOURCE_KEY: schema.ROLL_SOURCE_FYST,
        schema.ROLL_CONVENTION_KEY: schema.ROLL_CONVENTION,
        schema.ROLL_NASMYTH_PORT_KEY: science_timeline.site.nasmyth_port,
    }
    return samples


def _initial_report(
    patch, label, timeline, start, duration_s, iers_path
):
    from astropy.utils import iers

    import fyst_trajectories
    from fyst_trajectories.overhead import OverheadModel, io as overhead_io

    if timeline.overhead_model == science_only_overhead():
        overhead_preset = "science_only"
    elif timeline.overhead_model == OverheadModel():
        overhead_preset = "fyst_default"
    else:
        overhead_preset = "custom"
    science = timeline.science_blocks
    timestep = patch.scan_params.get("timestep")
    report = {
        "case": label,
        "patch_name": patch.name,
        "scan_type": patch.scan_type,
        "status": "scheduled",
        "fyst": {
            "__version__": fyst_trajectories.__version__,
            "__file__": fyst_trajectories.__file__,
            "overhead_io_file": overhead_io.__file__,
            "overhead_io_sha256": sha256(Path(overhead_io.__file__)),
        },
        "overhead_model": asdict(timeline.overhead_model),
        "overhead_preset": overhead_preset,
        "window_start_utc": start.utc.isot,
        "window_stop_utc": timeline.end_time.utc.isot,
        "window_s": float(duration_s),
        "requested_sample_rate_hz": 1.0 / float(timestep) if timestep is not None else None,
        "scheduler_time_step_s": 30,
        "iers_file": str(iers_path) if iers_path else None,
        "iers_sha256": sha256(iers_path) if iers_path else None,
        "iers_auto_download": iers.conf.auto_download,
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
    patch,
    output_dir,
    *,
    start,
    duration_s,
    sample_rate_hz=None,
    overhead_model=None,
    iers_file=None,
    overwrite=False,
):
    """Generate artifacts from a FYST ObservingPatch and return their report.

    ``start`` is an Astropy Time or UTC ISO string; ``duration_s`` is the
    scheduling window including overhead. Both must be supplied by the caller.

    ``overhead_model=None`` means FYST's default calibrations, as in FYST.
    Pass ``science_only_overhead()`` to zero calibration durations while keeping
    slews, settling and scan limits, or another ``OverheadModel`` instance for
    custom overhead.

    IERS / ephemeris configuration belongs to the caller. ``iers_file`` records
    the caller's selected table and hash; it does not install that table.
    With no science, only the full timeline and a no_science report are written.
    ``sample_rate_hz`` overrides the timestep on a copy, leaving ``patch``
    unchanged. If omitted, use the patch timestep or FYST's default.
    """
    import numpy as np
    from astropy.time import Time

    from fyst_trajectories.overhead import ObservingPatch, OverheadModel

    if not isinstance(patch, ObservingPatch):
        raise TypeError("patch must be a fyst_trajectories.overhead.ObservingPatch")
    if overhead_model is not None and not isinstance(overhead_model, OverheadModel):
        raise TypeError("overhead_model must be a fyst_trajectories.overhead.OverheadModel")
    try:
        label = SCAN_TYPE_LABELS[patch.scan_type]
    except KeyError:
        raise ValueError(f"Unknown scan_type: {patch.scan_type}") from None
    if sample_rate_hz is not None:
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be finite and positive")
        patch = replace(
            patch, scan_params={**dict(patch.scan_params), "timestep": 1.0 / sample_rate_hz}
        )
    start = Time(start, scale="utc", precision=3)
    folder = Path(output_dir).expanduser().resolve()
    outputs = [folder / name for name in OUTPUT_NAMES]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("Outputs exist; choose a new output_dir or specify overwrite.")
    iers_path = Path(iers_file).expanduser().resolve() if iers_file else None
    if iers_path is not None:
        if not iers_path.is_file():
            raise FileNotFoundError(f"IERS file not found: {iers_path}")
        if iers_path in outputs:
            raise ValueError("Outputs must not overwrite the input IERS file")

    with _capture_fyst_warnings() as fyst_warnings:
        timeline = generate_observing_timeline(patch, start, duration_s, overhead_model)
        report = _initial_report(patch, label, timeline, start, duration_s, iers_path)
        report["fyst_warnings"] = fyst_warnings
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
            label, science_timeline, pairs, source
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


def generate_window_ces(
    output_dir,
    *,
    az_window,
    elevation,
    velocity,
    start,
    duration_s,
    sample_rate_hz=None,
    az_accel=1.0,
    detector_offset=None,
    detector_label=None,
    field=None,
    iers_file=None,
    overwrite=False,
):
    """Export a native fixed-azimuth-window CES without running the scheduler.

    ``start`` is an Astropy Time or UTC ISO string. FYST validates the planner
    inputs and rounds ``duration_s`` to whole scan legs; output bounds use the
    actual first and last samples. ``detector_offset`` is a FYST InstrumentOffset
    or None, and ``field`` is an optional FieldRegion for FYST's drift warnings.
    Optional ``detector_label`` records the module name in the report and schedule.
    Omit ``sample_rate_hz`` to retain FYST's default timestep.

    The schedule records the commanded boresight envelope. Its ``rising`` flag
    is true when the requested window midpoint modulo 360 is in (0, 180):
    elevation rises east of the meridian. It is a summary of this planner's
    output, not a scheduler reconstruction recipe; use the exported trajectory.
    No full_timeline.ecsv is written. IERS / ephemeris configuration belongs to
    the caller; ``iers_file`` only records the selected table and its hash.
    """
    import numpy as np
    from astropy.time import Time
    from astropy.utils import iers

    import fyst_trajectories
    from fyst_trajectories import get_absolute_times, get_fyst_site
    from fyst_trajectories.overhead import (
        CalibrationPolicy, ObservingTimeline, TimelineBlock, write_timeline,
        io as overhead_io,
    )
    from fyst_trajectories.planning import plan_constant_el_scan

    sampling = {}
    if sample_rate_hz is not None:
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be finite and positive")
        sampling["timestep"] = 1.0 / sample_rate_hz
    start = Time(start, scale="utc", precision=3)
    folder = Path(output_dir).expanduser().resolve()
    outputs = [folder / name for name in (
        "schedule.ecsv", "fyst_trajectory.ecsv", "report.json",
    )]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("Outputs exist; choose a new output_dir or specify overwrite.")
    iers_path = Path(iers_file).expanduser().resolve() if iers_file else None
    if iers_path is not None:
        if not iers_path.is_file():
            raise FileNotFoundError(f"IERS file not found: {iers_path}")
        if iers_path in outputs:
            raise ValueError("Outputs must not overwrite the input IERS file")

    planner = "fyst_trajectories.planning.plan_constant_el_scan"
    site = get_fyst_site()
    with _capture_fyst_warnings() as fyst_warnings:
        planned = plan_constant_el_scan(
            field, az_window=az_window, elevation=elevation, velocity=velocity,
            duration=duration_s, site=site, start_time=start, az_accel=az_accel,
            detector_offset=detector_offset, **sampling,
        )
        trajectory = planned.trajectory
        absolute = get_absolute_times(trajectory)
        window = [float(value) for value in az_window]
        offset = asdict(detector_offset) if detector_offset is not None else None
        block = TimelineBlock(
            t_start=absolute[0], t_stop=absolute[-1], block_type="science",
            patch_name="window_ces", az_start=float(np.min(trajectory.az)),
            az_end=float(np.max(trajectory.az)), elevation=float(trajectory.el[0]),
            scan_index=0, subscan_index=0, scan_type="constant_el",
            rising=0.0 < (sum(window) / 2.0) % 360.0 < 180.0,
            az_final=float(trajectory.az[-1]),
        )
        timeline = ObservingTimeline(
            blocks=[block], site=site, start_time=block.t_start, end_time=block.t_stop,
            overhead_model=science_only_overhead(), calibration_policy=CalibrationPolicy(),
            metadata={
                "planner": planner,
                "requested_az_window": window,
                "requested_duration_s": float(duration_s),
                "detector_offset": {key: offset[key] for key in ("dx", "dy")} if offset else None,
            },
        )
        report = {
            "case": "ces_window",
            "status": "complete",
            "fyst": {
                "__version__": fyst_trajectories.__version__,
                "__file__": fyst_trajectories.__file__,
                "overhead_io_file": overhead_io.__file__,
                "overhead_io_sha256": sha256(Path(overhead_io.__file__)),
            },
            "planner": planner,
            "planner_inputs": {
                "az_window": window, "elevation": float(elevation),
                "velocity": float(velocity), "start_utc": start.utc.isot,
                "duration_s": float(duration_s), "az_accel": float(az_accel),
                "sample_rate_hz": float(sample_rate_hz) if sample_rate_hz is not None else None,
                "timestep": float(planned.config.timestep),
                "field": asdict(field) if field is not None else None,
            },
            "computed_params": dict(planned.computed_params),
            "detector_offset": offset,
            "first_sample_utc": block.t_start.utc.isot,
            "last_sample_utc": block.t_stop.utc.isot,
            "end_time_note": "FYST's end_time_iso is the requested end; the actual end is the last sample.",
            "fyst_warnings": fyst_warnings,
            "iers_file": str(iers_path) if iers_path else None,
            "iers_sha256": sha256(iers_path) if iers_path else None,
            "iers_auto_download": iers.conf.auto_download,
        }
        if detector_label is not None:
            timeline.metadata["detector_label"] = detector_label
            report["detector_label"] = detector_label
        if overwrite:
            for path in outputs:
                path.unlink(missing_ok=True)
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / "schedule.ecsv"
        write_timeline(timeline, source)
        samples = build_trajectory_table(
            "ces_window", timeline, [(block, planned)], source, generator=planner,
        )
        samples.write(folder / "fyst_trajectory.ecsv", format="ascii.ecsv")
        values, counts = np.unique(samples[schema.ECSV_SCAN_FLAG], return_counts=True)
        report["flag_value_counts"] = {
            str(int(value)): int(count) for value, count in zip(values, counts, strict=True)
        }
        report["total_science_samples"] = len(samples)
        report["science_blocks"] = samples.meta[schema.ECSV_BLOCKS]
        report["source_schedule_sha256"] = sha256(source)
        (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
