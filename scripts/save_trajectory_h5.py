"""Write an existing FYST trajectory ECSV as TOAST Observation HDF5.

Adapted from the author's earlier local FYST example (source SHA-256 ca2c9b3e…).
Preserve native blocks, samples, five illustrative detectors, and the TOAST SaveHDF5 schema.
This script is independent of SimFYST; it neither generates motion nor implements its H5 reader.
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
        "trajectory", type=Path, help="Input fyst_trajectory.ecsv"
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        help="Native timeline ECSV; defaults to schedule.ecsv beside the trajectory",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Default: hdf5/ beside the input ECSV"
    )
    parser.add_argument(
        "--iers-file",
        type=Path,
        help="Optional offline IERS-A table; defaults to Astropy bundled data",
    )
    parser.add_argument(
        "--roll-deg",
        type=float,
        default=0.0,
        help="Constant local boresight roll for legacy ECSV only; default 0 deg",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing matching H5 files and manifests; keep other files",
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
            parser.error(f"Input file not found: {path}")
    if args.iers_file is not None and not args.iers_file.expanduser().is_file():
        parser.error(f"IERS file not found: {args.iers_file}")

    # Configure serial execution for this CLI; the package leaves MPI / thread settings unchanged.
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

    from fyst_to_toast.trajectory_h5 import (
        ECSV_BORESIGHT_ROLL_DEG,
        ROLL_CONVENTION,
        ROLL_CONVENTION_KEY,
        ROLL_SOURCE_KEY,
        _roll_from_ecsv,
        _trajectory_arrays_from_ecsv,
        _validate_trajectory_arrays,
        write_trajectory_blocks,
    )

    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "error"
    if args.iers_file:
        iers.earth_orientation_table.set(
            iers.IERS_A.open(str(args.iers_file.expanduser().resolve()))
        )
    solar_system_ephemeris.set("builtin")

    if not np.isfinite(args.roll_deg):
        parser.error("roll must be finite")
    samples = QTable.read(source)
    try:
        _, roll_source = _roll_from_ecsv(samples, args.roll_deg)
    except ValueError as error:
        parser.error(str(error))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    schedule_hash = hashlib.sha256(schedule.read_bytes()).hexdigest()
    if samples.meta.get("source_schedule_sha256") != schedule_hash:
        parser.error(
            "Trajectory and schedule SHA-256 mismatch; use inputs from the same generation run."
        )
    case = samples.meta.get("case", "")
    if not isinstance(case, str) or re.fullmatch(r"[A-Za-z0-9_]+", case) is None:
        parser.error(
            "trajectory metadata.case must be a filename-safe pattern name, e.g. ces / daisy"
        )
    if not samples.meta.get("blocks") or not len(samples):
        parser.error("Trajectory has no blocks / samples to write")

    # Check that blocks cover the input before writing, without repartitioning.
    offset, names = 0, []
    for block in samples.meta["blocks"]:
        if block["first"] != offset or not offset < block["last"] <= len(samples):
            parser.error("Block [first,last) slices must be nonempty and cover the input contiguously")
        offset = block["last"]
        names.append(
            f"fyst_{case}_{block['scan_index']:03d}_{block['subscan_index']:03d}.h5"
        )
    if offset != len(samples) or len(set(names)) != len(names):
        parser.error("Blocks must cover all samples; scan/subscan pairs must produce unique filenames")
    # Validate every block before creating the output directory or writing files.
    for block in samples.meta["blocks"]:
        try:
            part = samples[block["first"] : block["last"]]
            _validate_trajectory_arrays(_trajectory_arrays_from_ecsv(part), block)
        except (KeyError, TypeError, ValueError) as error:
            parser.error(f"Block {block['scan_index']}/{block['subscan_index']}: {error}")
    if not np.all(np.diff(samples["timestamp_unix_s"].to_value(u.s)) > 0):
        parser.error("timestamps must be strictly increasing")

    folder = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else source.parent / "hdf5"
    )
    expected_paths = [
        folder / name for name in [*names, "files.json", "writer_report.json"]
    ]
    if not args.overwrite and any(path.exists() for path in expected_paths):
        parser.error(
            "Outputs exist; choose a new --output-dir or specify --overwrite."
        )
    if source in expected_paths or schedule in expected_paths:
        parser.error("Outputs must not overwrite input files")
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
        "roll_deg": (
            args.roll_deg if ECSV_BORESIGHT_ROLL_DEG not in samples.colnames else None
        ),
        ROLL_SOURCE_KEY: roll_source,
        ROLL_CONVENTION_KEY: ROLL_CONVENTION,
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
        f"Written: {len(files)} blocks → {len(files)} TOAST H5 files, {len(samples)} samples."
    )
    print("Instrument: 5 illustrative detectors; no detector signal saved.")
    print(f"Output directory: {folder}")
    print(f"File manifest: {folder / 'files.json'}")


if __name__ == "__main__":
    main()
