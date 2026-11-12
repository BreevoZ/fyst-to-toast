"""Export one fixed-azimuth-window CES using FYST's planner directly."""

import argparse
import json
import math
import os
from pathlib import Path
from tempfile import gettempdir

# Configure serial execution before NumPy / TOAST / FYST imports.
os.environ["MPI_DISABLE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from fyst_to_toast.generate import generate_window_ces


REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--az-window", type=float, nargs=2, required=True, metavar=("LO", "HI"))
    parser.add_argument("--elevation", type=float, required=True, metavar="DEG")
    parser.add_argument("--velocity", type=float, required=True, metavar="DEG_PER_S")
    parser.add_argument("--start", required=True, metavar="UTC", help="UTC ISO time (milliseconds)")
    parser.add_argument("--duration-s", type=float, required=True, metavar="S")
    parser.add_argument("--sample-rate", type=float, metavar="HZ", help="Default: FYST timestep")
    parser.add_argument("--az-accel", type=float, default=1.0, help="Azimuth acceleration (deg/s²)")
    parser.add_argument("--module", metavar="NAME", help="FYST PrimeCam module name")
    parser.add_argument(
        "--field", type=float, nargs=4, metavar=("RA", "DEC", "WIDTH", "HEIGHT"),
        help="Field in degrees, for FYST drift warnings",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Default: output/generated/window_ces/ in this repository",
    )
    parser.add_argument(
        "--iers-file", type=Path, help="Offline IERS-A table; defaults to Astropy bundled data",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacing this script's three output files",
    )
    args = parser.parse_args()
    for name in ("az_window", "elevation", "velocity", "duration_s", "sample_rate", "az_accel", "field"):
        value = getattr(args, name)
        values = value if isinstance(value, list) else [value]
        if any(item is not None and not math.isfinite(item) for item in values):
            parser.error(f"--{name.replace('_', '-')} must be finite")

    folder = (
        args.output_dir.expanduser().resolve()
        if args.output_dir else REPO / "output/generated/window_ces"
    )
    outputs = [folder / name for name in ("schedule.ecsv", "fyst_trajectory.ecsv", "report.json")]
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error("Outputs exist; choose a new --output-dir or specify --overwrite.")
    iers_path = args.iers_file.expanduser().resolve() if args.iers_file else None
    if iers_path is not None:
        if not iers_path.is_file():
            parser.error(f"IERS file not found: {iers_path}")
        if iers_path in outputs:
            parser.error("Outputs must not overwrite the input IERS file")

    for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        default_cache = Path(gettempdir()) / "fyst-to-toast" / key.lower()
        cache = Path(os.environ.setdefault(key, str(default_cache)))
        cache.mkdir(parents=True, exist_ok=True)

    from astropy.coordinates import solar_system_ephemeris
    from astropy.time import Time
    from astropy.utils import iers
    from fyst_trajectories import get_primecam_offset
    from fyst_trajectories.planning import FieldRegion

    iers.conf.auto_download = False
    # Offline runs may use bundled predictions older than Astropy's age limit.
    iers.conf.auto_max_age = None
    iers.conf.iers_degraded_accuracy = "error"
    solar_system_ephemeris.set("builtin")
    try:
        start = Time(args.start, scale="utc", precision=3)
    except ValueError as error:
        parser.error(f"Cannot parse UTC start: {error}")
    try:
        offset = get_primecam_offset(args.module) if args.module is not None else None
    except KeyError as error:
        # FYST supplies the valid names, including any upstream aliases.
        parser.error(str(error.args[0]))
    try:
        field = FieldRegion(*args.field) if args.field is not None else None
        if iers_path is not None:
            iers.earth_orientation_table.set(iers.IERS_A.open(str(iers_path)))
        report = generate_window_ces(
            folder, az_window=args.az_window, elevation=args.elevation,
            velocity=args.velocity, start=start, duration_s=args.duration_s,
            sample_rate_hz=args.sample_rate, az_accel=args.az_accel,
            detector_offset=offset, detector_label=args.module, field=field,
            iers_file=iers_path, overwrite=args.overwrite,
        )
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    report["iers_auto_max_age"] = iers.conf.auto_max_age
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Output: {folder}")


if __name__ == "__main__":
    main()
