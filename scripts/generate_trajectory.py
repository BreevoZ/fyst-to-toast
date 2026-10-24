"""Call FYST: scheduler inputs → native schedule → per-sample ECSV.

Adapted from the author's earlier local scheduling and trajectory examples.

Call native APIs and save their arrays without rescheduling, splitting, or resampling.
"""

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

from fyst_to_toast.generate import (
    DEFAULT_DURATION_S,
    DEFAULT_START,
    OUTPUT_NAMES,
    generate_trajectory,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["ces", "daisy", "pong"])
    parser.add_argument(
        "--start", help="UTC ISO time; defaults to the reference date for this case"
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"Scheduling window including overhead; default {DEFAULT_DURATION_S} s",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Default: output/generated/<case>/ in this repository"
    )
    parser.add_argument(
        "--iers-file", type=Path, help="Offline IERS-A table; defaults to Astropy bundled data"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacing this script's four output files"
    )
    args = parser.parse_args()
    if not math.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error("duration-s must be finite and positive")
    folder = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else Path(__file__).resolve().parents[1] / "output/generated" / args.case
    )
    outputs = [folder / name for name in OUTPUT_NAMES]
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

    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "error"
    if iers_path is not None:
        iers.earth_orientation_table.set(iers.IERS_A.open(str(iers_path)))
    solar_system_ephemeris.set("builtin")
    try:
        start = Time(args.start or DEFAULT_START[args.case], scale="utc", precision=3)
    except ValueError as error:
        parser.error(f"Cannot parse UTC start: {error}")
    report = generate_trajectory(
        args.case,
        folder,
        start=start,
        duration_s=args.duration_s,
        iers_file=iers_path,
        overwrite=args.overwrite,
    )
    if report["status"] == "no_science":
        raise SystemExit(
            f"No science in this window; saved full_timeline.ecsv and {folder / 'report.json'}"
        )
    print(json.dumps(report, indent=2))
    print(f"Output: {folder}")


if __name__ == "__main__":
    main()
