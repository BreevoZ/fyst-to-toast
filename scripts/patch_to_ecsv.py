"""Quick-start example: scheduler inputs → native schedule → per-sample ECSV.

Users may write their own scripts that build an ObservingPatch and call
generate_trajectory without editing src/.

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
    OUTPUT_NAMES,
    generate_trajectory,
)


DEFAULT_DURATION_S = 10800
DEFAULT_SAMPLE_RATE_HZ = 10.0
DEFAULT_START = {
    "ces": "2026-08-02T01:25:00",
    "daisy": "2026-03-15T04:00:00",
    # FYST docs/trajectory_examples.rst:143 (f095c51).
    "pong": "2026-03-15T01:00:00",
}


def example_patch(case, sample_rate_hz):
    """Build a quick-start CES / Daisy or FYST's documented Pong patch."""
    from fyst_trajectories.overhead import ObservingPatch

    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    timestep = 1.0 / sample_rate_hz
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
            scan_params={"timestep": timestep, "az_accel": 1.0},
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
            scan_params={"radius": 1, "turn_radius": 0.3, "timestep": timestep},
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
                "timestep": timestep,
            },
        )
    return patch


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
        "--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE_HZ, metavar="HZ",
        help=f"Requested trajectory sampling rate; default {DEFAULT_SAMPLE_RATE_HZ:g} Hz",
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
    if not math.isfinite(args.sample_rate) or args.sample_rate <= 0:
        parser.error("--sample-rate must be finite and positive")
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
    patch = example_patch(args.case, args.sample_rate)
    report = generate_trajectory(
        patch,
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
