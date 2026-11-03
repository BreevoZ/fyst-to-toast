"""Quick-start example: YAML patch → native schedule → per-sample ECSV.

Copy a template from examples/patches/ and pass --patch-file with --start.
Users may also build an ObservingPatch in Python and call generate_trajectory
without editing src/.

Adapted from the author's earlier local scheduling and trajectory examples.

Call native APIs and save their arrays without rescheduling, splitting, or resampling.
"""

import argparse
from collections.abc import Mapping
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


REPO = Path(__file__).resolve().parents[1]
PATCH_DIR = REPO / "examples/patches"
DEFAULT_DURATION_S = 10800
DEFAULT_START = {
    "ces": "2026-08-02T01:25:00",
    "daisy": "2026-03-15T04:00:00",
    # FYST docs/trajectory_examples.rst:143 (f095c51).
    "pong": "2026-03-15T01:00:00",
}


def load_patch(path):
    """Load ObservingPatch constructor arguments from a YAML mapping."""
    import yaml
    from fyst_trajectories.overhead import ObservingPatch

    path = Path(path).expanduser()
    try:
        with path.open() as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, Mapping):
            raise ValueError("expected a YAML mapping of ObservingPatch arguments")
        return ObservingPatch(**data)
    except (OSError, yaml.YAMLError, TypeError, ValueError, KeyError) as error:
        raise ValueError(f"Cannot load patch {path}: {error}") from error


def example_patch(case):
    """Read one of the quick-start YAML templates."""
    if case not in DEFAULT_START:
        raise ValueError(f"Unknown case: {case}")
    return load_patch(PATCH_DIR / f"{case}.yaml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case", nargs="?", choices=["ces", "daisy", "pong"],
        help="Shortcut for examples/patches/<case>.yaml; use this or --patch-file",
    )
    parser.add_argument("--patch-file", type=Path, help="YAML ObservingPatch; requires --start")
    parser.add_argument(
        "--start", help="UTC ISO time; required with --patch-file, otherwise the case reference date"
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"Scheduling window including overhead; default {DEFAULT_DURATION_S} s",
    )
    parser.add_argument(
        "--sample-rate", type=float, default=None, metavar="HZ",
        help="Override sampling rate; default: patch timestep or FYST default (templates: 10 Hz)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Default: output/generated/<case or YAML stem>/ in this repository"
    )
    parser.add_argument(
        "--iers-file", type=Path, help="Offline IERS-A table; defaults to Astropy bundled data"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacing this script's four output files"
    )
    args = parser.parse_args()
    if (args.case is None) == (args.patch_file is None):
        parser.error("Exactly one of case or --patch-file is required")
    if args.patch_file is not None and args.start is None:
        parser.error("--start is required with --patch-file")
    if not math.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error("duration-s must be finite and positive")
    if args.sample_rate is not None and (not math.isfinite(args.sample_rate) or args.sample_rate <= 0):
        parser.error("--sample-rate must be finite and positive")
    patch_path = (
        args.patch_file.expanduser().resolve()
        if args.patch_file is not None
        else PATCH_DIR / f"{args.case}.yaml"
    )
    try:
        patch = load_patch(patch_path)
    except ValueError as error:
        parser.error(str(error))
    folder = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else REPO / "output/generated" / (args.case or patch_path.stem)
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
        start = Time(
            args.start if args.start is not None else DEFAULT_START[args.case],
            scale="utc", precision=3,
        )
    except ValueError as error:
        parser.error(f"Cannot parse UTC start: {error}")
    report = generate_trajectory(
        patch,
        folder,
        start=start,
        duration_s=args.duration_s,
        sample_rate_hz=args.sample_rate,
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
