r"""Quick-start example: YAML patch → native schedule → per-sample ECSV.

Examples (from the repository root):
  python scripts/patch_to_ecsv.py ces  # or daisy, pong
  python scripts/patch_to_ecsv.py --patch-file examples/patches/ces.yaml \
      --start 2026-08-02T01:25:00.000 --duration-s 10800
  python scripts/patch_to_ecsv.py ces --with-calibrations
  python scripts/patch_to_ecsv.py ces --sample-rate 5

YAML accepts ObservingPatch arguments and an optional module: NAME (e.g. IM1)
to set the PrimeCam detector offset. Templates are in examples/patches/.
Minimal CES (angles in deg, velocity in deg/s, timestep in s):
  name: my CES
  ra_center: 330
  dec_center: -50
  width: 30
  height: 8
  scan_type: constant_el
  velocity: 1
  elevation: 50
  scan_params: {timestep: 0.1, az_accel: 1.0}

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
    science_only_overhead,
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


def _load_patch_with_module(path):
    """Read the patch and its optional PrimeCam module label together."""
    import yaml
    from fyst_trajectories import get_primecam_offset
    from fyst_trajectories.overhead import ObservingPatch

    path = Path(path).expanduser()
    try:
        with path.open() as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, Mapping):
            raise ValueError("expected a YAML mapping of ObservingPatch arguments")
        if "module" in data:
            module = data.pop("module")
            if not isinstance(module, str):
                raise ValueError("module must be a string")
            try:
                offset = get_primecam_offset(module)
            except KeyError as error:
                raise ValueError(f"module: {error.args[0]}") from error
            return ObservingPatch(**data, detector_offset=offset), module
        return ObservingPatch(**data), None
    except (OSError, yaml.YAMLError, TypeError, ValueError, KeyError) as error:
        raise ValueError(f"Cannot load patch {path}: {error}") from error


def load_patch(path):
    """Load an ObservingPatch from YAML, resolving its optional module."""
    patch, _ = _load_patch_with_module(path)
    return patch


def example_patch(case):
    """Read one of the quick-start YAML templates."""
    if case not in DEFAULT_START:
        raise ValueError(f"Unknown case: {case}")
    return load_patch(PATCH_DIR / f"{case}.yaml")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        help="Override the patch timestep (FYST default: 10 Hz). The scheduler accepts at most "
             "20 Hz (Go TCS /path minimum interval: 0.05 s).",
    )
    parser.add_argument(
        "--with-calibrations", action="store_true", help="Use FYST's default calibrations"
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
        patch, module = _load_patch_with_module(patch_path)
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
    # Offline runs may use bundled predictions older than Astropy's age limit.
    iers.conf.auto_max_age = None
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
    if args.with_calibrations:
        from fyst_trajectories.overhead import OverheadModel

        overhead_model = OverheadModel()
    else:
        overhead_model = science_only_overhead()
    try:
        report = generate_trajectory(
            patch,
            folder,
            start=start,
            duration_s=args.duration_s,
            sample_rate_hz=args.sample_rate,
            iers_file=iers_path,
            overwrite=args.overwrite,
            overhead_model=overhead_model,
            detector_label=module,
        )
    except ValueError as error:
        # FYST uses a plain ValueError for the scheduler's sample interval limit.
        if not str(error).startswith("patches whose sample interval is under the Go TCS /path minimum of "):
            raise
        parser.error(str(error))
    report["iers_auto_max_age"] = iers.conf.auto_max_age
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] == "no_science":
        raise SystemExit(
            f"No science in this window; saved full_timeline.ecsv and {folder / 'report.json'}"
        )
    print(json.dumps(report, indent=2))
    print(f"Output: {folder}")


if __name__ == "__main__":
    main()
