r"""Replay one fixed-azimuth-window CES at a given start time and duration.

Use this to reproduce a row of an observing plan (azimuth window, elevation,
start, duration), such as a line of the Prime-Cam season schedule. FYST's
planner runs directly: no scheduler, calibrations or wait for the field's pass.
To let FYST choose the time, use patch_to_ecsv.py with az_min/az_max.

Examples (from the repository root):
  python scripts/window_ces_to_ecsv.py \
      --window-file examples/windows/deep56_e50_rising.yaml \
      --start 2026-08-13T03:48:45.147 --duration-s 14400

YAML keys (see examples/windows/):
  az_window: [LO, HI]  required, deg
  elevation: DEG      required, deg
  velocity: DEG/S     required, azimuth coordinate rate; sky speed = v*cos(El)
  az_accel: DEG/S^2   optional, default 1.0
  timestep: S         optional, FYST default; --sample-rate HZ overrides it
  module: NAME        optional, FYST PrimeCam module (e.g. IM1)
  field:              optional, used only for FYST drift warnings; all in deg
    ra_center: RA
    dec_center: DEC
    width: W
    height: H
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

from fyst_to_toast.generate import generate_window_ces


REPO = Path(__file__).resolve().parents[1]


def load_window(path):
    """Validate a YAML scan definition and resolve its FYST planner arguments."""
    import yaml
    from fyst_trajectories import get_primecam_offset
    from fyst_trajectories.planning import FieldRegion

    def check_keys(data, required, optional=(), prefix=""):
        if not isinstance(data, Mapping):
            raise ValueError(f"{prefix or 'scan definition'} must be a YAML mapping")
        unknown = data.keys() - set(required) - set(optional)
        missing = set(required) - data.keys()
        errors = []
        if unknown:
            errors.append("unknown keys: " + ", ".join(sorted(f"{prefix}{key!r}" for key in unknown)))
        if missing:
            errors.append("missing keys: " + ", ".join(sorted(f"{prefix}{key}" for key in missing)))
        if errors:
            raise ValueError("; ".join(errors))

    def number(value, key, positive=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or (positive and value <= 0):
            raise ValueError(f"{key} must be finite" + (" and positive" if positive else ""))

    path = Path(path).expanduser()
    try:
        data = yaml.safe_load(path.read_text())
        check_keys(data, ("az_window", "elevation", "velocity"),
                   ("az_accel", "timestep", "module", "field"))
        window = data["az_window"]
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError("az_window must be a sequence of two numbers")
        for index, value in enumerate(window):
            number(value, f"az_window[{index}]")
        number(data["elevation"], "elevation")
        number(data["velocity"], "velocity", positive=True)
        number(data.get("az_accel", 1.0), "az_accel", positive=True)
        if "timestep" in data:
            number(data["timestep"], "timestep", positive=True)
        offset = None
        if "module" in data:
            if not isinstance(data["module"], str):
                raise ValueError("module must be a string")
            try:
                offset = get_primecam_offset(data["module"])
            except KeyError as error:
                # FYST supplies the valid names, including any upstream aliases.
                raise ValueError(f"module: {error.args[0]}") from error
        field = None
        if "field" in data:
            check_keys(data["field"], ("ra_center", "dec_center", "width", "height"),
                       prefix="field.")
            for key, value in data["field"].items():
                number(value, f"field.{key}")
            try:
                field = FieldRegion(**data["field"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"field: {error}") from error
        return {
            "az_window": window, "elevation": data["elevation"],
            "velocity": data["velocity"], "az_accel": data.get("az_accel", 1.0),
            "sample_rate_hz": 1.0 / data["timestep"] if "timestep" in data else None,
            "detector_offset": offset, "detector_label": data.get("module"), "field": field,
        }
    except (OSError, UnicodeError, yaml.YAMLError, TypeError, ValueError) as error:
        raise ValueError(f"Cannot load window {path}: {error}") from error


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--window-file", type=Path, required=True, metavar="PATH", help="YAML scan definition",
    )
    parser.add_argument("--start", required=True, metavar="UTC", help="UTC ISO time (milliseconds)")
    parser.add_argument("--duration-s", type=float, required=True, metavar="S")
    parser.add_argument(
        "--max-block-s", type=float, metavar="S",
        help="Split the built scan into equal blocks of at most S seconds; "
             "omit or set S >= the built duration to keep one identical block",
    )
    parser.add_argument(
        "--sample-rate", type=float, metavar="HZ",
        help="Override YAML timestep; otherwise FYST default if absent",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Default: output/generated/<YAML stem>/ in this repository",
    )
    parser.add_argument(
        "--iers-file", type=Path, help="Offline IERS-A table; defaults to Astropy bundled data",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacing this script's three output files",
    )
    args = parser.parse_args()
    for name in ("duration_s", "sample_rate", "max_block_s"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    window_path = args.window_file.expanduser().resolve()
    try:
        window = load_window(window_path)
    except ValueError as error:
        parser.error(str(error))
    if args.sample_rate is not None:
        window["sample_rate_hz"] = args.sample_rate

    folder = (
        args.output_dir.expanduser().resolve()
        if args.output_dir else REPO / "output/generated" / window_path.stem
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
        if iers_path is not None:
            iers.earth_orientation_table.set(iers.IERS_A.open(str(iers_path)))
        report = generate_window_ces(
            folder, **window, start=start, duration_s=args.duration_s,
            max_block_s=args.max_block_s,
            iers_file=iers_path, overwrite=args.overwrite,
        )
    except (ValueError, TypeError, OSError) as error:
        parser.error(f"Window {window_path}: {error}")
    report["iers_auto_max_age"] = iers.conf.auto_max_age
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Output: {folder}")


if __name__ == "__main__":
    main()
