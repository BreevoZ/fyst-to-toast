"""Serial trajectory HDF5 writer and block loader for SimFYST.

The caller configures MPI / thread settings before importing numerical libraries.
The caller also configures IERS before writing. The writer retains the five-detector
demo instrument; the loader returns trajectory inputs, not a runtime instrument.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from astropy.table import QTable
    from astropy.units import Quantity


# Per-sample ECSV contract shared by the generator and writer.
ECSV_TIME_S = "time_s"
ECSV_BLOCK_TIME_S = "block_time_s"
ECSV_TIMESTAMP_UNIX_S = "timestamp_unix_s"
ECSV_AZ_DEG = "az_deg"
ECSV_EL_DEG = "el_deg"
ECSV_AZ_VELOCITY = "az_velocity"
ECSV_EL_VELOCITY = "el_velocity"
ECSV_SCAN_FLAG = "scan_flag"
ECSV_SCAN_INDEX = "scan_index"
ECSV_SUBSCAN_INDEX = "subscan_index"
ECSV_BORESIGHT_ROLL_DEG = "boresight_roll_deg"
# Shared ECSV / H5 metadata and the TOAST shared-array name (radians).
H5_BORESIGHT_ROLL = "boresight_roll"
ROLL_SOURCE_KEY = "roll_source"
ROLL_CONVENTION_KEY = "roll_convention"
ROLL_NASMYTH_PORT_KEY = "roll_nasmyth_port"
ROLL_CONVENTION = "toast_q0_rz_local_z_v1"
ROLL_SOURCE_PLACEHOLDER_ZERO = "placeholder_zero"
ROLL_SOURCE_FYST = "fyst_focal_plane_rotation"
ROLL_SOURCE_LEGACY_SCALAR = "legacy_scalar"
ROLL_SOURCE_OVERRIDE = "trajectory_boresight_angle"
ECSV_CASE = "case"
ECSV_COORDINATE_FRAME = "coordinate_frame"
ECSV_START_UTC = "start_utc"
ECSV_GENERATOR = "generator"
ECSV_SOURCE_SCHEDULE_SHA256 = "source_schedule_sha256"
ECSV_BLOCKS = "blocks"
ECSV_TIME_NOTE = "time_note"
ECSV_FIRST = "first"
ECSV_LAST = "last"
ECSV_START_S = "start_s"
ECSV_STOP_S = "stop_s"
ECSV_FIRST_SAMPLE_S = "first_sample_s"
ECSV_LAST_SAMPLE_S = "last_sample_s"
ECSV_SAMPLES = "samples"
ECSV_NATIVE_TIMESTEP_S = "native_timestep_s"
ECSV_REQUESTED_TIMESTEP_S = "requested_timestep_s"

TRAJECTORY_COLUMNS = (
    ECSV_TIME_S,
    ECSV_BLOCK_TIME_S,
    ECSV_TIMESTAMP_UNIX_S,
    ECSV_AZ_DEG,
    ECSV_EL_DEG,
    ECSV_AZ_VELOCITY,
    ECSV_EL_VELOCITY,
    ECSV_SCAN_FLAG,
    ECSV_SCAN_INDEX,
    ECSV_SUBSCAN_INDEX,
    ECSV_BORESIGHT_ROLL_DEG,
)

TRAJECTORY_META_KEYS = (
    ECSV_CASE,
    ECSV_COORDINATE_FRAME,
    ECSV_START_UTC,
    ECSV_GENERATOR,
    ECSV_SOURCE_SCHEDULE_SHA256,
    ECSV_BLOCKS,
    ECSV_TIME_NOTE,
    ROLL_SOURCE_KEY,
    ROLL_CONVENTION_KEY,
    ROLL_NASMYTH_PORT_KEY,
)

TRAJECTORY_BLOCK_KEYS = (
    ECSV_SCAN_INDEX,
    ECSV_SUBSCAN_INDEX,
    ECSV_FIRST,
    ECSV_LAST,
    ECSV_START_S,
    ECSV_STOP_S,
    ECSV_FIRST_SAMPLE_S,
    ECSV_LAST_SAMPLE_S,
    ECSV_SAMPLES,
    ECSV_NATIVE_TIMESTEP_S,
    ECSV_REQUESTED_TIMESTEP_S,
)


_TRAJECTORY_FIELDS = (
    "times",
    "azimuth",
    "elevation",
    "fyst_time_s",
    "fyst_block_time_s",
    "fyst_az_deg",
    "fyst_el_deg",
    "fyst_az_velocity_deg_s",
    "fyst_el_velocity_deg_s",
    "fyst_scan_flag",
    "scan_index",
    "subscan_index",
)
_TRAJECTORY_ARRAY_FIELDS = (*_TRAJECTORY_FIELDS, H5_BORESIGHT_ROLL)


_TRAJECTORY_INFO_KEY = "trajectory_info"
_SHARED_ARRAY_SHA256_KEY = "shared_array_sha256"


@dataclass(frozen=True)
class _TrajectoryBlock:
    """Owned copies, independent of temporary TOAST storage.

    Frozen prevents attribute replacement, not edits inside arrays / dictionaries.
    Times are Unix seconds, azimuth/elevation radians; FYST units are in field names.
    source_site contains only lat/lon/height quantities, never the demo focalplane.
    """

    observation_name: str
    arrays: dict[str, np.ndarray]
    trajectory_info: dict
    trajectory_info_json: str
    source_shared_array_sha256: dict[str, str] | None
    source_site: dict[str, Quantity]
    actual_rate_hz: float
    max_phase_residual_s: float
    time_tolerance_s: float
    roll_source: str
    roll_convention: str


def _validate_trajectory_arrays(arrays, block):
    """Validate native input arrays and block count for both CLI and H5 loading."""
    for key in _TRAJECTORY_FIELDS:
        if key not in arrays:
            raise ValueError(f"Missing trajectory field: {key}")
    for key, values in arrays.items():
        if values.ndim != 1:
            raise ValueError(f"{key}: expected a 1-D array, got {values.shape}")
        try:
            finite = np.isfinite(values).all()
        except TypeError as error:
            raise ValueError(f"{key}: expected finite numeric values") from error
        if not finite:
            raise ValueError(f"{key}: all values must be finite")
    n_samples = len(arrays["times"])
    if n_samples < 2:
        raise ValueError(f"Expected N >= 2 samples, got {n_samples}")
    for key, values in arrays.items():
        if len(values) != n_samples:
            raise ValueError(f"{key}: length {len(values)} != N={n_samples}")
    if block.get("samples") != n_samples:
        raise ValueError(f"trajectory_info block.samples must equal N={n_samples}")
    unknown = np.setdiff1d(arrays["fyst_scan_flag"], [0, 1, 2, 3])
    if unknown.size:
        raise ValueError(f"Unknown fyst_scan_flag values: {unknown.tolist()}")
    return _validate_clock(arrays["times"])


def _trajectory_arrays_from_ecsv(samples):
    """Convert ECSV columns to the loader's native input names and units."""
    from astropy import units as u

    return dict(zip(_TRAJECTORY_FIELDS, (
        samples[ECSV_TIMESTAMP_UNIX_S].to_value(u.s),
        samples[ECSV_AZ_DEG].to_value(u.rad),
        samples[ECSV_EL_DEG].to_value(u.rad),
        samples[ECSV_TIME_S].to_value(u.s),
        samples[ECSV_BLOCK_TIME_S].to_value(u.s),
        samples[ECSV_AZ_DEG].to_value(u.deg),
        samples[ECSV_EL_DEG].to_value(u.deg),
        samples[ECSV_AZ_VELOCITY].to_value(u.deg / u.s),
        samples[ECSV_EL_VELOCITY].to_value(u.deg / u.s),
        np.asarray(samples[ECSV_SCAN_FLAG]),
        np.asarray(samples[ECSV_SCAN_INDEX]),
        np.asarray(samples[ECSV_SUBSCAN_INDEX]),
    ), strict=True))


def _load_trajectory_block(path: str | Path) -> _TrajectoryBlock:
    """Read and validate one writer H5 without changing any input array bytes.

    LoadHDF5 decodes the existing TOAST schema. All loaded shared arrays are
    checked against the optional source manifest. Return the 12 original inputs
    plus roll, synthesized from roll_deg only for legacy H5 without the array.
    Invalid trajectory content raises ValueError. Native IO errors propagate.
    """
    from toast import Data, ops
    from toast.mpi import Comm

    data = Data(Comm(world=None))
    try:
        ops.LoadHDF5(
            files=[str(Path(path).expanduser().resolve())],
            detdata=[],
            force_serial=True,
        ).apply(data)
        if len(data.obs) != 1:
            raise ValueError(f"Expected exactly one observation, got {len(data.obs)}")
        obs = data.obs[0]
        arrays = {}
        fields = list(_TRAJECTORY_FIELDS)
        if H5_BORESIGHT_ROLL in obs.shared:
            fields.append(H5_BORESIGHT_ROLL)
        for key in fields:
            if key not in obs.shared:
                raise ValueError(f"Missing trajectory field: {key}")
            arrays[key] = np.copy(obs.shared[key].data)

        raw_info = obs.get(_TRAJECTORY_INFO_KEY)
        try:
            info = json.loads(raw_info)
        except (TypeError, ValueError) as error:
            raise ValueError("trajectory_info must be valid JSON") from error
        if not isinstance(info, dict) or not isinstance(info.get("block"), dict):
            # This is malformed file content, not a caller argument type error.
            raise ValueError("trajectory_info must contain a block object")  # noqa: TRY004
        actual_rate, residual, tolerance = _validate_trajectory_arrays(arrays, info["block"])
        n_samples = len(arrays["times"])

        if H5_BORESIGHT_ROLL in arrays:
            _validate_boresight_roll(arrays[H5_BORESIGHT_ROLL], n_samples)
            roll_source = _validate_roll_metadata(info)
        else:
            try:
                scalar = float(info["roll_deg"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "trajectory_info.roll_deg must be a finite scalar angle"
                ) from error
            if not np.isfinite(scalar):
                raise ValueError(
                    "trajectory_info.roll_deg must be a finite scalar angle"
                )
            arrays[H5_BORESIGHT_ROLL] = np.full(
                n_samples, np.deg2rad(scalar), dtype=np.float64
            )
            roll_source = ROLL_SOURCE_LEGACY_SCALAR

        manifest = None
        if _SHARED_ARRAY_SHA256_KEY in obs:
            try:
                manifest = json.loads(obs[_SHARED_ARRAY_SHA256_KEY])
            except (TypeError, ValueError) as error:
                raise ValueError("shared_array_sha256 must be valid JSON") from error
            if not isinstance(manifest, dict):
                raise ValueError("shared_array_sha256 must be a JSON object")
            for key in obs.shared:
                if key not in manifest:
                    raise ValueError(f"shared_array_sha256: missing hash for {key}")
                digest = hashlib.sha256(
                    np.ascontiguousarray(obs.shared[key].data).tobytes()
                ).hexdigest()
                if digest != manifest[key]:
                    raise ValueError(f"shared_array_sha256: hash mismatch for {key}")

        location = obs.telescope.site.earthloc
        return _TrajectoryBlock(
            observation_name=obs.name,
            arrays=arrays,
            trajectory_info=info,
            trajectory_info_json=raw_info,
            source_shared_array_sha256=manifest,
            source_site={
                "lat": location.lat.copy(),
                "lon": location.lon.copy(),
                "height": location.height.copy(),
            },
            actual_rate_hz=actual_rate,
            max_phase_residual_s=residual,
            time_tolerance_s=tolerance,
            roll_source=roll_source,
            roll_convention=ROLL_CONVENTION,
        )
    finally:
        data.clear()


def _validate_clock(times: np.ndarray) -> tuple[float, float, float]:
    """Check finite 1-D times (N >= 2); return rate Hz, residual s, tolerance s.

    The reference grid is only a validation aid, never a replacement for times.
    Unix float quantization permits up to two ULPs (and at least 1 ns).
    """
    if times.dtype != np.dtype(np.float64):
        raise ValueError(f"times: expected float64 Unix seconds, got {times.dtype}")
    if not np.all(times[1:] > times[:-1]):
        raise ValueError("times must be strictly increasing")
    actual_rate = float((len(times) - 1) / (times[-1] - times[0]))
    reference = times[0] + np.arange(len(times)) / actual_rate
    residual = float(np.max(np.abs(times - reference)))
    tolerance = max(1e-9, 2 * float(np.max(np.spacing(np.abs(times)))))
    if residual > tolerance:
        raise ValueError(
            f"times must have a uniform clock: max phase residual {residual:.9g} s "
            f"> tolerance {tolerance:.9g} s (gap or jitter)"
        )
    return actual_rate, residual, tolerance


def _check_sample_rate(
    nominal_rate_hz: float, actual_rate_hz: float, *, rtol: float = 1e-6
) -> dict[str, float]:
    """Compare rates expressed in Hz, independently of the clock phase check.

    Return numbers for caller metadata; reject a relative error above rtol.
    No rate or timestamp is changed. Quantity callers should first convert to Hz.
    """
    nominal, actual, tolerance = map(float, (nominal_rate_hz, actual_rate_hz, rtol))
    if not (
        np.isfinite(nominal) and np.isfinite(actual) and nominal > 0 and actual > 0
    ):
        raise ValueError(
            "nominal_rate_hz and actual_rate_hz must be finite and positive"
        )
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("sample-rate rtol must be finite and nonnegative")
    relative_error = abs(nominal - actual) / actual
    if relative_error > tolerance:
        raise ValueError(
            f"Nominal sample rate {nominal:.12g} Hz does not match actual "
            f"{actual:.12g} Hz: relative error {relative_error:.9g} > rtol {tolerance:g}"
        )
    return {
        "nominal_rate_hz": nominal,
        "actual_rate_hz": actual,
        "relative_error": relative_error,
        "rtol": tolerance,
    }


def _validate_roll_metadata(metadata):
    """Reject ambiguous per-sample angles instead of guessing their convention."""
    if metadata.get(ROLL_CONVENTION_KEY) != ROLL_CONVENTION:
        raise ValueError(f"{ROLL_CONVENTION_KEY} must be {ROLL_CONVENTION!r}")
    source = metadata.get(ROLL_SOURCE_KEY)
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{ROLL_SOURCE_KEY} must be a non-empty string")
    return source


def _validate_boresight_roll(values, n_samples):
    """Validate the radians array without changing its dtype or bytes."""
    if values.ndim != 1 or len(values) != n_samples:
        raise ValueError(f"{H5_BORESIGHT_ROLL}: expected shape ({n_samples},)")
    if values.dtype != np.dtype(np.float64):
        raise ValueError(f"{H5_BORESIGHT_ROLL}: expected float64")
    if not np.isfinite(values).all():
        raise ValueError(f"{H5_BORESIGHT_ROLL}: all values must be finite")


def _roll_from_ecsv(samples, roll_deg=0.0):
    """Resolve and validate roll once for all blocks; also used by the CLI."""
    if np.ndim(roll_deg) != 0 or not np.isfinite(roll_deg):
        raise ValueError("roll_deg must be a finite scalar angle")
    if ECSV_BORESIGHT_ROLL_DEG in samples.colnames:
        if roll_deg != 0:
            raise ValueError(
                f"{ECSV_BORESIGHT_ROLL_DEG} cannot be combined with non-default --roll-deg"
            )
        source = _validate_roll_metadata(samples.meta)
        from astropy import units as u

        values = np.asarray(samples[ECSV_BORESIGHT_ROLL_DEG].to_value(u.rad))
    else:
        source = ROLL_SOURCE_LEGACY_SCALAR
        values = np.full(len(samples), np.deg2rad(roll_deg), dtype=np.float64)
    _validate_boresight_roll(values, len(samples))
    return values, source


def _boresight_with_roll(az, el, roll_rad):
    """Q0(-Az, El, 0) right-multiplied by local +Z rotation, in radians."""
    from toast import qarray as qa

    bore = qa.from_lonlat_angles(-az, el, np.zeros_like(el))
    # Preserve Q0 bytes for zero placeholders and legacy zero-roll trajectories.
    if np.any(roll_rad != 0):
        bore = qa.mult(bore, qa.rotation(np.array([0.0, 0.0, 1.0]), roll_rad))
    return bore


def write_trajectory_blocks(
    samples: QTable,
    *,
    site_name: str,
    latitude_deg: float,
    longitude_deg: float,
    height_m: float,
    output_dir: str | Path,
    source_sha256: str,
    roll_deg: float = 0.0,
) -> list[str]:
    """Write in-memory trajectory samples as one native TOAST H5 per block.

    ``samples.meta`` supplies ``case``, ``start_utc`` and the original ``blocks``
    with half-open ``first`` / ``last`` slices and ``native_timestep_s``. Columns
    retain the ECSV units; no samples or block boundaries are regenerated.
    ``source_sha256`` identifies the original trajectory ECSV in H5 metadata.
    Per-sample roll takes precedence; a nonzero legacy roll_deg with that column
    is rejected. Legacy ECSV broadcasts roll_deg. All-zero roll preserves Q0 bytes.

    The caller configures MPI / threads and IERS, validates inputs and overwrite
    policy, and creates the output directory. Return ordered H5 basenames; the
    CLI owns files.json / writer_report.json. No FYST import is needed.
    """
    from astropy import units as u
    from toast import Data, Observation, ops
    from toast.coordinates import azel_to_radec
    from toast.instrument import Focalplane, GroundSite, Session, Telescope
    from toast.instrument_sim import fake_hexagon_focalplane
    from toast.mpi import Comm
    from toast.observation import default_values as defaults

    from fyst_to_toast.fyst_flags import (
        SCAN_FLAG_SCIENCE,
        SCAN_FLAG_TURNAROUND,
        to_toast_flags,
    )

    roll_values, roll_source = _roll_from_ecsv(samples, roll_deg)
    folder = Path(output_dir)
    case = samples.meta[ECSV_CASE]
    # 1. Use TOAST's illustrative five-detector focalplane.
    site = GroundSite(
        site_name,
        latitude_deg * u.deg,
        longitude_deg * u.deg,
        height_m * u.m,
    )
    data = Data(Comm(world=None))
    try:
        files = []
        for block in samples.meta[ECSV_BLOCKS]:
            # Read the preserved block arrays without changing block boundaries.
            part = samples[block[ECSV_FIRST] : block[ECSV_LAST]]
            inputs = _trajectory_arrays_from_ecsv(part)
            times, az, el = (inputs[key] for key in _TRAJECTORY_FIELDS[:3])
            dt = np.diff(part[ECSV_BLOCK_TIME_S].to_value(u.s))
            np.testing.assert_allclose(
                dt, block[ECSV_NATIVE_TIMESTEP_S], rtol=0, atol=1e-9
            )
            rate = (1 / block[ECSV_NATIVE_TIMESTEP_S]) * u.Hz
            full = fake_hexagon_focalplane(n_pix=7, width=0.1 * u.deg, sample_rate=rate)
            fp = Focalplane(
                detector_data=full.detector_data[:5],
                sample_rate=rate,
                field_of_view=full.field_of_view,
            )
            telescope = Telescope("FYST-trajectory-demo", site=site, focalplane=fp)
            name = f"fyst_{case}_{block[ECSV_SCAN_INDEX]:03d}_{block[ECSV_SUBSCAN_INDEX]:03d}"
            session = Session(
                name=name,
                start=datetime.fromtimestamp(times[0], tz=timezone.utc),
                end=datetime.fromtimestamp(times[-1], tz=timezone.utc),
            )
            obs = Observation(
                data.comm, telescope, len(part), name=name, session=session
            )
            data.obs.append(obs)
            files.append(f"{name}.h5")

            # 2. TOAST convention: Az is clockwise, longitude is counterclockwise.
            roll = roll_values[block[ECSV_FIRST] : block[ECSV_LAST]]
            bore = _boresight_with_roll(az, el, roll)
            radec = azel_to_radec(site, times, bore, use_ephem=True)
            position, velocity = site.position_velocity(times)

            # 3. FYST enum → TOAST bitmask; retain raw enums and turnaround samples.
            original_flags = inputs["fyst_scan_flag"]
            flags = to_toast_flags(
                original_flags,
                turnaround_mask=defaults.shared_mask_unstable_scanrate,
                invalid_mask=defaults.shared_mask_invalid,
            )
            # Keep the original shared-column order as well as the array bytes.
            # The loader and writer use the same 12-field definition above.
            arrays = dict(zip(_TRAJECTORY_FIELDS[:3], (times, az, el)))
            arrays.update(
                {
                    "flags": flags,
                    "boresight_azel": bore,
                    "boresight_radec": radec,
                    "position": position,
                    "velocity": velocity,
                }
            )
            arrays.update({key: inputs[key] for key in _TRAJECTORY_FIELDS[3:]})
            arrays[H5_BORESIGHT_ROLL] = roll
            for key, values in arrays.items():
                assert np.isfinite(values).all()
                obs.shared.create_column(key, shape=values.shape, dtype=values.dtype)
                obs.shared[key].set(values, fromrank=0)

            # 4. Record adapter metadata and validation information, not new FYST/TOAST interfaces.
            obs[_TRAJECTORY_INFO_KEY] = json.dumps(
                {
                    "case": case,
                    "source_sha256": source_sha256,
                    "start_utc": samples.meta[ECSV_START_UTC],
                    "block": block,
                    **(
                        {"roll_deg": roll_deg}
                        if ECSV_BORESIGHT_ROLL_DEG not in samples.colnames
                        else {}
                    ),
                    ROLL_SOURCE_KEY: roll_source,
                    ROLL_CONVENTION_KEY: ROLL_CONVENTION,
                    "sample_rate_hz": rate.to_value(u.Hz),
                    "fyst_flag_values": {
                        "scanning": SCAN_FLAG_SCIENCE,
                        "turnaround": SCAN_FLAG_TURNAROUND,
                    },
                    "adapter_note": "One native block per Observation; no resampling or motion generation",
                    "interval_policy": "Rebuild sample-index intervals from flags after loading",
                }
            )
            obs[_SHARED_ARRAY_SHA256_KEY] = json.dumps(
                {
                    key: hashlib.sha256(
                        np.ascontiguousarray(value).tobytes()
                    ).hexdigest()
                    for key, value in arrays.items()
                }
            )

        # 5. Use native SaveHDF5: one file per block, detdata=[].
        ops.SaveHDF5(
            volume=str(folder), volume_index=None, detdata=[], force_serial=True
        ).apply(data)
        if not all((folder / name).is_file() for name in files):
            raise RuntimeError("SaveHDF5 did not produce all expected files")
        return files
    finally:
        data.clear()
