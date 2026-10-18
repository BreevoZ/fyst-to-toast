"""Private, serial input adapter for one native writer block; no SimFYST wiring yet.

The caller configures MPI / thread settings before importing numerical libraries.
This module neither changes that environment nor constructs a runtime instrument.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from astropy.units import Quantity


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


def _load_trajectory_block(path: str | Path) -> _TrajectoryBlock:
    """Read and validate one writer H5 without changing any input array bytes.

    LoadHDF5 decodes the existing TOAST schema. All loaded shared arrays are
    checked against the optional source manifest; only the 12 inputs are returned.
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
        for key in _TRAJECTORY_FIELDS:
            if key not in obs.shared:
                raise ValueError(f"Missing trajectory field: {key}")
            values = np.copy(obs.shared[key].data)
            if values.ndim != 1:
                raise ValueError(f"{key}: expected a 1-D array, got {values.shape}")
            try:
                finite = np.isfinite(values).all()
            except TypeError as error:
                raise ValueError(f"{key}: expected finite numeric values") from error
            if not finite:
                raise ValueError(f"{key}: all values must be finite")
            arrays[key] = values
        n_samples = len(arrays["times"])
        if n_samples < 2:
            raise ValueError(f"Expected N >= 2 samples, got {n_samples}")
        for key, values in arrays.items():
            if len(values) != n_samples:
                raise ValueError(f"{key}: length {len(values)} != N={n_samples}")

        raw_info = obs.get("trajectory_info")
        try:
            info = json.loads(raw_info)
        except (TypeError, ValueError) as error:
            raise ValueError("trajectory_info must be valid JSON") from error
        if not isinstance(info, dict) or not isinstance(info.get("block"), dict):
            # This is malformed file content, not a caller argument type error.
            raise ValueError("trajectory_info must contain a block object")  # noqa: TRY004
        if info["block"].get("samples") != n_samples:
            raise ValueError(f"trajectory_info block.samples must equal N={n_samples}")

        unknown = np.setdiff1d(arrays["fyst_scan_flag"], [0, 1, 2, 3])
        if unknown.size:
            raise ValueError(f"Unknown fyst_scan_flag values: {unknown.tolist()}")
        actual_rate, residual, tolerance = _validate_clock(arrays["times"])

        manifest = None
        if "shared_array_sha256" in obs:
            try:
                manifest = json.loads(obs["shared_array_sha256"])
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
        )
    finally:
        data.clear()


def _validate_clock(times: np.ndarray) -> tuple[float, float, float]:
    """Check finite 1-D times (N >= 2); return rate Hz, residual s, tolerance s.

    The reference grid is only a validation aid, never a replacement for times.
    Unix float quantization permits up to two ULPs (and at least 1 ns).
    """
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
