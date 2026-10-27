# Architecture and data contracts

The implementation defines the contract in
[trajectory_h5.py](../src/fyst_to_toast/trajectory_h5.py) and
[sim_fyst.py](../src/fyst_to_toast/sim_fyst.py).

## Data flow and dependency boundary

```text
generate.py -> FYST scheduler -> full_timeline.ecsv + schedule.ecsv
            -> FYST trajectory reconstruction -> fyst_trajectory.ecsv
save_trajectory_h5.py -> native TOAST SaveHDF5 -> one H5 per block + files.json
SimFYST -> loader -> validated arrays -> caller Telescope -> TOAST Observations
```

Only [generate.py](../src/fyst_to_toast/generate.py) imports FYST within the runtime
package. The writer, loader, and SimFYST consume files without importing FYST;
examples and optional contract tests may import it directly.
The library does not set MPI/thread environment variables or configure IERS.
Callers configure these before numerical imports and coordinate transformations.

## Generation and ECSV

The generator supports CES, Daisy, and Pong. It calls the native FYST scheduler
with default overhead settings, then reconstructs science trajectories from the
written schedule. It preserves scan/subscan IDs, samples, and block boundaries.
The default window is 10800 seconds, including overhead. A window without science
produces only the full timeline and a report; the CLI exits unsuccessfully.

| ECSV columns | Units and meaning |
|---|---|
| `timestamp_unix_s` | Absolute UTC Unix seconds |
| `time_s` | Seconds since timeline start |
| `block_time_s` | Original trajectory-relative seconds |
| `az_deg`, `el_deg` | Mount AltAz degrees |
| `az_velocity`, `el_velocity` | Degrees per second |
| `scan_flag` | Original FYST enum |
| `scan_index`, `subscan_index` | Original integer IDs |
| `boresight_roll_deg` | Per-sample float64 roll in degrees |

Metadata records `case`, `coordinate_frame`, `start_utc`, `generator`,
`source_schedule_sha256`, `time_note`, `blocks`, `roll_source`,
`roll_convention`, and `roll_nasmyth_port`.
Each block records IDs, half-open `first/last` array slices, sample count,
nominal start/stop, actual first/last sample times, and native/requested timesteps.
Absolute sample times retain their precision; generated UTC text uses milliseconds.

Roll is `-nasmyth_sign * El`, excluding module clocking and parallactic rotation.
The convention `toast_q0_rz_local_z_v1` is
`qa.from_lonlat_angles(-az, el, 0) * Rz(roll)`.
Clocking belongs in each module's detector quaternions as `Rz(-c) * Q_fp`.
An all-zero roll skips quaternion multiplication to preserve unrotated bytes.

The writer CLI checks the schedule hash, nonempty contiguous block coverage,
unique output names, and strictly increasing timestamps. It reads site coordinates
from schedule metadata. It does not split blocks, fill gaps, or resample.

## H5 arrays and metadata

The writer uses native TOAST `SaveHDF5`, with one Observation per file named
`fyst_<case>_<scan:03d>_<subscan:03d>.h5`. Its five-detector focalplane is illustrative.
It saves no detector signal or detector flags (`detdata=[]`).
The CLI writes `files.json` as a list of H5 basenames and a separate writer report.

Default writer output contains 18 shared arrays:

| Arrays | Meaning / units |
|---|---|
| `times`, `azimuth`, `elevation` | Unix seconds; radians; radians |
| `fyst_time_s`, `fyst_block_time_s` | Preserved relative seconds |
| `fyst_az_deg`, `fyst_el_deg` | Original degree arrays |
| `fyst_az_velocity_deg_s`, `fyst_el_velocity_deg_s` | Original degrees per second |
| `fyst_scan_flag`, `scan_index`, `subscan_index` | Raw enum and IDs |
| `boresight_roll` | Float64 radians, shape `(N,)` |
| `flags` | Derived uint8 TOAST bitmask |
| `boresight_azel`, `boresight_radec` | Quaternions, shape `(N, 4)` |
| `position`, `velocity` | TOAST site motion in km and km/s |

Shared datasets do not carry unit attributes; units are part of this contract.
JSON `trajectory_info` records the input hash, case, start time, block, roll
source/convention, sample rate, and adapter policies.
JSON `shared_array_sha256` hashes each shared array's contiguous bytes.

Per-sample ECSV roll takes precedence; a nonzero `--roll-deg` alongside it is rejected.
Legacy ECSV without roll uses the CLI's scalar `--roll-deg` (default zero).
Legacy H5 without `boresight_roll` uses `trajectory_info.roll_deg`, converted to radians
and broadcast by the loader; source metadata is preserved.

## Validation

The loader uses native `LoadHDF5(detdata=[], force_serial=True)`, requiring one
Observation with at least two samples. It checks required arrays for finite,
one-dimensional, equal-length values; block sample count; valid metadata JSON;
and raw flags in `{0, 1, 2, 3}`. If a hash manifest exists, every loaded shared
array must have a matching hash.
Present per-sample roll must be float64, finite, length `N`, with the supported
convention and a nonempty source label.

Times must strictly increase and follow a uniform clock. The maximum phase
residual against an endpoint-derived grid must be at most
`max(1e-9, 2 * max(spacing(abs(times))))` seconds.
The actual rate is `(N-1)/(times[-1]-times[0])`; the grid is only a check.
The loader returns owned copies of the 12 input arrays plus roll, then clears
temporary TOAST storage.

SimFYST additionally checks the caller's nominal sample rate:
`abs(nominal-actual)/actual <= 1e-6`. The caller and source sites must be within
1 mm in Earth-centered Cartesian coordinates. Neither check alters input samples.

## SimFYST trajectory mode

Supply either `trajectory_file` for one H5 or `trajectory_files` for H5 paths
and/or JSON manifests. Manifest entries resolve relative to that manifest.
Lists reject duplicate paths, directories, glob patterns, and nested manifests.
Inputs must have unique Observation names and nonoverlapping sample time ranges.

All inputs are validated first, then sorted by their first sample time.
One Observation with the caller's telescope name, UID and focalplane is built per
native block. Construction is all-or-nothing: failure clears newly created
Observations and leaves `data.obs` empty.
The caller must provide empty Data, a GroundSite, and a nonempty focalplane,
with no MPI communicator and world/group sizes of one.

SimFYST preserves input arrays, recomputes geometry, and derives shared flags
and intervals. It initializes detector signal/flags for the caller's detectors.
`trajectory_boresight_angle=None` uses input roll; a finite scalar angular Quantity
overrides it for all samples and records that choice.
`trajectory_info` and `source_shared_array_sha256` retain source provenance;
`simfyst_info` records current geometry settings, rate checks, roll, and classifications.
Multi-file input also records the original expanded input index.
With `weather` set, each block gets a copied site and a TOAST `SimWeather` at its
sample-time midpoint, with SimGround's name/file fallback, `realization`, `max_pwv`
and `median_weather` settings. That site is used for geometry and the observation
Telescope. `weather=None` preserves the caller site and any existing weather;
the caller Telescope and site are not modified.
SimFYST has no schedule-driven path: it raises `ValueError` without trajectory input,
and rejects unknown or removed SimGround parameters (schedule, scan rates, el_mod,
elnod, HWP, distribution, direction and Sun options) with `TypeError`.
Schedule-driven CES uses TOAST's native `SimGround`.

## Flags, intervals, and limitations

Default TOAST masks map science enum 1 to 0, turnaround enum 2 to 4, and
unclassified/retune enums 0/3 to 1; the turnaround mask is configurable.
The original enum is retained separately. SimFYST builds `scanning`, `turnaround`,
and `fyst_invalid` intervals from contiguous integer `[first,last)` spans.
These three classes partition the samples; direction and Sun intervals are not derived.

SimFYST rejects detector filtering and MPI communicators.
Irregular clocks and MPI are unsupported.
Real focalplane geometry, module clocking, independent sky astrometry, and scan
dynamics are not validated by this adapter. FYST dynamics warnings require separate
assessment; successful file conversion is not a scan-feasibility check.
