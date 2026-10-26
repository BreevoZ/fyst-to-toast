# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
# TOAST's LICENSE and AUTHORS for this file are in LICENSES/TOAST-BSD-2-Clause.txt and LICENSES/TOAST-AUTHORS.txt.

# Some trait definitions, validators, and Operator scaffolding are adapted from
# TOAST SimGround. The trajectory-file path was added by this project.
# Source: hpc4cmb/toast 9595449a,
# src/toast/ops/sim_ground.py (sha256 eff1ac3ad41a9095670ba5357903581a4da01b1c6a75c37f8f236aef2cf2c4df).

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import traitlets
from astropy import units as u
from astropy.coordinates import EarthLocation

from toast.coordinates import azel_to_radec
from toast.instrument import GroundSite, Session, Telescope
from toast.intervals import IntervalList
from toast.observation import Observation
from toast.observation import default_values as defaults
from toast.ops.operator import Operator
from toast.timing import function_timer
from toast.traits import Bool, Instance, Int, List, Quantity, Unicode, Unit, trait_docs
from toast.utils import Logger, name_UID

from .fyst_flags import (
    SCAN_FLAG_RETUNE,
    SCAN_FLAG_SCIENCE,
    SCAN_FLAG_TURNAROUND,
    SCAN_FLAG_UNCLASSIFIED,
    to_toast_flags,
)
from .trajectory_h5 import (
    H5_BORESIGHT_ROLL,
    ROLL_CONVENTION_KEY,
    ROLL_SOURCE_KEY,
    ROLL_SOURCE_OVERRIDE,
    _TRAJECTORY_ARRAY_FIELDS,
    _TRAJECTORY_FIELDS,
    _TRAJECTORY_INFO_KEY,
    _boresight_with_roll,
    _check_sample_rate,
    _load_trajectory_block,
)


def _mask_to_sample_spans(mask):
    """Return half-open integer spans for the True runs of a 1-D boolean mask."""
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist())
    )


@trait_docs
class SimFYST(Operator):
    """Build CCAT Observations from native FYST trajectory HDF5 blocks.

    Supply trajectory_file or trajectory_files and the caller's Telescope.
    Preserve native samples and block boundaries, recompute geometry, and derive
    motion flags and intervals. Only serial execution is supported.
    Schedule-driven CES simulation belongs to TOAST's native SimGround.
    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    trajectory_file = Unicode(
        None, allow_none=True, help="One native trajectory block in a writer HDF5 file"
    )

    trajectory_files = List(
        [], help="Trajectory HDF5 paths or writer files.json manifests; serial only"
    )

    trajectory_boresight_angle = Quantity(
        None,
        allow_none=True,
        help="Fixed trajectory roll override; None uses the source per-sample roll",
    )

    telescope = Instance(
        klass=Telescope, allow_none=True, help="This must be an instance of a Telescope"
    )

    use_ephem = Bool(
        True,
        help="Use PyEphem to convert between horizontal and equatorial systems",
    )

    use_qpoint = Bool(
        False,
        help="Use qpoint to convert between horizontal and equatorial systems",
    )

    so3g_compat_mode = Bool(
        False,
        help="Implies use_qpoint=True. Tweak weather in azel_to_radec to match "
        "so3g and disable certain qpoint defaults.",
    )

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for common flags",
    )

    det_data = Unicode(
        defaults.det_data,
        allow_none=True,
        help="Observation detdata key to initialize",
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to initialize",
    )

    azimuth = Unicode(defaults.azimuth, help="Observation shared key for Azimuth")

    elevation = Unicode(defaults.elevation, help="Observation shared key for Elevation")

    boresight_azel = Unicode(
        defaults.boresight_azel, help="Observation shared key for boresight AZ/EL"
    )

    boresight_radec = Unicode(
        defaults.boresight_radec, help="Observation shared key for boresight RA/DEC"
    )

    position = Unicode(defaults.position, help="Observation shared key for position")

    velocity = Unicode(defaults.velocity, help="Observation shared key for velocity")

    scanning_interval = Unicode(
        defaults.scanning_interval, help="Interval name for scanning"
    )

    turnaround_interval = Unicode(
        defaults.turnaround_interval, help="Interval name for turnarounds"
    )

    fyst_invalid_interval = Unicode(
        "fyst_invalid", help="Trajectory interval name for unclassified / retune samples"
    )

    turnaround_mask = Int(
        defaults.shared_mask_unstable_scanrate,
        help="Bit mask to raise turnaround flags with",
    )

    @traitlets.validate("telescope")
    def _check_telescope(self, proposal):
        tele = proposal["value"]
        if tele is not None:
            try:
                dets = tele.focalplane.detectors
            except Exception:
                raise traitlets.TraitError(
                    "telescope must be a Telescope instance with a focalplane"
                )
        return tele

    @traitlets.validate("use_ephem")
    def _check_use_ephem(self, proposal):
        log = Logger.get()
        use_ephem = proposal["value"]
        if use_ephem:
            if self.so3g_compat_mode:
                log.warning("so3g_compat_mode overrides use_ephem=True")
            elif self.use_qpoint:
                raise traitlets.TraitError("Cannot use both ephem and qpoint")

        return use_ephem

    @traitlets.validate("use_qpoint")
    def _check_use_qpoint(self, proposal):
        log = Logger.get()
        use_qpoint = proposal["value"]
        if self.so3g_compat_mode:
            if not use_qpoint:
                log.warning("so3g_compat_mode overrides use_qpoint=False")
        elif use_qpoint and self.use_ephem:
            raise traitlets.TraitError("Cannot use both ephem and qpoint")
        if use_qpoint or self.so3g_compat_mode:
            try:
                import qpoint
            except ModuleNotFoundError as e:
                raise RuntimeError(f"Cannot use qpoint: '{e}'")
        return use_qpoint

    @traitlets.validate("so3g_compat_mode")
    def _check_use_qpoint(self, proposal):
        log = Logger.get()
        so3g_compat_mode = proposal["value"]
        if so3g_compat_mode:
            try:
                import qpoint
            except ModuleNotFoundError as e:
                raise RuntimeError(f"Cannot use qpoint: '{e}'")
        return so3g_compat_mode

    def __init__(self, **kwargs):
        unknown = sorted(set(kwargs) - set(self.class_trait_names()))
        if unknown:
            raise TypeError(f"Unknown SimFYST parameter(s): {', '.join(unknown)}")
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        if self.trajectory_file is None and not self.trajectory_files:
            raise ValueError("trajectory_file or trajectory_files must be provided")
        return self._exec_trajectory(data, detectors=detectors)

    def _obs_telescopes(self, data, det_ranks, detectors):
        """Return the caller's telescope without splitting or copying it."""
        comm = data.comm
        if (
            det_ranks != 1
            or comm.comm_world is not None
            or comm.comm_group is not None
            or comm.world_size != 1
            or comm.group_size != 1
            or comm.ngroups != 1
        ):
            raise ValueError(
                "trajectory mode requires serial execution: no MPI, world/group size 1"
            )
        if detectors is not None:
            raise ValueError(
                "trajectory mode does not support detector filtering (detectors)"
            )
        return {"ALL": (self.telescope, None)}

    def _trajectory_shared_names(self):
        """Map native inputs / derived geometry to configured output names."""
        return {
            key: getattr(self, key)
            for key in (
                *_TRAJECTORY_FIELDS[:3],
                "boresight_azel",
                "boresight_radec",
                "position",
                "velocity",
            )
        }

    def _validate_trajectory_config(self, data, detectors):
        """Reject unsupported settings before loading or allocating an observation."""
        if self.trajectory_file is not None and self.trajectory_files:
            raise ValueError("trajectory_file and trajectory_files are mutually exclusive")
        if self.telescope is None:
            raise ValueError("trajectory mode requires the caller's Telescope")
        if not self.telescope.focalplane.detectors:
            raise ValueError("trajectory mode requires a non-empty caller focalplane")
        if not isinstance(self.telescope.site, GroundSite):
            raise TypeError("trajectory mode requires a caller GroundSite")
        if data.obs:
            raise ValueError("trajectory mode requires empty data.obs")
        comm = data.comm
        if (
            comm.comm_world is not None
            or comm.comm_group is not None
            or comm.world_size != 1
            or comm.group_size != 1
            or comm.ngroups != 1
        ):
            raise ValueError(
                "trajectory mode requires serial execution: no MPI, world/group size 1"
            )
        if detectors is not None:
            raise ValueError(
                "trajectory mode does not support detector filtering (detectors)"
            )
        if self.telescope.site.weather is not None:
            raise ValueError("trajectory mode does not support caller site weather")

        names = self._trajectory_shared_names()
        if len(set(names.values())) != len(names):
            raise ValueError("trajectory shared output names must be distinct")
        for native, output in names.items():
            if not output or (output in _TRAJECTORY_ARRAY_FIELDS and output != native):
                raise ValueError(
                    f"trajectory output {native}={output!r} conflicts with original fields"
                )
        if self.det_data is not None and self.det_data == self.det_flags:
            raise ValueError("det_data and det_flags must have distinct names")
        if (
            not self.shared_flags
            or self.shared_flags in _TRAJECTORY_ARRAY_FIELDS
            or self.shared_flags in names.values()
        ):
            raise ValueError(
                "trajectory shared_flags must be a non-empty, distinct output name"
            )
        intervals = (
            self.scanning_interval,
            self.turnaround_interval,
            self.fyst_invalid_interval,
        )
        if len(set(intervals)) != 3 or any(
            not key or key == "ALL_OBSERVATION_SAMPLES" for key in intervals
        ):
            raise ValueError(
                "trajectory interval names must be non-empty, distinct and not reserved"
            )
        if not 0 <= self.turnaround_mask <= np.iinfo(np.uint8).max:
            raise ValueError("trajectory turnaround_mask must fit uint8 (0..255)")

    def _trajectory_sources(self):
        """Expand one level of writer manifests, preserving their listed order."""
        if self.trajectory_file is not None:
            # Preserve the original single-file path and native IO errors.
            return [Path(self.trajectory_file).expanduser().resolve()]

        def file_path(entry, parent=None):
            try:
                path = Path(entry).expanduser()
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid trajectory path {entry!r}") from error
            if any(char in str(path) for char in "*?[]"):
                raise ValueError(f"{path}: trajectory paths do not support glob patterns")
            if parent is not None and not path.is_absolute():
                path = parent / path
            path = path.resolve()
            if not path.is_file():
                raise ValueError(f"{path}: expected a trajectory file, not a directory or missing path")
            return path

        sources = []
        seen = set()
        for entry in self.trajectory_files:
            path = file_path(entry)
            if path.suffix.lower() == ".json":
                try:
                    names = json.loads(path.read_text())
                except (OSError, ValueError) as error:
                    raise ValueError(f"{path}: cannot read trajectory manifest: {error}") from error
                if not isinstance(names, list) or not names or any(
                    not isinstance(name, str) or not name.strip() for name in names
                ):
                    raise ValueError(f"{path}: expected a non-empty JSON list of HDF5 paths")
                expanded = [file_path(name, path.parent) for name in names]
                if any(source.suffix.lower() == ".json" for source in expanded):
                    raise ValueError(f"{path}: nested JSON manifests are not supported")
            else:
                expanded = [path]
            for source in expanded:
                if source in seen:
                    raise ValueError(f"{source}: duplicate trajectory path")
                seen.add(source)
                sources.append(source)
        return sources

    def _load_validate_trajectory(self, source):
        """Keep only validated input arrays and small metadata, not Observations."""
        block = _load_trajectory_block(source)
        telescope = self.telescope
        site = telescope.site
        source_location = EarthLocation.from_geodetic(**block.source_site)
        distance_m = np.linalg.norm(
            [
                (a - b).to_value(u.m)
                for a, b in zip(site.earthloc.geocentric, source_location.geocentric)
            ]
        )
        if not np.isfinite(distance_m) or distance_m > 1e-3:
            raise ValueError(
                f"Caller site must match source site within 1 mm; distance={distance_m:.9g} m"
            )
        rate_check = _check_sample_rate(
            telescope.focalplane.sample_rate.to_value(u.Hz),
            block.actual_rate_hz,
            rtol=1e-6,
        )
        roll_override = None
        if self.trajectory_boresight_angle is not None:
            roll_override = self.trajectory_boresight_angle.to_value(u.rad)
            if np.ndim(roll_override) != 0 or not np.isfinite(roll_override):
                raise ValueError(f"{ROLL_SOURCE_OVERRIDE} must be a finite scalar angle")
        return {
            "source": source,
            "block": block,
            "rate_check": rate_check,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "roll_override": roll_override,
        }

    def _exec_trajectory(self, data, detectors=None):
        """Validate every input first; then build all observations or roll back."""
        self._validate_trajectory_config(data, detectors)
        multiple = bool(self.trajectory_files)
        prepared, names = [], {}
        for index, source in enumerate(self._trajectory_sources()):
            try:
                item = self._load_validate_trajectory(source)
            except Exception as error:
                if not multiple:
                    raise
                raise ValueError(f"{source}: {error}") from error
            name = item["block"].observation_name
            if name in names:
                raise ValueError(
                    f"{source}: duplicate observation name {name!r}, also in {names[name]}"
                )
            names[name] = source
            if multiple:
                item["input_index"] = index
            prepared.append(item)

        prepared.sort(key=lambda item: item["block"].arrays["times"][0])
        for previous, current in zip(prepared, prepared[1:]):
            if current["block"].arrays["times"][0] <= previous["block"].arrays["times"][-1]:
                raise ValueError(
                    f"{current['source']}: sample time interval overlaps {previous['source']}"
                )
        try:
            for item in prepared:
                self._build_trajectory_observation(data, **item)
        except Exception:
            for observation in data.obs:
                observation.clear()
            data.obs.clear()
            raise

    def _build_trajectory_observation(
        self, data, source, block, rate_check, source_sha256, roll_override,
        input_index=None,
    ):
        """Shared single-block construction after input validation (serial only)."""
        telescope = self.telescope
        site = telescope.site
        if roll_override is None:
            roll_source = block.roll_source
            roll_rad = block.arrays[H5_BORESIGHT_ROLL]
        else:
            roll_source = ROLL_SOURCE_OVERRIDE
            roll_rad = np.full(len(block.arrays["times"]), roll_override, dtype=np.float64)

        times, az, el = (block.arrays[key] for key in _TRAJECTORY_FIELDS[:3])
        position, velocity = site.position_velocity(times)
        bore_azel = _boresight_with_roll(az, el, roll_rad)
        backend = {
            "use_ephem": self.use_ephem,
            "use_qpoint": self.use_qpoint,
            "so3g_compat_mode": self.so3g_compat_mode,
        }
        bore_radec = azel_to_radec(site, times, bore_azel, **backend)
        name = block.observation_name
        session = Session(
            name,
            uid=name_UID(name),
            start=datetime.fromtimestamp(times[0], tz=timezone.utc),
            end=datetime.fromtimestamp(times[-1], tz=timezone.utc),
        )
        info = {
            "contract_version": "RL-A",
            "source_file": source.name,
            "source_sha256": source_sha256,
            "samples": len(times),
            "first_time_unix_s": float(times[0]),
            "last_time_unix_s": float(times[-1]),
            "rate_check": rate_check,
            "max_phase_residual_s": block.max_phase_residual_s,
            "time_tolerance_s": block.time_tolerance_s,
            # Retain the scalar summary only when the whole series is constant.
            "roll_deg": (
                float(np.rad2deg(roll_rad[0]))
                if np.all(roll_rad == roll_rad[0])
                else None
            ),
            ROLL_SOURCE_KEY: roll_source,
            ROLL_CONVENTION_KEY: block.roll_convention,
            "caller_telescope": telescope.name,
            "caller_site": site.name,
            "coordinate_backend": backend,
            "flags/intervals": "motion derived (B2.3); solar/directions not yet derived",
            "flag_mapping": {
                SCAN_FLAG_SCIENCE: 0,
                SCAN_FLAG_TURNAROUND: int(self.turnaround_mask),
                SCAN_FLAG_UNCLASSIFIED: int(defaults.shared_mask_invalid),
                SCAN_FLAG_RETUNE: int(defaults.shared_mask_invalid),
            },
        }
        if input_index is not None:
            info["trajectory_input_index"] = input_index
        ob = Observation(
            data.comm,
            telescope,
            len(times),
            name=name,
            uid=name_UID(name),
            session=session,
            process_rows=1,
            sample_sets=[[len(times)]],
        )
        try:
            # Keep all native names as well as any configured time / Az / El aliases.
            geometry = dict(
                zip(
                    ("boresight_azel", "boresight_radec", "position", "velocity"),
                    (bore_azel, bore_radec, position, velocity),
                )
            )
            values = {**block.arrays, **geometry}
            arrays = dict(block.arrays)
            arrays[H5_BORESIGHT_ROLL] = roll_rad
            arrays.update(
                {
                    output: values[key]
                    for key, output in self._trajectory_shared_names().items()
                }
            )
            for key, array in arrays.items():
                ob.shared.create_column(key, shape=array.shape, dtype=array.dtype)
                ob.shared[key].set(array, offset=(0,) * array.ndim, fromrank=0)

            raw = block.arrays["fyst_scan_flag"]
            classes = {
                "science": (raw == SCAN_FLAG_SCIENCE, self.scanning_interval),
                "turnaround": (raw == SCAN_FLAG_TURNAROUND, self.turnaround_interval),
                "invalid": (
                    np.isin(raw, (SCAN_FLAG_UNCLASSIFIED, SCAN_FLAG_RETUNE)),
                    self.fyst_invalid_interval,
                ),
            }
            # Same mapping as the writer; the loader already rejected values outside 0-3.
            flags = to_toast_flags(
                raw,
                turnaround_mask=self.turnaround_mask,
                invalid_mask=defaults.shared_mask_invalid,
            )
            ob.shared.create_column(
                self.shared_flags, shape=flags.shape, dtype=flags.dtype
            )
            ob.shared[self.shared_flags].set(flags, offset=(0,), fromrank=0)
            # Integer spans avoid the TOAST 3.0.5 interval set-operation endpoint bugs.
            for mask, key in classes.values():
                ob.intervals[key] = IntervalList(
                    times, samplespans=_mask_to_sample_spans(mask)
                )
            info["class_counts"] = {
                key: int(np.count_nonzero(mask)) for key, (mask, _) in classes.items()
            }

            if self.det_data is not None:
                ob.detdata.ensure(
                    self.det_data, dtype=np.float64, create_units=self.det_data_units
                )
            if self.det_flags is not None:
                ob.detdata.ensure(self.det_flags, dtype=np.uint8)
            ob[_TRAJECTORY_INFO_KEY] = block.trajectory_info_json
            ob["source_shared_array_sha256"] = json.dumps(
                block.source_shared_array_sha256
            )
            ob["simfyst_info"] = json.dumps(info, sort_keys=True, allow_nan=False)
            # No partially constructed observation is exposed to the caller.
            data.obs.append(ob)
        except Exception:
            ob.clear()
            raise

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        return {
            "shared": list(
                dict.fromkeys(
                    [
                        *_TRAJECTORY_ARRAY_FIELDS,
                        *self._trajectory_shared_names().values(),
                        self.shared_flags,
                    ]
                )
            ),
            "detdata": [
                key for key in (self.det_data, self.det_flags) if key is not None
            ],
            "intervals": [
                self.scanning_interval,
                self.turnaround_interval,
                self.fyst_invalid_interval,
            ],
        }
