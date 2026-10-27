"""Read-only validation: load writer output with native LoadHDF5 and compare every input sample.

This is not the SimFYST reader; it generates no motion, intervals, or detector pointing.
Print JSON to stdout on success; exit with mismatched fields on failure.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import gettempdir


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory", type=Path, help="Original trajectory ECSV used by the writer"
    )
    parser.add_argument(
        "output_dir", type=Path, help="Writer output directory containing files.json"
    )
    args = parser.parse_args()
    source = args.trajectory.expanduser().resolve()
    folder = args.output_dir.expanduser().resolve()
    for path in (source, folder / "files.json"):
        if not path.is_file():
            parser.error(f"Input file not found: {path}")

    # Configure before numerical libraries and TOAST; disable MPI and extra numerical threads.
    for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["TOAST_LOGLEVEL"] = "WARNING"
    for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        cache = Path(gettempdir()) / "fyst-to-toast" / key.lower()
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(cache))

    import h5py
    import numpy as np
    from astropy import units as u
    from astropy.table import QTable
    from toast import Data, ops
    from toast import qarray as qa
    from toast.mpi import Comm
    from toast.observation import default_values as defaults

    from fyst_to_toast.fyst_flags import SCAN_FLAG_SCIENCE, SCAN_FLAG_TURNAROUND
    from fyst_to_toast.trajectory_h5 import _load_trajectory_block

    samples = QTable.read(source)
    blocks = samples.meta["blocks"]
    case = samples.meta["case"]
    require(bool(blocks) and len(samples) > 0, "Input must contain nonempty blocks / samples")
    expected_names = [
        f"fyst_{case}_{b['scan_index']:03d}_{b['subscan_index']:03d}" for b in blocks
    ]
    names = json.loads((folder / "files.json").read_text())
    require(
        names == [f"{name}.h5" for name in expected_names],
        "File manifest does not match native blocks",
    )
    require(len(set(names)) == len(names), "File manifest contains duplicates")
    paths = [folder / name for name in names]
    source_hash = file_sha256(source)
    original_hashes = {path: file_sha256(path) for path in paths}

    # LoadHDF5(detdata=[]) skips detector data; also check that the files contain none.
    for path in paths:
        # A successful round trip must also satisfy the actual SimFYST reader.
        _load_trajectory_block(path)
        with h5py.File(path, "r") as saved:
            require(len(saved["detdata"]) == 0, f"{path.name}: File must not contain detdata")

    data = Data(Comm(world=None))
    try:
        ops.LoadHDF5(
            files=[str(p) for p in paths], detdata=[], force_serial=True
        ).apply(data)
        observations = {obs.name: obs for obs in data.obs}
        require(len(data.obs) == len(blocks), "Observation count does not match native blocks")
        require(
            set(observations) == set(expected_names),
            "Observation names do not match native blocks",
        )
        reports = []
        offset = 0
        for block, name in zip(blocks, expected_names):
            require(block["first"] == offset, f"{name}: Input block slices are not contiguous")
            offset = block["last"]
            part = samples[block["first"] : block["last"]]
            obs = observations[name]
            require(obs.n_all_samples == len(part) > 0, f"{name}: Sample count mismatch")
            require(len(obs.all_detectors) == 5, f"{name}: Expected five illustrative detectors")
            require(len(obs.detdata) == 0, f"{name}: Loaded Observation must not contain detdata")
            info = json.loads(obs["trajectory_info"])
            require(info["block"] == block, f"{name}: Block metadata mismatch")
            require(info["source_sha256"] == source_hash, f"{name}: Input file hash mismatch")
            require(info["case"] == case, f"{name}: Pattern name mismatch")
            require(
                info["start_utc"] == samples.meta["start_utc"],
                f"{name}: start_utc mismatch",
            )
            np.testing.assert_equal(
                obs.telescope.focalplane.sample_rate.to_value(u.Hz),
                1 / block["native_timestep_s"],
            )

            arrays = {key: np.asarray(obs.shared[key].data) for key in obs.shared}
            recorded = json.loads(obs["shared_array_sha256"])
            # Current writers save per-sample boresight_roll (18 fields); older files omit it (17).
            has_roll = "boresight_roll" in arrays
            require(
                len(arrays) == (18 if has_roll else 17) and set(arrays) == set(recorded),
                f"{name}: Shared field mismatch",
            )
            for key, values in arrays.items():
                actual = hashlib.sha256(
                    np.ascontiguousarray(values).tobytes()
                ).hexdigest()
                require(actual == recorded[key], f"{name}/{key}: Hash changed after saving")
                require(np.isfinite(values).all(), f"{name}/{key}: Contains nonfinite values")

            # Compare directly with the original ECSV in addition to the writer's saved hashes.
            expected_arrays = {
                "times": part["timestamp_unix_s"].to_value(u.s),
                "azimuth": part["az_deg"].to_value(u.rad),
                "elevation": part["el_deg"].to_value(u.rad),
                "fyst_time_s": part["time_s"].to_value(u.s),
                "fyst_block_time_s": part["block_time_s"].to_value(u.s),
                "fyst_az_deg": part["az_deg"].to_value(u.deg),
                "fyst_el_deg": part["el_deg"].to_value(u.deg),
                "fyst_az_velocity_deg_s": part["az_velocity"].to_value(u.deg / u.s),
                "fyst_el_velocity_deg_s": part["el_velocity"].to_value(u.deg / u.s),
                "fyst_scan_flag": np.asarray(part["scan_flag"]),
                "scan_index": np.asarray(part["scan_index"]),
                "subscan_index": np.asarray(part["subscan_index"]),
            }
            for key, expected in expected_arrays.items():
                np.testing.assert_array_equal(
                    arrays[key], expected, err_msg=f"{name}/{key}"
                )
            # Derive geometry from input roll, including legacy scalar-only H5.
            if "boresight_roll_deg" in part.colnames:
                expected_roll = part["boresight_roll_deg"].to_value(u.rad)
            else:
                expected_roll = np.full(len(part), np.radians(info["roll_deg"]))
            if has_roll:
                np.testing.assert_array_equal(
                    arrays["boresight_roll"], expected_roll, err_msg=f"{name}/boresight_roll"
                )
            require(np.all(np.diff(arrays["times"]) > 0), f"{name}: Times are not strictly increasing")
            np.testing.assert_array_equal(arrays["scan_index"], block["scan_index"])
            np.testing.assert_array_equal(
                arrays["subscan_index"], block["subscan_index"]
            )

            # Check each FYST enum category separately; TOAST uses bitmasks, not raw enums.
            raw = np.asarray(part["scan_flag"])
            science = raw == SCAN_FLAG_SCIENCE
            turnaround = raw == SCAN_FLAG_TURNAROUND
            for mask, value in (
                (science, 0),
                (turnaround, defaults.shared_mask_unstable_scanrate),
                (~(science | turnaround), defaults.shared_mask_invalid),
            ):
                np.testing.assert_array_equal(arrays["flags"][mask], value)

            for key in ("boresight_azel", "boresight_radec"):
                require(
                    arrays[key].shape == (len(part), 4),
                    f"{name}/{key}: Quaternion shape mismatch",
                )
                np.testing.assert_allclose(
                    np.linalg.norm(arrays[key], axis=1), 1, rtol=0, atol=1e-12
                )
            expected_bore = qa.mult(
                qa.from_lonlat_angles(
                    -part["az_deg"].to_value(u.rad),
                    part["el_deg"].to_value(u.rad),
                    np.zeros(len(part)),
                ),
                qa.rotation(np.array([0.0, 0.0, 1.0]), expected_roll),
            )
            quat_error = np.minimum(
                np.linalg.norm(arrays["boresight_azel"] - expected_bore, axis=1),
                np.linalg.norm(arrays["boresight_azel"] + expected_bore, axis=1),
            )
            require(
                np.max(quat_error) < 1e-12,
                f"{name}: Boresight quaternion orientation does not match input Az/El/roll",
            )
            lon, lat, _ = qa.to_lonlat_angles(arrays["boresight_azel"])
            az_error = (
                np.degrees(-lon) - part["az_deg"].to_value(u.deg) + 180
            ) % 360 - 180
            el_error = np.degrees(lat) - part["el_deg"].to_value(u.deg)
            max_bore_error = float(
                max(np.max(np.abs(az_error)), np.max(np.abs(el_error)))
            )
            require(max_bore_error < 1e-9, f"{name}: Boresight Az/El does not match input")
            reports.append(
                {
                    "name": name,
                    "samples": len(part),
                    "boresight_max_error_deg": max_bore_error,
                    "fyst_flag_values_observed": np.unique(raw).tolist(),
                }
            )

        require(offset == len(samples), "Native blocks do not cover the full input")
        require(
            all(file_sha256(p) == original_hashes[p] for p in paths),
            "H5 files changed during validation",
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "case": case,
                    "observations": reports,
                    "samples": len(samples),
                    "shared_arrays_per_observation": len(arrays),
                    "input_arrays_compared_exactly": list(expected_arrays),
                    "flags_mapping_checked": True,
                    "detectors_per_observation": 5,
                    "detdata": [],
                    "h5_unchanged": True,
                },
                indent=2,
            )
        )
    finally:
        data.clear()


if __name__ == "__main__":
    main()
