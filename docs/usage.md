# Usage

Commands run from the repository root, with `MPI_DISABLE=1 OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1` set before NumPy or TOAST is imported. Both generation
scripts use Astropy's bundled IERS data and never download; `--iers-file` supplies
another offline table, and the report records it as `iers_auto_max_age`.

## Scheduled patches: `patch_to_ecsv.py`

FYST's scheduler decides when to observe. A patch is a YAML file of `ObservingPatch`
arguments plus an optional `module:`. The [templates](../examples/patches/) list every
key FYST accepts for their scan type, with defaults:

| Template | Scan |
|---|---|
| [ces.yaml](../examples/patches/ces.yaml) | Crossing CES: FYST computes the azimuth range from the field |
| [ces_window.yaml](../examples/patches/ces_window.yaml) | CES over a fixed azimuth window (`scan_params: {az_min, az_max}`), observed with module IM1 |
| [pong.yaml](../examples/patches/pong.yaml), [daisy.yaml](../examples/patches/daisy.yaml) | Patterns scanned where the field is on the sky |

```bash
cp examples/patches/pong.yaml my_patch.yaml   # edit the field and scan parameters
python scripts/patch_to_ecsv.py --patch-file my_patch.yaml --start 2026-03-15T01:00:00
```

- Use a case shortcut (`ces`, `daisy`, `pong`) or `--patch-file`; custom files need `--start`.
  The scheduling window is 10800 s unless `--duration-s` is given. Output defaults to
  `output/generated/<case or YAML stem>/`.
- The CLI is science-only by default: calibration durations are 0, but slews and settle
  time remain. `--with-calibrations` uses FYST's default calibrations.
- `module: IM1` (any Prime-Cam bay `IM0`..`IM6`) schedules the patch for that module:
  FYST moves the boresight so the module holds the field, and on a CES `elevation` is the
  module's. The old ring labels `i1`..`i6` are refused.
- Sampling uses `scan_params.timestep` (0.1 s in the templates, FYST's default when unset);
  `--sample-rate HZ` overrides it. The scheduler accepts at most 20 Hz (Go TCS minimum
  interval 0.05 s).
- FYST warns but does not block a pattern that exceeds the mount's warning limits; a scan
  beyond the axis limits FYST checks is not booked. Warnings are listed in `report.json`
  under `fyst_warnings`.

The Python API is `fyst_to_toast.generate.generate_trajectory`. It follows FYST:
`overhead_model=None` means FYST's default calibrations.

```python
from astropy.coordinates import solar_system_ephemeris
from astropy.utils import iers
from fyst_trajectories.overhead import ObservingPatch
from fyst_to_toast.generate import generate_trajectory, science_only_overhead
iers.conf.auto_download = False
patch = ObservingPatch(name="my Pong", ra_center=185, dec_center=-35,
                      width=1.2, height=1.0, scan_type="pong", velocity=0.3)
with solar_system_ephemeris.set("builtin"):
    report = generate_trajectory(patch, "output/my_patch", start="2026-03-15T04:00:00",
                                 duration_s=1800, overhead_model=science_only_overhead())
```

## Replaying a window: `window_ces_to_ecsv.py`

Reproduces one row of an observing plan (azimuth window, elevation, start, duration),
such as a line of the Prime-Cam season schedule. FYST's planner runs directly: no
scheduler, calibrations or wait for the field's pass.

```bash
export RUN_DIR=output/generated/deep56_e50_rising
python scripts/window_ces_to_ecsv.py --window-file examples/windows/deep56_e50_rising.yaml \
    --start 2026-08-13T03:48:45.147 --duration-s 14400
python scripts/save_trajectory_h5.py "$RUN_DIR/fyst_trajectory.ecsv"
```

The [template](../examples/windows/deep56_e50_rising.yaml) has the required `az_window`,
`elevation`, `velocity` and the optional `az_accel`, `timestep`, `module`, `field`
(`field` only feeds FYST's drift warnings). `--max-block-s S` splits the scan into equal
consecutive blocks of at most S seconds, one H5 file and one TOAST Observation each.
The schedule records the scan, so FYST's `schedule_to_trajectories` rebuilds it sample
for sample.

## Loading the observations

`save_trajectory_h5.py` writes one H5 file per block and a `files.json` manifest;
`SimFYST` turns them into TOAST Observations without resampling.

```python
import os
from pathlib import Path
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.table import QTable
from astropy.utils import iers
from toast import Data
from toast.instrument import GroundSite, Telescope
from toast.instrument_sim import fake_boresight_focalplane
from toast.mpi import Comm
from fyst_to_toast.sim_fyst import SimFYST

iers.conf.auto_download = False
run_dir = Path(os.environ["RUN_DIR"])
meta = QTable.read(run_dir / "schedule.ecsv").meta
site = GroundSite(meta["site_name"], meta["site_lat"],
                  meta["site_lon"], meta["site_alt"])
telescope = Telescope(
    "FYST-demo", site=site,
    focalplane=fake_boresight_focalplane(n_pix=1, sample_rate=10 * u.Hz),
)
data = Data(Comm(world=None))
try:
    with solar_system_ephemeris.set("builtin"):
        SimFYST(
            telescope=telescope,
            trajectory_files=[str(run_dir / "hdf5/files.json")],
        ).apply(data)
    print([(obs.name, obs.n_all_samples) for obs in data.obs])
    # Apply downstream TOAST operators here, before clearing the data.
finally:
    data.clear()
```

SimFYST checks `telescope.focalplane.sample_rate` against the stored H5 rate and the
timestamp rate with `sample_rate_rtol=1e-3` (FYST spreads samples evenly over each block,
so the measured rate differs slightly from the nominal one). It uses PyEphem for the
horizontal-to-equatorial transform by default (`use_ephem=True`). `weather="atacama"`
attaches a TOAST `SimWeather` to each block, as SimGround does.

| | Content of each Observation |
|---|---|
| Shared | `times`, `azimuth`, `elevation`, `boresight_azel`, `boresight_radec`, `position`, `velocity`, `flags`; original FYST arrays (`fyst_*`, scan/subscan indices) and `boresight_roll` |
| Flags | science 0, turnaround 4, unclassified/retune 1; the FYST enum stays in `fyst_scan_flag` |
| Intervals | `scanning`, `turnaround`, `fyst_invalid`; `throw` (half turn + scan + half turn, as in SimGround); SimGround's other interval names, empty |
| Metadata | `scan_el`, `scan_min_el`, `scan_max_el`, `scan_min_az`, `scan_max_az` in SimGround's units; provenance |

## Conventions

- **Roll:** per sample, `boresight_roll = -nasmyth_sign * El`, with the port read from the
  FYST schedule. Since FYST 0.12.0, `boresight_angle` stores the TOAST roll, marked by
  `boresight_angle_convention: toast_roll`; `downstream_maps.py` preserves it and
  corrects older (<= 0.11) schedules for SimGround.
- **Clocking:** per module, in the focalplane detector quaternions as `Rz(-c) * Q_fp`, not in roll.
- **Time:** float64 Unix seconds, native samples and block boundaries; UTC strings use milliseconds.

## Examples and tools

| Example | Purpose |
|---|---|
| [simfyst_from_h5.py](../examples/simfyst_from_h5.py) | Load one H5 block and save the Observation |
| [pipeline_walkthrough.py](../examples/pipeline_walkthrough.py) | Step through generation, H5, loading and roll geometry, with a GIF |
| [downstream_maps.py](../examples/downstream_maps.py) | TOAST sky, noise and binned maps; `ces --compare-simground` compares with SimGround |

[Tools](../tools/README.md) browse ECSV/H5 files; [fixtures](../data/README.md) are small sample H5 files.
