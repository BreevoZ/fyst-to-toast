# fyst-to-toast

Feed FYST telescope trajectories into TOAST simulations.
FYST plans the scan and generates per-sample motion; `SimFYST` turns those samples
into TOAST Observations without resampling, so TOAST operators run on FYST's motion.

```text
FYST scheduler ─► schedule.ecsv ────────────────────────────► TOAST SimGround (CES only)
               └► fyst_trajectory.ecsv ─► one H5 per block ─► SimFYST (CES, Daisy, Pong)
```

## Status

Experimental **0.1.0.dev0**, tested with TOAST 3.0.5 and FYST trajectories 0.11.0.

- Serial execution only (no MPI).
- Included focalplanes are demonstrations; real Nasmyth ports and module clocking are not validated.
- Scan-direction, elnod and Sun intervals exist but are empty.
- For one CES, SimGround (from the schedule) and SimFYST (from H5) give closely matching
  hits, I/Q/U maps and polarization angles; see [downstream_maps.py](examples/downstream_maps.py).

## Install

In a Python 3.10+ environment that already has TOAST (setuptools 77+), from a clone
of this repository:

```bash
python -m pip install --no-deps --no-build-isolation -e .
export MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # before importing NumPy/TOAST
```

This is enough to read H5 files and run SimFYST. Generating trajectories also needs
FYST, which is installed from its release tag:

```bash
python -m pip install "fyst-trajectories @ git+https://github.com/ccatobs/fyst-trajectories.git@v0.11.0"
```

## Quick start

From the repository root:

```bash
export RUN_DIR=output/quickstart
python scripts/generate_trajectory.py ces --output-dir "$RUN_DIR"     # or daisy, pong
python scripts/save_trajectory_h5.py "$RUN_DIR/fyst_trajectory.ecsv"  # writes $RUN_DIR/hdf5/
```

Both scripts use Astropy's bundled IERS data and never download; pass `--iers-file`
for another offline table.

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

The focalplane sample rate must match the trajectory (10 Hz by default).
Pass `weather="atacama"` to attach a TOAST `SimWeather` to each block, as SimGround does.

## What each Observation contains

| | Content |
|---|---|
| Shared | `times`, `azimuth`, `elevation`, `boresight_azel`, `boresight_radec`, `position`, `velocity`, `flags`; original FYST arrays (`fyst_*`, scan/subscan indices) and `boresight_roll` |
| Flags | science 0, turnaround 4, unclassified/retune 1; the FYST enum stays in `fyst_scan_flag` |
| Intervals | `scanning`, `turnaround`, `fyst_invalid`; `throw` (half turn + scan + half turn, as in SimGround); SimGround's other interval names, empty |
| Metadata | `scan_el`, `scan_min_el`, `scan_max_el`, `scan_min_az`, `scan_max_az` in SimGround's units; provenance |

## Conventions

- **Roll:** per sample, `boresight_roll = -nasmyth_sign * El`, with the port read from the
  FYST schedule. FYST 0.11.0 schedules store a sky-frame angle in `boresight_angle`;
  `downstream_maps.py` corrects it for SimGround, and FYST main fixes it
  ([PR #2](https://github.com/ccatobs/fyst-trajectories/pull/2)).
- **Clocking:** per module, in the focalplane detector quaternions as `Rz(-c) * Q_fp`, not in roll.
- **Time:** float64 Unix seconds, native samples and block boundaries; UTC strings use milliseconds.

Contracts and limitations: [docs/architecture.md](docs/architecture.md).
Upstream TOAST issues met along the way: [docs/toast_issues.md](docs/toast_issues.md).

## Examples and tools

| Example | Purpose |
|---|---|
| [simfyst_from_h5.py](examples/simfyst_from_h5.py) | Load one H5 block and save the Observation |
| [pipeline_walkthrough.py](examples/pipeline_walkthrough.py) | Step through generation, H5, loading and roll geometry, with a GIF |
| [downstream_maps.py](examples/downstream_maps.py) | TOAST sky, noise and binned maps; `ces --compare-simground` compares with SimGround |

[Tools](tools/README.md) browse ECSV/H5 files; [fixtures](data/README.md) are small sample H5 files.

## Tests

```bash
python -m pytest -p no:cacheprovider
```

Tests that need FYST are skipped when it is not installed.

## License

[BSD-2-Clause](LICENSE). [sim_fyst.py](src/fyst_to_toast/sim_fyst.py) adapts parts of TOAST
SimGround ([BSD-2-Clause](LICENSES/TOAST-BSD-2-Clause.txt), [attribution](LICENSES/TOAST-AUTHORS.txt)).
