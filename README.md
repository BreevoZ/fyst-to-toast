# fyst-to-toast

Feed FYST telescope trajectories into TOAST simulations. FYST plans the scan and
generates per-sample motion; `SimFYST` turns those samples into TOAST Observations
without resampling.

```text
FYST scheduler ─► schedule.ecsv ────────────────────────────► TOAST SimGround (CES only)
               └► fyst_trajectory.ecsv ─► one H5 per block ─► SimFYST (CES, Daisy, Pong)
```

Experimental **0.4.0**, tested with TOAST 3.0.5 and FYST trajectories 0.15.0. Serial only
(no MPI); the included focalplanes are demonstrations.

## Installation

In a Python 3.10+ environment that already has TOAST (setuptools 77+), from a clone:

```bash
python -m pip install --no-deps --no-build-isolation -e .
python -m pip install "fyst-trajectories @ git+https://github.com/ccatobs/fyst-trajectories.git@v0.15.0"
export MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # before importing NumPy/TOAST
```

FYST is needed only to generate trajectories; reading H5 files and running SimFYST do not need it.

## Quick start

```bash
export RUN_DIR=output/quickstart
python scripts/patch_to_ecsv.py ces --output-dir "$RUN_DIR"           # or daisy, pong
python scripts/save_trajectory_h5.py "$RUN_DIR/fyst_trajectory.ecsv"  # writes $RUN_DIR/hdf5/
```

| Script | Use it to |
|---|---|
| `patch_to_ecsv.py` | Let FYST schedule a patch (crossing CES, fixed-window CES, Pong, Daisy), with optional calibrations and Prime-Cam module |
| `window_ces_to_ecsv.py` | Replay one observing-plan row (azimuth window, elevation, start, duration) at its own start time |

## Documentation

- [Usage](docs/usage.md): patch YAML, options, Python API, loading the Observations, conventions, examples
- [Architecture](docs/architecture.md): contracts and limitations
- [TOAST issues](docs/toast_issues.md) met along the way

## Tests

```bash
python -m pytest -p no:cacheprovider
```

Tests that need FYST are skipped when it is not installed.

## License

[BSD-2-Clause](LICENSE). [sim_fyst.py](src/fyst_to_toast/sim_fyst.py) adapts parts of TOAST
SimGround ([BSD-2-Clause](LICENSES/TOAST-BSD-2-Clause.txt), [attribution](LICENSES/TOAST-AUTHORS.txt)).
