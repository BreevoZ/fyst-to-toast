# Trajectory fixtures

Two small native TOAST Observation H5 files support loader tests and examples
without FYST generation. Each contains one native scan/subscan block.

| File | Scan / subscan | Size | Samples | Raw FYST flag counts |
|---|---|---:|---:|---|
| [fyst_ces_000_001.h5](fyst_ces_000_001.h5) | 0 / 1 (CES) | 533,001 bytes (520.509 KiB) | 2611 | `{1: 2531, 2: 80}` |
| [fyst_daisy_003_000.h5](fyst_daisy_003_000.h5) | 3 / 0 (Daisy) | 103,075 bytes (100.659 KiB) | 399 | `{1: 394, 2: 5}` |

Total size: 636,076 bytes. SHA-256:

```text
fyst_ces_000_001.h5
  2a46b18a6a9e3a6aa9f5d197b42c0ae29d0958a073dece5ad588ef8430035635
fyst_daisy_003_000.h5
  409931e4caeeda4fcf3307ddb3207f6bbc3a234900a0e657b5ca8260f3d42425
```

The inputs were generated with FYST trajectories 0.8.0 plus a historical UTC
patch, then saved with TOAST 3.0.5 (HDF5 format version 2). They use the older
writer format without `boresight_roll`; the loader broadcasts
`trajectory_info.roll_deg` (zero here) in radians. They were not regenerated
with the currently supported FYST version.

CES has constant 50-degree elevation, five science runs with Az velocity signs
`- + - + -`, and four complete 20-sample turnarounds. Daisy begins within a
turnaround: samples `[0, 5)` are turnaround and `[5, 399)` are science.
Daisy elevation runs from 79.04457108202241 to 80.51415846553718 degrees.

| Measurement | CES | Daisy |
|---|---|---|
| First sample, Unix s | `1785635187.8999493` | `1773548960.101578` |
| Last sample, Unix s | `1785635448.9000187` | `1773548999.901578` |
| First sample, UTC | `2026-08-02T01:46:27.899949+00:00` | `2026-03-15T04:29:20.101578+00:00` |
| Last sample, UTC | `2026-08-02T01:50:48.900019+00:00` | `2026-03-15T04:29:59.901578+00:00` |
| Last minus first, s | `261.0000693798065` | `39.799999952316284` |
| Actual rate, `(N-1)/span`, Hz | `9.999997341770571` | `10.000000011980832` |

Both clock phase residuals are `2.384185791015625e-7 s`, below the loader tolerance
`4.76837158203125e-7 s`; nominal 10 Hz passes the `1e-6` relative rate tolerance.
The UTC values above retain the historical microsecond display.

These fixtures contain neither raw flags 0/3 nor detector signals or scan intervals.
Their five illustrative detectors and zero roll are not a real focalplane model.
They do not cover full schedules, Pong, irregular sampling, MPI, or calibration.
