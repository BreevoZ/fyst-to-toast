# Known TOAST issues

Upstream TOAST 3.0.5 behaviour found while building and validating SimFYST.
Each issue was reproduced with the installed TOAST 3.0.5; line numbers refer to that release.
SimGround is affected in the same way unless noted.

| # | Issue | Affects SimGround | SimFYST |
|---|---|---|---|
| 1 | `max_pwv` modifies TOAST's shared weather tables | yes | same behaviour; avoid mixing caps in one process |
| 2 | `use_qpoint` validator is never registered | yes | fixed |
| 3 | PolyFilter fails on an empty interval view | n/a | `throw` is never empty for valid motion |
| 4 | `azel_to_radec(use_ephem=False)` has large pointing errors twice per sidereal day | only with `use_ephem=False` | default `use_ephem=True` is not affected |

## 1. `max_pwv` modifies the shared weather tables

`SimWeather(name=...)` takes its distributions from a process-wide cache
(`toast/weather.py:372-373`). With `max_pwv` set, `_truncate_distributions`
overwrites those cached arrays in place (`weather.py:456`, `cdf[:] = new_cdf`).
Every later `SimWeather` with the same name in that process, with or without a cap,
then draws from the truncated distribution. Weather objects created earlier change too
once they are redrawn. A cap below what a month supports raises `RuntimeError`
after earlier months have already been truncated.

```python
from datetime import datetime, timezone
import astropy.units as u
from toast.weather import SimWeather

t = datetime(2026, 8, 2, 2, tzinfo=timezone.utc)
a = SimWeather(time=t, name="atacama", site_uid=1, realization=0)
print(a.pwv)   # 1.151 mm
SimWeather(time=t, name="atacama", site_uid=1, realization=0, max_pwv=1.41 * u.mm)
a.set(time=t, realization=0, site_uid=1)
print(a.pwv)   # 0.856 mm: the cap leaked into `a`
print(SimWeather(time=t, name="atacama", site_uid=1, realization=0).pwv)   # 0.856 mm
```

**Workaround:** use one `max_pwv` setting per Python process, or a new process for each setting.

## 2. `use_qpoint` validator is never registered

In `toast/ops/sim_ground.py`, the validators for `use_qpoint` (line 378) and
`so3g_compat_mode` (line 394) are both named `_check_use_qpoint`. The second
definition replaces the first, so `use_qpoint` is never validated:
`SimGround(use_qpoint=True)` keeps the default `use_ephem=True` without error.
`azel_to_radec` then prefers qpoint (`toast/coordinates.py:262`), and a missing
qpoint install is only reported when the operator runs.

```python
import toast
op = toast.ops.SimGround(use_qpoint=True)
print(op.use_ephem, op.use_qpoint)   # True True, expected TraitError
```

**Workaround:** set `use_ephem=False` explicitly when enabling qpoint.
SimFYST gives the two validators distinct names.

## 3. PolyFilter fails on an empty interval view

PolyFilter builds its span arrays with `np.array(local_starts)`
(`toast/ops/polyfilter/polyfilter.py:542-543`). For a view with no intervals the
array is empty float64, while the compiled kernel requires int64, so `apply` raises
`RuntimeError: Python buffer is type 'd', which is not in compatible list {'q','l',}`.
A missing view raises a clear error instead (`polyfilter.py:526-532`).

**Workaround:** use a non-empty view, or `view=None` for whole observations.
SimFYST derives `throw` from FYST turnarounds, so the default view works whenever an
observation contains science or turnaround samples. The direction, elnod and Sun
intervals are empty, so PolyFilter cannot use them as views yet.

## 4. `azel_to_radec(use_ephem=False)` has large pointing errors twice per sidereal day

With `use_ephem=False`, `azel_to_radec` builds the horizontal-to-ICRS rotation at
sparse times from three transformed axes (`toast/coordinates.py:164-231`). After the
Astropy transform the axes are not mutually perpendicular (13-24 arcsec at the FYST
site, because aberration depends on direction), and the orthonormalisation step is
commented out (`coordinates.py:214-220`). The closed-form quaternion solution that
follows divides by products of quaternion components (`coordinates.py:223-229`), so
the inconsistency is amplified whenever a component passes through zero. That happens
twice per sidereal day, near local sidereal time 0 h and 12 h. The PyEphem path uses
the same closed form but orthonormalises the axes first (`coordinates.py:80-85`) and
does not show the error. The 120 s sparse grid and SLERP are not the cause: the error
is already present at the sparse nodes. The same code is on the upstream `master`
branch as of 2026-11-16.

On a 4 h constant-elevation scan of Deep56 the boresight error against a direct
per-sample Astropy transform peaks at 842 arcsec and puts a bright band into the hit
map; with `use_ephem=True` it stays below 29 arcsec. The peak depends on how close a
sparse node falls to the zero crossing: shifting the start time by up to 120 s moves
it between 559 and 26407 arcsec. Near LST 12 h the error stays above 60 arcsec for
about 4 h (fixed Az 63 deg, El 50 deg, same date).

```python
import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, ICRS
from astropy.time import Time
from toast import qarray as qa
from toast.coordinates import azel_to_radec
from toast.instrument import GroundSite

site = GroundSite("FYST", -22.985639 * u.deg, -67.740278 * u.deg, 5611.8 * u.m)
t = Time("2026-08-13T03:48:45.147").unix + np.arange(14401.0)   # 4 h at 1 Hz
az, el = np.full(t.size, 63.0), np.full(t.size, 50.0)            # fixed pointing
azel = qa.from_lonlat_angles(np.radians(-az), np.radians(el), np.zeros(t.size))
ref = AltAz(az=az * u.deg, alt=el * u.deg, obstime=Time(t, format="unix"),
            location=site.earthloc).transform_to(ICRS()).cartesian.xyz.value.T
for use_ephem in (False, True):
    v = qa.rotate(azel_to_radec(site, t, azel, use_ephem=use_ephem), [0.0, 0.0, 1.0])
    err = np.degrees(np.arccos(np.clip(np.sum(v * ref, axis=1), -1, 1))) * 3600
    print(use_ephem, f"max {err.max():.0f} arcsec at {(t[err.argmax()] - t[0]) / 60:.0f} min")
# False max 1890 arcsec at 196 min
# True max 24 arcsec at 240 min
```

**Workaround:** keep `use_ephem=True`, the default of SimGround and SimFYST;
fyst-to-toast also uses it when writing H5 files. `azel_to_radec` itself defaults to
`use_ephem=False`, so pass the flag when calling it directly. With `use_qpoint=True`
(issue 2) the qpoint path is used and this issue does not apply.
