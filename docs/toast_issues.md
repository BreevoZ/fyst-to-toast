# Known TOAST issues

Upstream TOAST 3.0.5 behaviour found while building and validating SimFYST.
Each issue was reproduced with the installed TOAST 3.0.5; line numbers refer to that release.
SimGround is affected in the same way unless noted.

| # | Issue | Affects SimGround | SimFYST |
|---|---|---|---|
| 1 | `max_pwv` modifies TOAST's shared weather tables | yes | same behaviour; avoid mixing caps in one process |
| 2 | `use_qpoint` validator is never registered | yes | fixed |
| 3 | PolyFilter fails on an empty interval view | n/a | `throw` is never empty for valid motion |

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
