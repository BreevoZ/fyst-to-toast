"""Per-block weather and caller-site ownership, using real offline TOAST weather."""

import os

for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

from datetime import datetime, timezone
from importlib import resources
from unittest.mock import Mock

import pytest
from astropy import units as u
from toast.instrument import GroundSite
from toast.weather import SimWeather
import toast.weather as toast_weather

from fyst_to_toast import sim_fyst
from fyst_to_toast.sim_fyst import SimFYST
from test_sim_fyst_trajectory import SAMPLES, caller, data, source


@pytest.fixture(autouse=True)
def isolated_weather_cache(monkeypatch):
    # TOAST truncates its cached distributions in-place when max_pwv is set.
    monkeypatch.setattr(toast_weather, "package_weather_data", None)


def midpoint(times):
    return datetime.fromtimestamp(float((times[0] + times[-1]) / 2), timezone.utc)


def geometry_sites(monkeypatch):
    """Record the site actually used at each geometry boundary, then run TOAST."""
    seen = []
    original_position = GroundSite.position_velocity
    original_radec = sim_fyst.azel_to_radec

    def position(site, times):
        seen.append(("position_velocity", site, site.weather))
        return original_position(site, times)

    def radec(site, *args, **kwargs):
        seen.append(("azel_to_radec", site, site.weather))
        return original_radec(site, *args, **kwargs)

    monkeypatch.setattr(GroundSite, "position_velocity", position)
    monkeypatch.setattr(sim_fyst, "azel_to_radec", radec)
    return seen


@pytest.mark.parametrize("existing_weather", [False, True])
def test_weather_none_preserves_caller(source, data, monkeypatch, existing_weather):
    telescope = caller(source)
    site = telescope.site
    weather = None
    if existing_weather:
        weather = SimWeather(time=midpoint(source.arrays["times"]), name="atacama",
                             site_uid=site.uid, realization=4)
        site.weather = weather
    factory = Mock(side_effect=AssertionError("weather=None must not create weather"))
    monkeypatch.setattr(sim_fyst, "SimWeather", factory)
    seen = geometry_sites(monkeypatch)
    SimFYST(telescope=telescope, trajectory_file=str(source.path), weather=None).apply(data)
    obs = data.obs[0]
    factory.assert_not_called()
    assert obs.telescope is not telescope
    assert obs.telescope.name == telescope.name and obs.telescope.uid == telescope.uid
    assert obs.telescope.focalplane is telescope.focalplane
    assert obs.telescope.site is telescope.site is site
    assert site.weather is weather
    assert [item[0] for item in seen] == ["position_velocity", "azel_to_radec"]
    assert all(item[1] is site and item[2] is weather for item in seen)


@pytest.mark.parametrize("settings", [
    {}, {"realization": 7}, {"max_pwv": 1.41 * u.mm}, {"median_weather": True},
    {"realization": 3, "max_pwv": 2.0 * u.mm, "median_weather": True},
])
def test_builtin_weather_matches_direct_toast(source, data, monkeypatch, settings):
    telescope = caller(source)
    site = telescope.site
    seen = geometry_sites(monkeypatch)
    SimFYST(telescope=telescope, trajectory_file=str(source.path),
            weather="atacama", **settings).apply(data)
    obs = data.obs[0]
    weather = obs.telescope.site.weather
    # A fresh built-in cache gives the direct oracle the same initial distribution.
    monkeypatch.setattr(toast_weather, "package_weather_data", None)
    expected = SimWeather(time=midpoint(source.arrays["times"]), name="atacama",
                          site_uid=site.uid, **settings)
    assert isinstance(weather, SimWeather) and weather == expected
    assert abs((weather._date - expected._date).total_seconds()) <= 1e-6
    assert weather.realization == settings.get("realization", 0)
    assert weather.max_pwv == settings.get("max_pwv")
    assert weather.median_weather == settings.get("median_weather", False)
    assert weather.site_uid == site.uid
    assert weather.pwv == expected.pwv
    if weather.max_pwv is not None:
        assert weather.pwv <= weather.max_pwv
    assert obs.telescope is not telescope
    assert obs.telescope.name == telescope.name and obs.telescope.uid == telescope.uid
    assert obs.telescope.focalplane is telescope.focalplane
    assert obs.telescope.site is not site
    assert telescope.site is site and site.weather is None
    assert [item[0] for item in seen] == ["position_velocity", "azel_to_radec"]
    assert all(item[1] is obs.telescope.site and item[2] is weather for item in seen)


def test_weather_file_fallback(source, data, tmp_path, monkeypatch):
    weather_file = tmp_path / "local-weather.h5"
    weather_file.write_bytes(resources.files("toast._aux.weather").joinpath("atacama.h5").read_bytes())
    telescope = caller(source)
    settings = dict(realization=2, max_pwv=1.41 * u.mm, median_weather=True)
    factory = Mock(wraps=SimWeather)
    monkeypatch.setattr(sim_fyst, "SimWeather", factory)
    SimFYST(telescope=telescope, trajectory_file=str(source.path),
            weather=str(weather_file), **settings).apply(data)
    expected = SimWeather(time=midpoint(source.arrays["times"]), file=str(weather_file),
                          site_uid=telescope.site.uid, **settings)
    assert data.obs[0].telescope.site.weather == expected
    assert data.obs[0].telescope.site.weather.pwv == expected.pwv
    assert len(factory.call_args_list) == 2
    assert factory.call_args_list[0].kwargs["name"] == str(weather_file)
    assert factory.call_args_list[1].kwargs["file"] == str(weather_file)
    assert telescope.site.weather is None


@pytest.mark.parametrize("invalid_weather", ["name", "missing_file", "invalid_file"])
def test_invalid_weather_raises_without_mutation(source, data, tmp_path, invalid_weather):
    weather = tmp_path / "invalid-weather.h5"
    if invalid_weather == "name":
        weather = "not-a-built-in-weather-site"
    elif invalid_weather == "invalid_file":
        weather.write_text("Not an HDF5 weather file")
    telescope = caller(source)
    site = telescope.site
    with pytest.raises(OSError):
        SimFYST(telescope=telescope, trajectory_file=str(source.path),
                weather=str(weather)).apply(data)
    assert not data.obs
    assert telescope.site is site and site.weather is None


def test_weather_is_separate_per_block_and_replaces_only_copied_site(data):
    import h5py
    from types import SimpleNamespace

    paths = [SAMPLES / "fyst_ces_000_001.h5", SAMPLES / "fyst_daisy_003_000.h5"]
    with h5py.File(paths[0], "r") as saved:
        telescope = caller(SimpleNamespace(site=dict(saved["instrument"].attrs)))
    site = telescope.site
    original = SimWeather(time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                          name="atacama", site_uid=site.uid, realization=99)
    site.weather = original
    original_pwv = original.pwv.copy()
    SimFYST(telescope=telescope, trajectory_files=paths, weather="atacama",
            realization=1).apply(data)
    assert len(data.obs) == 2
    assert len({id(obs.telescope.site) for obs in data.obs}) == 2
    assert len({id(obs.telescope.site.weather) for obs in data.obs}) == 2
    for obs in data.obs:
        expected = SimWeather(time=midpoint(obs.shared["times"].data), name="atacama",
                              site_uid=site.uid, realization=1)
        assert obs.telescope.site is not site
        assert obs.telescope.site.weather == expected
        assert obs.telescope.site.weather.pwv == expected.pwv
    assert telescope.site is site and site.weather is original
    assert original.realization == 99 and original.pwv == original_pwv
