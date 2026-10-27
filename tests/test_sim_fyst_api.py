"""The standalone H5 operator API and its serial telescope interface."""
import os

for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from astropy import units as u
from toast import Data
from toast.instrument import GroundSite, Telescope
from toast.instrument_sim import fake_boresight_focalplane
from toast.mpi import Comm

from fyst_to_toast import sim_fyst
from fyst_to_toast.sim_fyst import SimFYST


OWN_TRAITS = set("""
API trajectory_file trajectory_files trajectory_boresight_angle telescope
weather realization max_pwv median_weather
use_ephem use_qpoint so3g_compat_mode times shared_flags det_data det_data_units
det_flags azimuth elevation boresight_azel boresight_radec position velocity
scanning_interval turnaround_interval fyst_invalid_interval turnaround_mask
""".split())
INHERITED_TRAITS = {"name", "enabled", "kernel_implementation", "timing", "timing_total"}
REMOVED_TRAITS = """
telescope_file session_split_key schedule schedule_file
sort_schedule_file randomize_phase track_azimuth scan_rate_az fix_rate_on_sky
scan_rate_el scan_accel_az scan_accel_el scan_cosecant_modulation sun_angle_min
el_mod_step el_mod_rate el_mod_amplitude el_mod_sine el_mod_sine_phase
distribute_time detset_key hwp_angle hwp_rpm hwp_step hwp_step_time elnod_start
elnod_end elnods elnod_every_scan throw_leftright_interval throw_rightleft_interval
throw_interval scan_leftright_interval turn_leftright_interval scan_rightleft_interval
turn_rightleft_interval elnod_interval sun_up_interval sun_close_interval
sun_close_distance sun_up_mask sun_close_mask elnod_mask
""".split()


def test_exact_standalone_traits():
    assert len(OWN_TRAITS) == 27
    assert len(REMOVED_TRAITS) == 43
    assert set(SimFYST.class_trait_names()) == OWN_TRAITS | INHERITED_TRAITS


@pytest.mark.parametrize("name", REMOVED_TRAITS + ["unknown_parameter"])
def test_removed_and_unknown_names_rejected(name):
    # Even None must be rejected, before any trait value validation or file IO.
    with pytest.raises(TypeError, match=name):
        SimFYST(telescope=object(), **{name: None})


def test_unknown_error_lists_all_names():
    with pytest.raises(TypeError) as error:
        SimFYST(unknown_z=1, schedule_file=None, unknown_a=2)
    for name in ("unknown_z", "schedule_file", "unknown_a"):
        assert name in str(error.value)


@pytest.mark.parametrize("options", [{}, {"trajectory_file": None, "trajectory_files": []}])
def test_missing_trajectory_input(options):
    data = Data(Comm(world=None))
    try:
        with pytest.raises(ValueError, match="trajectory_file or trajectory_files"):
            SimFYST(**options).apply(data)
        assert data.obs == []
    finally:
        data.clear()


def test_no_simground_imports():
    forbidden = {"sim_ground", "sim_ground_utils", "sim_hwp", "SimGround"}
    tree = ast.parse(Path(sim_fyst.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                assert forbidden.isdisjoint(name.split(".")), name


@pytest.fixture
def serial_telescope_inputs():
    telescope = Telescope(
        "caller", site=GroundSite("caller-site", -23 * u.deg, -68 * u.deg, 5600 * u.m),
        focalplane=fake_boresight_focalplane(n_pix=1, sample_rate=10 * u.Hz),
    )
    data = SimpleNamespace(comm=SimpleNamespace(
        comm_world=None, comm_group=None, world_size=1, group_size=1, ngroups=1,
    ))
    return SimFYST(telescope=telescope), data, telescope


def test_obs_telescopes_preserves_caller_identity(serial_telescope_inputs):
    op, data, telescope = serial_telescope_inputs
    result = op._obs_telescopes(data, 1, None)
    assert list(result) == ["ALL"]
    assert result["ALL"][0] is telescope
    assert result["ALL"][1] is None


@pytest.mark.parametrize("key,value", [
    ("comm_world", object()), ("comm_group", object()), ("world_size", 2),
    ("group_size", 2), ("ngroups", 2), ("det_ranks", 2), ("detectors", []),
])
def test_obs_telescopes_serial_constraints(serial_telescope_inputs, key, value):
    op, data, _ = serial_telescope_inputs
    det_ranks, detectors = 1, None
    if key == "det_ranks":
        det_ranks = value
    elif key == "detectors":
        detectors = value
    else:
        setattr(data.comm, key, value)
    with pytest.raises(ValueError, match="serial|detectors"):
        op._obs_telescopes(data, det_ranks, detectors)
