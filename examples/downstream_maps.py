"""Native TOAST sky / noise / binned-map pipeline; leaves SimFYST outputs unchanged.

CovarianceAndHits + BinMap bin directly, without templates or solvers.
Tests reuse this example's pipeline functions; installed TOAST operators perform all simulations.
"""

import argparse
import json
import os
from pathlib import Path
from tempfile import gettempdir
import time

for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
for key, suffix in (("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "cache")):
    cache = Path(gettempdir()) / "fyst-to-toast" / suffix
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(key, str(cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.time import Time
from astropy.table import QTable
from astropy.utils import iers
from toast import Data, qarray as qa
from toast.instrument import GroundSite, Telescope
from toast.instrument_sim import fake_hexagon_focalplane
from toast.mpi import Comm
from toast.ops import (
    BinMap, DefaultNoiseModel, PixelsHealpix, PointingDetectorSimple,
    ScanHealpixMap, SimGround, SimNoise, StokesWeights,
)
from toast.ops.mapmaker_utils import CovarianceAndHits
from toast.schedule import GroundSchedule

from fyst_to_toast.sim_fyst import SimFYST
from fyst_to_toast.trajectory_h5 import _load_trajectory_block

iers.conf.auto_download = False
PATCHES = {"ces": (330.0, -50.0), "daisy": (180.0, -30.0), "pong": (180.0, -30.0)}
# Installed TOAST 3.0.5: observation.py:62-68,100-110; ops/mapmaker_binning.py:82-85;
# CovarianceAndHits (ops/mapmaker_utils/mapmaker_utils.py):1039-1041,1060-1062.
SHARED_MASK = 15  # invalid | processing | unstable_scanrate | irregular; includes turnaround=4.
DET_MASK = 7
RCOND = 1e-8


def prepare_simground_schedule(source, folder, mode="corrected"):
    """Correct pre-field_rotation FYST schedules without importing FYST."""
    source, folder = Path(source), Path(folder)
    table = QTable.read(source, format="ascii.ecsv")
    original = table["boresight_angle"].to_value(u.deg).tolist()
    path = source
    action = "original schedule"
    if mode not in ("corrected", "schedule"):
        raise ValueError(f"Unknown SimGround roll mode: {mode}")
    if mode == "corrected" and "field_rotation" not in table.colnames:
        port = table.meta.get("site_nasmyth_port")
        signs = {"right": 1, "left": -1, "cassegrain": 0}
        if port not in signs:
            raise ValueError(f"Missing or unknown site_nasmyth_port: {port!r}")
        table["boresight_angle"] = -signs[port] * table["el"].to(u.deg)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "schedule_corrected.ecsv"
        if path.resolve() == source.resolve():
            raise ValueError("Corrected schedule output must differ from its source")
        table.write(path, format="ascii.ecsv", overwrite=True)
        action = "replaced boresight_angle with -nasmyth_sign * el"
    elif mode == "corrected":
        action = "field_rotation present; already corrected, unchanged"
    return path, {"mode": mode, "action": action, "path": str(path),
                  "original_boresight_angle_deg": original,
                  "boresight_angle_deg": table["boresight_angle"].to_value(u.deg).tolist()}


def polarization_histogram(data, quats):
    """All detector samples, including turnarounds; ISO psi modulo 180 degrees.

    TOAST qarray.to_iso_angles: local e_theta meridian toward detector +x,
    right-handed about line of sight. With no HWP, COSMO StokesWeights uses
    Q ~ cos(2 psi), U ~ -sin(2 psi). This is not IAU position angle.
    """
    edges = np.linspace(0, 360, 73)
    counts = np.zeros(72, dtype=np.int64)
    for obs in data.obs:
        for det in obs.local_detectors:
            psi = qa.to_iso_angles(obs.detdata[quats][det])[2]
            counts += np.histogram(np.rad2deg(2 * psi) % 360, bins=edges)[0]
    return {"edges_deg": edges.tolist(), "counts": counts.tolist(),
            "probability": (counts / counts.sum()).tolist()}


def apply_simground(data, telescope, schedule):
    SimGround(telescope=telescope, schedule=schedule, weather=None,
              scan_rate_az=1 * u.deg / u.s, scan_accel_az=1 * u.deg / u.s**2,
              fix_rate_on_sky=False, randomize_phase=False).apply(data)


def ground_angles(telescope, schedule):
    data = Data(Comm(world=None))
    try:
        apply_simground(data, telescope, schedule)
        PointingDetectorSimple(boresight="boresight_radec", quats="angle_quats",
                               coord_in="C", coord_out="C", hwp_angle=None, shared_flags=None).apply(data)
        return polarization_histogram(data, "angle_quats")
    finally:
        data.clear()


def good_pixels(result):
    return (result["hits"] > 0) & (result["rcond"] >= RCOND)


def compare_maps(fyst, ground, reference, noise, compare_rcond=0.01):
    a, b = fyst["hits"] > 0, ground["hits"] > 0
    union = a | b
    fg, gg = good_pixels(fyst), good_pixels(ground)
    common = fg & gg
    if not common.any():
        raise ValueError("No common well-conditioned pixels for I/Q/U comparison")
    def maximum(values, mask):
        return np.max(np.abs(values[:, mask]), axis=1).tolist()
    def rms(values, mask):
        return np.sqrt(np.mean(values[:, mask] ** 2, axis=1)).tolist()
    def robust(values, mask):
        # Scale absolute residual about the known input, not about its median.
        return (1.4826 * np.median(np.abs(values[:, mask]), axis=1)).tolist()
    def statistics(fmask, gmask, shared):
        return {
            "good_pixels_simfyst_simground_common": [int(fmask.sum()), int(gmask.sum()), int(shared.sum())],
            "noiseless_max_residual_simfyst_K": maximum(fyst["noiseless"]["map"] - reference, fmask),
            "noiseless_max_residual_simground_K": maximum(ground["noiseless"]["map"] - reference, gmask),
            "noiseless_max_difference_common_K": maximum(ground["noiseless"]["map"] - fyst["noiseless"]["map"], shared),
            "noisy_difference_rms_common_K": rms(ground["map"] - fyst["map"], shared) if noise else None,
            "noise_rms_simfyst_common_K": rms(fyst["map"] - reference, shared) if noise else None,
            "noise_rms_simground_common_K": rms(ground["map"] - reference, shared) if noise else None,
            "noise_robust_simfyst_common_K": robust(fyst["map"] - reference, shared) if noise else None,
            "noise_robust_simground_common_K": robust(ground["map"] - reference, shared) if noise else None,
        }
    all_binned = {"label": "all binned pixels", "rcond_threshold": RCOND,
                  "mask_policy": "Legacy: own hit pixels with rcond >= threshold for noiseless residuals; common pixels for differences and noise statistics.",
                  **statistics(fg, gg, common)}
    selected = common & (fyst["rcond"] > compare_rcond) & (ground["rcond"] > compare_rcond)
    if not selected.any():
        raise ValueError(f"No common pixels with rcond > {compare_rcond} for I/Q/U comparison")
    return {
        "hits_jaccard": float(np.count_nonzero(a & b) / union.sum()),
        "hits_pearson_union": float(np.corrcoef(fyst["hits"][union], ground["hits"][union])[0, 1]),
        "center_separation_deg": separation(fyst["center_deg"], ground["center_deg"]),
        "compare_rcond": compare_rcond, "comparison_pixels": int(selected.sum()),
        "mask_policy": "All primary I/Q/U statistics use the same common hit pixels with both rcond strictly above compare_rcond (and passing native binning threshold).",
        "component_order": "I,Q,U; all residuals in K",
        **statistics(selected, selected, selected),
        "all_binned_pixels": all_binned,
        "robust_definition": "1.4826 * median(abs(map - input)) per component; no median subtraction",
        "interpretation": "Noiseless scan/bin closure uses each engine's own pointing, so small residuals alone do not validate roll; use the independent polarization comparison.",
        "noise_policy": "All noise statistics use common pixels; realization=0, component=0, native observation UIDs (not paired noise).",
        "motion": "Native turnaround, initial phase and endpoint sampling differ despite matching mount rate and acceleration.",
    }


def comparison_figures(fyst, ground, angles, folder, patch, mode, noise, compare_rcond=0.01):
    folder.mkdir(parents=True, exist_ok=True)
    selected = np.flatnonzero((fyst["hits"] > 0) | (ground["hits"] > 0))
    projection = hp.projector.GnomonicProj(rot=patch)
    x, y = projection.vec2xy(*hp.pix2vec(hp.get_nside(fyst["hits"]), selected, nest=True))
    reso = float(np.rad2deg(max(np.max(np.abs(x)) / 300, np.max(np.abs(y)) / 225)) * 60 * 1.05)
    def panel(values, mask, slot, title, unit, limits):
        values = values.astype(float).copy()
        values[~mask] = hp.UNSEEN
        hp.gnomview(values, fig=plt.gcf().number, sub=slot, nest=True, rot=patch,
                    xsize=600, ysize=450, reso=reso, title=title, unit=unit,
                    min=limits[0], max=limits[1], notext=True)
    fig = plt.figure(figsize=(14, 5))
    limit = max(fyst["hits"].max(), ground["hits"].max())
    for i, (label, result) in enumerate((("SimFYST", fyst), (f"SimGround ({mode})", ground)), 1):
        panel(result["hits"], result["hits"] > 0, (1, 2, i), label + " hits", "hits", (0, limit))
    fig.savefig(folder / "hits.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    fig = plt.figure(figsize=(16, 13))
    fg = gg = good_pixels(fyst) & good_pixels(ground) & (fyst["rcond"] > compare_rcond) & (ground["rcond"] > compare_rcond)
    for i, component in enumerate("IQU"):
        lo, hi = np.percentile(np.concatenate((fyst["map"][i, fg], ground["map"][i, gg])) * 1e6, [1, 99])
        for j, (label, result, mask) in enumerate((("SimFYST", fyst, fg), (f"SimGround ({mode})", ground, gg))):
            panel(result["map"][i] * 1e6, mask, (3, 3, i * 3 + j + 1), f"{label}: {component}", "uK", (lo, hi))
        diff = (ground["map"][i] - fyst["map"][i]) * 1e6
        lim = max(float(np.percentile(np.abs(diff[fg & gg]), 99)), 1e-12)
        panel(diff, fg & gg, (3, 3, i * 3 + 3), f"Ground - FYST: {component}", "uK", (-lim, lim))
    fig.suptitle(("Noisy" if noise else "Noiseless") + f" I/Q/U; common pixels: both rcond > {compare_rcond:g}; display clipped at 1/99 percentiles (difference: 99% absolute)", y=1.02)
    fig.savefig(folder / "iqu.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, hist in angles.items():
        ax.stairs(hist["probability"], hist["edges_deg"], label=label)
    ax.set(xlabel="2 psi (degrees; TOAST ISO convention)", ylabel="Probability per 5-degree bin",
           title="All detector samples, including turnarounds")
    ax.legend(); fig.tight_layout()
    fig.savefig(folder / "psi.png", dpi=140); plt.close(fig)


def caller_telescope(source):
    site = _load_trajectory_block(source).source_site
    focalplane = fake_hexagon_focalplane(
        n_pix=7, width=0.4 * u.deg, sample_rate=10 * u.Hz,
        psd_net=300e-6 * u.K * u.s**0.5, psd_fknee=0.05 * u.Hz,
        psd_fmin=0 * u.Hz, psd_alpha=1.0, random_seed=123456,
    )
    return Telescope("DS1", site=GroundSite("source-site", site["lat"], site["lon"], site["height"]),
                     focalplane=focalplane)


def synthetic_sky(nside, path, mode="IQU"):
    """Pixel-constant C-coordinate sky in K, stored as float32 like ScanHealpixMap."""
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=True)
    sky = np.array([
        1e-3 * (2 + 0.3 * np.cos(theta) + 0.2 * np.sin(phi)),
        1e-4 * np.sin(theta) * np.cos(2 * phi),
        5e-5 * np.sin(theta) * np.sin(2 * phi),
    ], dtype=np.float32)
    if mode == "I":
        sky = sky[:1]
    hp.write_map(str(path), sky, nest=True, coord="C", dtype=np.float32,
                 column_units="K", overwrite=True)
    return sky


def full_map(pixel_data):
    """Serial export of native PixelData; no mapmaking or resampling here."""
    dist = pixel_data.distribution
    result = np.zeros((dist.n_pix, pixel_data.n_value), dtype=pixel_data.data.dtype)
    for local, global_submap in enumerate(dist.local_submaps):
        first = int(global_submap) * dist.n_pix_submap
        last = min(first + dist.n_pix_submap, dist.n_pix)
        result[first:last] = pixel_data.data[local, :last-first]
    return result.T


def center(hits, nside):
    selected = np.flatnonzero(hits)
    vectors = np.array(hp.pix2vec(nside, selected, nest=True)).T
    mean = np.sum(vectors * hits[selected, None], axis=0)
    if not np.any(mean):
        raise ValueError("No hits: cannot compute pointing center")
    lon, lat = hp.vec2ang(mean, lonlat=True)
    return [float(lon[0]), float(lat[0])]


def separation(a, b):
    av, bv = hp.ang2vec(*a, lonlat=True), hp.ang2vec(*b, lonlat=True)
    return float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(av, bv)), np.dot(av, bv))))


def run_chain(data, sky_file, nside, *, mode="IQU", noise=True, prefix="iqu"):
    """Native operators only.  Leave outputs available for assertions / export."""
    DefaultNoiseModel().apply(data)
    pointing = PointingDetectorSimple(boresight="boresight_radec", quats=f"{prefix}_quats",
                                     coord_in="C", coord_out="C", hwp_angle=None,
                                     shared_flag_mask=SHARED_MASK)
    pixels = PixelsHealpix(detector_pointing=pointing, nside=nside, nside_submap=1,
                          nest=True, pixels=f"{prefix}_pixels")
    weights = StokesWeights(detector_pointing=pointing, mode=mode, weights=f"{prefix}_weights",
                            hwp_angle=None, IAU=False)
    dist = f"{prefix}_dist"
    pointing.apply(data)
    pixels.apply(data)
    weights.apply(data)
    ScanHealpixMap(file=str(sky_file), det_data="signal", zero=True,
                   pixel_pointing=pixels, stokes_weights=weights,
                   pixel_dist=dist, save_pointing=True).apply(data)
    if noise:
        SimNoise(det_data="signal", realization=0, component=0, serial=True).apply(data)
    common = dict(pixel_pointing=pixels, stokes_weights=weights, pixel_dist=dist,
                  covariance=f"{prefix}_cov", shared_flag_mask=SHARED_MASK,
                  det_mask=DET_MASK, det_flag_mask=DET_MASK)
    CovarianceAndHits(**common, hits=f"{prefix}_hits", rcond=f"{prefix}_rcond",
                      rcond_threshold=RCOND, save_pointing=True).apply(data)
    BinMap(**common, det_data="signal", binned=f"{prefix}_binned", full_pointing=True).apply(data)
    hits = full_map(data[f"{prefix}_hits"])[0]
    binned = full_map(data[f"{prefix}_binned"])
    rcond = full_map(data[f"{prefix}_rcond"])[0]
    assert np.isfinite(hits).all() and np.isfinite(binned).all() and np.isfinite(rcond).all()

    # Independent hit oracle: all detectors selected, all detector flags zero,
    # full observation view, valid pixels for all unmasked samples. No rcond cut
    # is applied to hits (native CovarianceAndHits:1242-1272).
    expected = np.zeros_like(hits)
    per_observation = []
    for obs in data.obs:
        assert not any(obs.local_detector_flags.values())
        assert not np.any(obs.detdata["flags"].data)
        good = (obs.shared["flags"].data & SHARED_MASK) == 0
        hist = np.zeros_like(hits)
        for det in obs.local_detectors:
            pix = obs.detdata[pixels.pixels][det][good]
            assert np.all((pix >= 0) & (pix < len(hits)))
            hist += np.bincount(pix, minlength=len(hits))
        expected += hist
        assert int(hist.sum()) == int(good.sum()) * len(obs.local_detectors)
        per_observation.append({"name": obs.name, "samples": int(len(good)),
                                "masked_samples": int((~good).sum()), "hits": int(hist.sum()),
                                "center_deg": center(hist, nside)})
    np.testing.assert_array_equal(hits, expected)
    return {"hits": hits, "map": binned, "rcond": rcond, "center_deg": center(hits, nside),
            "per_observation": per_observation, "native_prefix": prefix}


def structure(data):
    return [{"name": obs.name, "samples": int(obs.n_local_samples),
             "start_utc": Time(obs.shared["times"].data[0], format="unix", precision=3).isot + "+00:00",
             "stop_utc": Time(obs.shared["times"].data[-1], format="unix", precision=3).isot + "+00:00",
             "shared": sorted(obs.shared), "intervals": sorted(obs.intervals)} for obs in data.obs]


def save_products(result, folder, patch):
    folder.mkdir(parents=True, exist_ok=True)
    valid = (result["hits"] > 0) & (result["rcond"] >= RCOND)
    fields = {"hits": result["hits"], **dict(zip("IQU", result["map"], strict=True)),
              "rcond": result["rcond"]}
    for label, values in fields.items():
        hp.write_map(str(folder / f"{label}.fits"), values, nest=True, coord="C",
                     dtype=values.dtype, overwrite=True, column_units="K" if label in "IQU" else "")
        if label == "rcond":
            continue
        image_map = values.astype(float).copy()
        image_map[~(result["hits"] > 0 if label == "hits" else valid)] = hp.UNSEEN
        hp.gnomview(image_map, nest=True, rot=patch, xsize=600, ysize=360,
                    reso=4 if patch == PATCHES["ces"] else 0.6,
                    title=f"{folder.name}: {label}", unit="K" if label in "IQU" else "hits",
                    notext=True)
        fig = plt.gcf()
        fig.set_size_inches(8, 5)
        hp.graticule()
        fig.savefig(folder / f"{label}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


def summarize(result, patch):
    rows = result["per_observation"]
    samples = sum(row["samples"] for row in rows)
    report = {"observations": len(rows), "samples": samples, "detectors": 14,
            "masked_samples": sum(row["masked_samples"] for row in rows),
            "masked_fraction": sum(row["masked_samples"] for row in rows) / samples,
            "total_hits": int(result["hits"].sum()), "hit_pixels": int(np.count_nonzero(result["hits"])),
            "well_conditioned_pixels": int(np.count_nonzero((result["hits"] > 0) & (result["rcond"] >= RCOND))),
            "center_deg": result["center_deg"],
            "per_observation": rows}
    if patch == PATCHES["ces"]:
        pixel = int(hp.ang2pix(hp.get_nside(result["hits"]), *patch, lonlat=True, nest=True))
        report["patch_center_pixel"] = pixel
        report["patch_center_pixel_hits"] = int(result["hits"][pixel])
        report["patch_center_in_footprint"] = bool(result["hits"][pixel] > 0)
    else:
        report["patch_center_offset_deg"] = separation(result["center_deg"], patch)
    return report


def simulate(telescope, manifest, sky_iqu, sky_i, nside, folder, patch, noise, schedule=None):
    data = Data(Comm(world=None))
    try:
        if schedule is None:
            SimFYST(telescope=telescope, trajectory_files=[str(manifest)]).apply(data)
        else:
            apply_simground(data, telescope, schedule)
        layout = structure(data)
        result = run_chain(data, sky_iqu, nside, noise=noise)
        PointingDetectorSimple(boresight="boresight_radec", quats="angle_quats",
                               coord_in="C", coord_out="C", hwp_angle=None, shared_flags=None).apply(data)
        result["angles"] = polarization_histogram(data, "angle_quats")
        result["noiseless"] = run_chain(data, sky_iqu, nside, noise=False, prefix="noiseless")
        save_products(result, folder, patch)
        # Separate I-only, noise-free control on the same Observations; sky scanner
        # zeros signal. This is not an I projection of polarized/noisy TOD.
        control = run_chain(data, sky_i, nside, mode="I", noise=False, prefix="control")
        reference_i = hp.read_map(str(sky_i), nest=True, dtype=np.float32)
        residual = float(np.max(np.abs(control["map"][0, control["hits"] > 0] - reference_i[control["hits"] > 0])))
        np.testing.assert_allclose(control["map"][0, control["hits"] > 0],
                                   reference_i[control["hits"] > 0], rtol=5e-7, atol=1e-10)
        report = summarize(result, patch)
        report.update(structure=layout, signal_only_I_max_residual_K=residual)
        return report, result
    finally:
        data.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", choices=PATCHES, default="ces")
    parser.add_argument("--hdf5-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--nside", type=int, default=256)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--compare-simground", action="store_true")
    parser.add_argument("--simground-roll", choices=("corrected", "schedule"), default="corrected",
                        help="Correct legacy FYST schedule roll (default), or use schedule unchanged")
    parser.add_argument("--compare-rcond", type=float, default=0.01,
                        help="Require both rcond > threshold for comparison I/Q/U statistics and plots (default: 0.01)")
    args = parser.parse_args()
    if not np.isfinite(args.compare_rcond) or not 0 <= args.compare_rcond < 1:
        parser.error("--compare-rcond must be finite and in [0, 1)")
    if not hp.isnsideok(args.nside, nest=True):
        parser.error("--nside must be a power of two supported by HEALPix NESTED")
    if args.compare_simground and args.case != "ces":
        parser.error("--compare-simground supports only ces")
    folder = (args.output_dir or ROOT / "output/downstream" / args.case).resolve()
    if not folder.is_relative_to(ROOT / "output") or folder == ROOT / "output":
        parser.error("--output-dir must be a subdirectory of this repository's output/")
    if folder.exists() and not args.overwrite:
        parser.error(f"Output directory exists: {folder}; specify --overwrite to rerun")
    hdf5 = (args.hdf5_dir or ROOT / "output/generated" / args.case / "hdf5").resolve()
    manifest = hdf5 / "files.json"
    names = json.loads(manifest.read_text())
    telescope = caller_telescope(hdf5 / names[0])
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sky_iqu, sky_i = folder / "input_IQU.fits", folder / "input_I.fits"
    reference = synthetic_sky(args.nside, sky_iqu)
    synthetic_sky(args.nside, sky_i, "I")
    print("Pipeline: SimFYST → DefaultNoiseModel → PointingDetectorSimple(boresight_radec) → PixelsHealpix")
    print("      → StokesWeights(IQU) → ScanHealpixMap → SimNoise(optional) → CovarianceAndHits → BinMap")
    print(f"Serial, 14 detectors / 10 Hz; nside={args.nside}, NESTED / C; shared mask={SHARED_MASK} (includes turnaround=4); rcond={RCOND}")
    print("CovarianceAndHits + BinMap bin directly without templates; plots mask unhit / ill-conditioned pixels, while FITS retain native zeros and include rcond.")
    report = {"case": args.case, "nside": args.nside, "noise": not args.no_noise,
              "shared_flag_mask": SHARED_MASK, "det_flag_mask": DET_MASK, "rcond_threshold": RCOND}
    with solar_system_ephemeris.set("builtin"):
        report["simfyst"], fyst = simulate(telescope, manifest, sky_iqu, sky_i, args.nside,
                                          folder / "simfyst", PATCHES[args.case], not args.no_noise)
        if args.compare_simground:
            schedules, roll_info = {}, {}
            for mode in ("corrected", "schedule"):
                path, roll_info[mode] = prepare_simground_schedule(hdf5.parent / "schedule.ecsv", folder, mode)
                schedules[mode] = GroundSchedule()
                schedules[mode].read(str(path))
            report["simground_roll"] = roll_info[args.simground_roll]
            report["simground"], ground = simulate(telescope, manifest, sky_iqu, sky_i, args.nside,
                                                   folder / "simground", PATCHES[args.case], not args.no_noise,
                                                   schedule=schedules[args.simground_roll])
            other = "schedule" if args.simground_roll == "corrected" else "corrected"
            histograms = {args.simground_roll: ground["angles"],
                          other: ground_angles(telescope, schedules[other])}
            angles = {"SimFYST": fyst["angles"], "SimGround (corrected)": histograms["corrected"],
                      "SimGround (schedule)": histograms["schedule"]}
            report["comparison"] = compare_maps(fyst, ground, reference, not args.no_noise, args.compare_rcond)
            report["comparison"]["polarization"] = {
                "convention": "TOAST qarray.to_iso_angles; ISO local e_theta to detector +x; COSMO Q~cos(2psi), U~-sin(2psi), no HWP. All samples, including turnarounds.",
                "histogram_overlap_with_simfyst": {mode: float(np.minimum(fyst["angles"]["probability"], histograms[mode]["probability"]).sum()) for mode in histograms},
                "schedule_minus_corrected_roll_deg": (np.array(roll_info["schedule"]["boresight_angle_deg"]) - np.array(roll_info["corrected"]["boresight_angle_deg"])).tolist(),
                "rolls": roll_info, "histograms": angles,
            }
            comparison_figures(fyst, ground, angles, folder / "figures", PATCHES[args.case],
                               args.simground_roll, not args.no_noise, args.compare_rcond)
    report["elapsed_s"] = round(time.perf_counter() - started, 3)
    report["evidence"] = "Verified by execution"
    (folder / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for engine in ("simfyst", "simground"):
        if engine in report:
            item = report[engine]
            print(f"{engine}: {item['observations']} Observations, {item['samples']} samples, mask={item['masked_fraction']:.3%}, hits={item['total_hits']}, pixels={item['hit_pixels']}")
            print(f"  hits-weighted center={item['center_deg']}; signal-only I max residual={item['signal_only_I_max_residual_K']:.3e} K")
            if args.case == "ces":
                print(f"  pixel at patch center 330/−50={item['patch_center_pixel']}; within combined hits footprint={item['patch_center_in_footprint']}; hits in that pixel={item['patch_center_pixel_hits']}(informational, not a failure condition)")
            else:
                print(f"  distance from patch center={item['patch_center_offset_deg']:.6f} deg")
            for row in item["structure"]:
                print(f"  {row['name']} / {row['samples']} samples; shared={row['shared']}; intervals={row['intervals']}")
    if "comparison" in report:
        summary = dict(report["comparison"])
        summary["polarization"] = {key: value for key, value in summary["polarization"].items() if key != "histograms"}
        print("SimGround comparison: " + json.dumps(summary, ensure_ascii=False))
    print(f"Elapsed {report['elapsed_s']:.3f} s; files: {folder}/input_*.fits, simfyst/[hits,I,Q,U,rcond].fits, PNG, report.json (comparison also in simground/).")


if __name__ == "__main__":
    main()
