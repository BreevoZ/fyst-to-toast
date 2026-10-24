"""DS1：native TOAST sky / noise / binned-map 链路；不修改 SimFYST 输出。

CovarianceAndHits + BinMap 直接做分 bin，无 template / solver。
测试复用本 example 的链路函数；所有数值模拟都由安装版 TOAST operators 完成。
"""

import argparse
import json
import os
from pathlib import Path
import time

for key in ("MPI_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
for key, suffix in (("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "cache")):
    cache = ROOT / "output/codex/DS1" / suffix
    cache.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(cache)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.time import Time
from astropy.utils import iers
from toast import Data
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
            SimGround(telescope=telescope, schedule=schedule, weather=None,
                      scan_rate_az=1 * u.deg / u.s, scan_accel_az=1 * u.deg / u.s**2,
                      fix_rate_on_sky=False, randomize_phase=False).apply(data)
        layout = structure(data)
        result = run_chain(data, sky_iqu, nside, noise=noise)
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
    args = parser.parse_args()
    if not hp.isnsideok(args.nside, nest=True):
        parser.error("--nside 必须为 HEALPix NESTED 支持的 2 的幂")
    if args.compare_simground and args.case != "ces":
        parser.error("--compare-simground 仅支持 ces")
    folder = (args.output_dir or ROOT / "output/downstream" / args.case).resolve()
    if not folder.is_relative_to(ROOT / "output") or folder == ROOT / "output":
        parser.error("--output-dir 必须在本 repo 的 output/ 子目录下")
    if folder.exists() and not args.overwrite:
        parser.error(f"输出目录已存在：{folder}；重跑请传 --overwrite")
    hdf5 = (args.hdf5_dir or ROOT / "output/generated" / args.case / "hdf5").resolve()
    manifest = hdf5 / "files.json"
    names = json.loads(manifest.read_text())
    telescope = caller_telescope(hdf5 / names[0])
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sky_iqu, sky_i = folder / "input_IQU.fits", folder / "input_I.fits"
    synthetic_sky(args.nside, sky_iqu)
    synthetic_sky(args.nside, sky_i, "I")
    print("链路：SimFYST → DefaultNoiseModel → PointingDetectorSimple(boresight_radec) → PixelsHealpix")
    print("      → StokesWeights(IQU) → ScanHealpixMap → SimNoise(可关) → CovarianceAndHits → BinMap")
    print(f"串行，14 detectors / 10 Hz；nside={args.nside}，NESTED / C；shared mask={SHARED_MASK}（含 turnaround=4）；rcond={RCOND}")
    print("CovarianceAndHits + BinMap 直接分 bin，无 template；图中的无 hits / 病态像素留白，FITS 保留原生零值并附 rcond。")
    report = {"case": args.case, "nside": args.nside, "noise": not args.no_noise,
              "shared_flag_mask": SHARED_MASK, "det_flag_mask": DET_MASK, "rcond_threshold": RCOND}
    with solar_system_ephemeris.set("builtin"):
        report["simfyst"], fyst = simulate(telescope, manifest, sky_iqu, sky_i, args.nside,
                                          folder / "simfyst", PATCHES[args.case], not args.no_noise)
        if args.compare_simground:
            schedule = GroundSchedule()
            schedule.read(str(hdf5.parent / "schedule.ecsv"))
            report["simground"], ground = simulate(telescope, manifest, sky_iqu, sky_i, args.nside,
                                                   folder / "simground", PATCHES[args.case], not args.no_noise,
                                                   schedule=schedule)
            a, b = fyst["hits"] > 0, ground["hits"] > 0
            f_intervals = set().union(*(row["intervals"] for row in report["simfyst"]["structure"]))
            g_intervals = set().union(*(row["intervals"] for row in report["simground"]["structure"]))
            report["comparison"] = {
                "hits_jaccard": float(np.count_nonzero(a & b) / np.count_nonzero(a | b)),
                "center_separation_deg": separation(fyst["center_deg"], ground["center_deg"]),
                "intervals_only_in_simground": sorted(g_intervals - f_intervals),
                "intervals_only_in_simfyst": sorted(f_intervals - g_intervals),
                "schedule_boresight_angle_deg": [float(scan.boresight_angle.to_value(u.deg)) for scan in schedule.scans],
                "note": "SimGround 直接把 FYST schedule 的天空系 boresight_angle 当 local roll（安装版 ops/sim_ground.py:771-776）；仅比较 I / hits，不作 Q/U 角度比较。",
                "motion": "SimGround mount rate=1 deg/s、accel=1 deg/s²、fix_rate_on_sky=False；与 FYST CES 参数相符，但原生换向 / 起始相位 / 端点采样算法不同。",
            }
    report["elapsed_s"] = round(time.perf_counter() - started, 3)
    report["evidence"] = "执行验证"
    (folder / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for engine in ("simfyst", "simground"):
        if engine in report:
            item = report[engine]
            print(f"{engine}: {item['observations']} Observations，{item['samples']} samples，mask={item['masked_fraction']:.3%}，hits={item['total_hits']}，像素={item['hit_pixels']}")
            print(f"  hits 加权中心={item['center_deg']}；signal-only I 最大残差={item['signal_only_I_max_residual_K']:.3e} K")
            if args.case == "ces":
                print(f"  patch 中心 330/−50 所在像素={item['patch_center_pixel']}；在合并 hits 足迹内={item['patch_center_in_footprint']}；该像素 hits={item['patch_center_pixel_hits']}（信息展示，不作为失败条件）")
            else:
                print(f"  距 patch 中心={item['patch_center_offset_deg']:.6f} deg")
            for row in item["structure"]:
                print(f"  {row['name']} / {row['samples']} samples；shared={row['shared']}；intervals={row['intervals']}")
    if "comparison" in report:
        print("SimGround 对照：" + json.dumps(report["comparison"], ensure_ascii=False))
    print(f"耗时 {report['elapsed_s']:.3f} s；文件：{folder}/input_*.fits、simfyst/[hits,I,Q,U,rcond].fits、PNG、report.json（对照另有 simground/）。")


if __name__ == "__main__":
    main()
