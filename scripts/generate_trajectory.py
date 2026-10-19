"""调用外部 FYST library：scheduler inputs → native schedule → 逐样本 ECSV。

移植自本地 legacy examples/fyst_to_toast/01_fyst_schedule.py、
02_fyst_trajectory.py 和 _support/environment.py；来源见 G1 学习笔记。
只负责调用原生 API 和保存返回数组，不另造 scheduler、切块或重采样。
"""

import argparse
import json
import math
import os
from pathlib import Path
from tempfile import gettempdir

# 串行 CLI 的进程设置，必须先于 NumPy / TOAST / FYST imports。
os.environ["MPI_DISABLE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from fyst_toast.generate import (
    DEFAULT_DURATION_S,
    DEFAULT_START,
    OUTPUT_NAMES,
    generate_trajectory,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["ces", "daisy"])
    parser.add_argument(
        "--start", help="UTC ISO 时间；省略时使用该 case 的 legacy 日期"
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"整个 scheduling window，含 overhead；默认 {DEFAULT_DURATION_S} s",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="默认本 repo 的 output/generated/<case>/"
    )
    parser.add_argument(
        "--iers-file", type=Path, help="离线 IERS-A 表；省略时使用 Astropy 自带数据"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="允许替换本脚本的四个同名输出"
    )
    args = parser.parse_args()
    if not math.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error("duration-s 必须是有限正数")
    folder = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else Path(__file__).resolve().parents[1] / "output/generated" / args.case
    )
    outputs = [folder / name for name in OUTPUT_NAMES]
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error("输出已存在；使用新的 --output-dir 或显式 --overwrite。")
    iers_path = args.iers_file.expanduser().resolve() if args.iers_file else None
    if iers_path is not None:
        if not iers_path.is_file():
            parser.error(f"找不到 IERS 文件：{iers_path}")
        if iers_path in outputs:
            parser.error("输出不能覆盖 IERS 输入文件")

    for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        default_cache = Path(gettempdir()) / "fyst-toast" / key.lower()
        cache = Path(os.environ.setdefault(key, str(default_cache)))
        cache.mkdir(parents=True, exist_ok=True)

    from astropy.coordinates import solar_system_ephemeris
    from astropy.time import Time
    from astropy.utils import iers

    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "error"
    if iers_path is not None:
        iers.earth_orientation_table.set(iers.IERS_A.open(str(iers_path)))
    solar_system_ephemeris.set("builtin")
    try:
        start = Time(args.start or DEFAULT_START[args.case], scale="utc", precision=6)
    except ValueError as error:
        parser.error(f"无法解析 UTC start：{error}")
    report = generate_trajectory(
        args.case,
        folder,
        start=start,
        duration_s=args.duration_s,
        iers_file=iers_path,
        overwrite=args.overwrite,
    )
    if report["status"] == "no_science":
        raise SystemExit(
            f"本窗口没有 science；已保存 full_timeline.ecsv 和 {folder / 'report.json'}"
        )
    print(json.dumps(report, indent=2))
    print(f"输出：{folder}")


if __name__ == "__main__":
    main()
