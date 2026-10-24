# fyst-to-toast

将 FYST trajectory 转为 TOAST Observation 的独立研究 package：FYST 决定运动，
TOAST 使用逐样本轨迹。安装名称为 `fyst-to-toast`，import 名称为 `fyst_to_toast`。

**MB1 已实现：`SimFYST` 一次读取一个或多个 native trajectory H5，每个 block 创建一个
使用 caller Telescope 的 Observation，保留原始轨迹、计算 geometry，并从 FYST enum 派生
shared flags 和 scanning / turnaround / fyst_invalid intervals。** CES / Daisy 均可输入；
未设置 `trajectory_file` / `trajectory_files` 时仍走原 CES baseline。solar intervals、方向区间（B3）、真实仪器
及 sky / noise simulation 留待后续。

## 安装与运行环境

需要 Python ≥ 3.10、setuptools ≥ 77，以及已安装的 TOAST 等
[依赖](pyproject.toml)。生成器和可选 contract tests 另需 `fyst` extra 对应的
FYST trajectories 0.11.x；writer / loader 不 import FYST。
已验证环境为 TOAST 3.0.5、FYST 0.11.0（`c933b7a`，已内置 UTC offset 修复，无本地补丁）；
版本范围不代表所有版本均已测试。历史补丁、实际导入路径与 pip metadata 的区别见
依赖重建说明，本 repo 不自动修改依赖。

在已有依赖的环境、repo 根目录做 editable install（不会安装或更新依赖）：

```bash
conda activate fyst
export MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pip install --no-deps --no-build-isolation --no-index -e .
```

环境名称可自行替换。运行以下命令或 notebook 前也要设置上述三个变量；已初始化
MPI 的 kernel 需重启。CLI 禁止 IERS 自动下载，默认用 Astropy bundled data；
可给 generator / writer 都传 `--iers-file FILE` 使用同一张离线表。
library 不修改进程环境或 IERS 设置，调用者负责配置。

## 从生成到读回

以下命令从 repo 根目录执行；`RUN_DIR` 请选择尚未使用的目录。默认窗口 10800 s
包含 overhead，保留原生 science blocks、timestamps 和换向样本，不补空档或重采样。

```bash
export MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN_DIR=output/generated/ces_demo
export RUN_DIR
python scripts/generate_trajectory.py --help
python scripts/save_trajectory_h5.py --help
python scripts/generate_trajectory.py ces --output-dir "$RUN_DIR"
python scripts/save_trajectory_h5.py "$RUN_DIR/fyst_trajectory.ecsv" \
  --output-dir "$RUN_DIR/hdf5"
python tools/ecsv_reader.py "$RUN_DIR/schedule.ecsv" --view text --pager never --rows 5
python tools/ecsv_reader.py "$RUN_DIR/fyst_trajectory.ecsv" --view text --pager never \
  --rows 5 --columns time_s,az_deg,el_deg,scan_flag --full
python tools/h5reader.py "$RUN_DIR/hdf5/fyst_ces_000_000.h5" --view text --tree
```

生成结果为 `full_timeline.ecsv`、science-only `schedule.ecsv`、逐样本
`fyst_trajectory.ecsv` 和 `report.json`。writer 默认读取同目录的 schedule，核对其
SHA-256，每 native block 写一个 H5；`files.json` 列出本轮文件，
`writer_report.json` 记录写入设置。writer / loader 的原生 `LoadHDF5` roundtrip 由
`pytest tests/test_trajectory_h5.py` 验证；`tools/check_writer_roundtrip.py` 可手动核对
新 writer 的 18 个 shared arrays（含 `boresight_roll`）或旧 writer 的 17 个。
改为 `daisy` 或 `pong` 并另选目录可运行 Daisy / Pong（Pong 参数取自 FYST
`docs/trajectory_examples.rst` 的示例：RA/Dec 180°/−30°，2°×2°，0.4°/s，周期 240 s）；
已有输出默认拒绝覆盖。
完整参数见 `--help`，交互查看方式见 [tools](tools/README.md)。

独立 loader 的最小调用（`SimFYST` 的 trajectory 分支也复用此接口）：

```bash
python - <<'PY'
import os
from pathlib import Path
from astropy.utils import iers
iers.conf.auto_download = False
from fyst_to_toast.trajectory_h5 import _load_trajectory_block, _check_sample_rate
block = _load_trajectory_block(Path(os.environ["RUN_DIR"]) / "hdf5/fyst_ces_000_000.h5")
print(block.observation_name, len(block.arrays["times"]))
print(_check_sample_rate(10.0, block.actual_rate_hz))
PY
python examples/ces_baseline_vs_simground.py --output-dir "$RUN_DIR/ces_baseline"
```

最后一条是独立的 30 s、20 Hz、两 detector CES 教学对照：比较 `SimGround` 与
`SimFYST` 的 Observation，不使用生成的 FYST trajectory。数据约定与限制见
[architecture](docs/architecture.md)。

## SimFYST trajectory 用法与教学示例

直接使用已提交 fixture，无需重新生成轨迹：

```bash
export MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python examples/simfyst_from_h5.py
python tools/h5reader.py output/simfyst/fyst_ces_000_001/fyst_ces_000_001.h5 --tree
# 也可输入 Daisy；输出目录按 source Observation name 分开。
python examples/simfyst_from_h5.py data/fyst_daisy_003_000.h5
python -m pytest tests -q
```

示例显式建立 source site 坐标上的 caller Telescope（两个 boresight detectors、标称
10 Hz），打印字段 dtype / shape / 契约单位、metadata 与 rate check，再通过原生
`SaveHDF5` 保存；输出目录存在时必须传 `--overwrite` 才能重跑。
输出默认含 18 个 shared arrays（12 个原始输入 + per-sample roll + 4 个派生 geometry + uint8 shared flags）
与清零的 detector signal / flags；文件里的五个 demo detectors 不用作运行仪器。
FYST science → 0、turnaround → turnaround_mask（默认 4）、unclassified / retune → invalid
（默认 1）。三个 motion intervals 从原 enum 的整数 `[first,last)` spans 创建；示例打印
实际映射、各类样本数和区间，不使用存在端点 bug 的 IntervalList 集合运算。

单文件用法为 `SimFYST(telescope=caller_telescope, trajectory_file=str(path)).apply(data)`。
多文件可直接传 H5 路径列表，或 writer 的清单：

```python
SimFYST(telescope=caller_telescope, trajectory_files=["output/generated/daisy/hdf5/files.json"]).apply(data)
```

两个输入参数互斥；清单内路径相对清单目录解析，不支持 glob 或目录。先校验所有文件的
路径 / Observation name 唯一、样本时间区间不重叠及下述 site / rate 条件，再按首样本时间
升序构建；任一失败全部撤回，`data.obs` 保持为空。允许混合不同生成 run，每个 Observation
保留自己的来源，另记 `simfyst_info.trajectory_input_index`（清单展开后从 0 开始的输入序号）。
旧单文件 metadata 不变。

要求空 `Data`、串行、caller site 与 source site 的 Cartesian 距离 ≤1 mm；caller sample
rate 与实际时钟相对误差 ≤1e-6，不重采样。`trajectory_boresight_angle=None` 使用来源
per-sample roll；显式 scalar angle Quantity 广播覆盖整段并记录来源。weather、HWP、detector filtering /
splitting、修改运动的参数暂不支持；原 JSON 与 source hashes 单独保留。

**RL-B 已填入逐样本真实 roll。** ECSV `boresight_roll_deg`（deg、float64）由 FYST 计算
`ρ(t) = −nasmyth_sign·El(t)`；端口取自 schedule，
clocking 应逐模块用 `Rz(−c)` 作用在 focalplane detector 四元数上；FYST 的 i1–i6 是环位置编号，与仪器团队的 IM1–IM6 方向相反。metadata 记录
`roll_source="fyst_focal_plane_rotation"`、`roll_nasmyth_port`，
`roll_convention="toast_q0_rz_local_z_v1"` 不变。只用机械角，视差旋转由 TOAST 处理。
writer 转为 H5 shared `boresight_roll`（rad、float64、每 sample 一个值），纳入 hash manifest；
`trajectory_info` 和 `simfyst_info` 记录来源与 convention。该 convention 明确表示
`Q0 = from_lonlat_angles(-az, el, 0)` 右乘绕 local +Z 的 `Rz(roll)`。
缺列的 legacy ECSV 广播 `--roll-deg`，缺字段的 legacy H5 广播原 `trajectory_info.roll_deg`，
来源记为 `legacy_scalar`；已有 ECSV roll 列时，非零 `--roll-deg` 会报错。
全零 roll 跳过 quaternion 乘法，保持既有 boresight 数组字节；有非零值时整段向量化计算。
`data/` fixtures 仍保留 legacy 格式。

## 全流程 walkthrough

逐步重跑 生成 → H5 → 读回 → SimFYST，打印每个中间文件的内容，并用数值对照 FYST 与 TOAST
在时间、Az 方向、flags、roll、intervals 上的约定差异；roll 部分比较一个偏轴 1° detector
在四种 roll 输入下相对 FYST 的指向误差。图与 GIF 写到 `output/walkthrough/<case>/figures/`。

```bash
export MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python examples/pipeline_walkthrough.py            # 默认 daisy，含 walkthrough.gif
python examples/pipeline_walkthrough.py ces --no-gif
```

该 example 为对照直接 import FYST；library 的依赖边界不变。

## 下游冒烟测试

`04` 将全部 trajectory blocks 接入原生 TOAST pointing / pixels / weights、合成天空与可选
noise，再用 `CovarianceAndHits → BinMap` 输出 hits / I / Q / U FITS 和 PNG，同时核对无噪声 I 回收。
CES 可与同 schedule 的 `SimGround` 比较 hits 足迹和 Observation 结构；其天空系 schedule roll
被直接当 local roll 使用，所以不比较两边 Q/U 角度。
默认输出 `output/downstream/<case>/`，重跑加 `--overwrite`；`--no-noise` 可关闭噪声。

```bash
export MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python examples/downstream_maps.py ces --compare-simground
python examples/downstream_maps.py daisy
python examples/downstream_maps.py pong
```

## 目录与阅读入口

| 位置 | 用途 |
|---|---|
| `src/fyst_to_toast/` | `generate.py` 生成；`trajectory_h5.py` 共用字段、writer / loader；`fyst_flags.py` enum；`sim_fyst.py` CES / 串行多 block trajectory operator |
| `scripts/` | generate / save 两个 pipeline CLI |
| `examples/` | `ces_baseline_vs_simground.py` CES 对照；`simfyst_from_h5.py` caller instrument 教学 demo；`pipeline_walkthrough.py` 全流程与 FYST/TOAST 约定对照；`downstream_maps.py` 下游模拟、binned map 与 SimGround 对照 |
| `tools/`、`tests/`、[data/](data/README.md) | 手动检查、pytest、两份已提交 H5 fixtures |
| [docs/architecture.md](docs/architecture.md) | 当前数据流与稳定契约 |
| `docs/steps/`（仅本地） | 学习笔记与阅读顺序，不上传远端 |
| `docs/records/`（仅本地） | 2026-10-18 handoff / validation 历史快照，不上传远端 |
| `output/` | 忽略的本机生成物和执行日志 |

本地另有 `AGENTS.md` / `CLAUDE.md`（给 AI 的开发规则）、`docs/steps/`、`docs/records/` 和 `docs/UPSTREAM.json`，
已在 `.gitignore` 中排除，只留在本机、不上传远端。

## 第三方代码与许可

- [`src/fyst_to_toast/sim_fyst.py`](src/fyst_to_toast/sim_fyst.py) 来自 TOAST `SimGround`（hpc4cmb/toast `9595449a`，`src/toast/ops/sim_ground.py`），
  保留 BSD-2-Clause：[许可](LICENSES/TOAST-BSD-2-Clause.txt) / [作者](LICENSES/TOAST-AUTHORS.txt)。
- writer 源自用户的本地、当时未跟踪的 working-tree example，并非 FYST upstream
  已发布贡献；来源及原文件 hash 记录在本地 `docs/UPSTREAM.json`（不上传）。TOAST / FYST 仍是独立依赖。
  **用户自己的代码暂未声明许可：仓库保持私有期间保留所有权利（all rights reserved），分享前再决定许可。**
