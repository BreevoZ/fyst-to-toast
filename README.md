# fyst-to-toast

将 FYST trajectory 转为 TOAST Observation 的独立研究 package：FYST 决定运动，
TOAST 使用逐样本轨迹。安装名称为 `fyst-to-toast`，import 名称为 `fyst_to_toast`。

**B2.2 已实现：`SimFYST` 可从一个 native trajectory H5 block 创建使用 caller Telescope
的 Observation，保留原始时间 / 轨迹并计算 geometry。** CES / Daisy 均可输入；未提供
`trajectory_file` 时仍走原 CES baseline。trajectory 模式尚未派生 shared flags / 扫描
intervals（B2.3）或方向区间（B3），也未验收真实仪器或 sky / noise simulation。

## 安装与运行环境

需要 Python ≥ 3.10、setuptools ≥ 77，以及已安装的 TOAST 等
[依赖](pyproject.toml)。生成器和可选 contract tests 另需 `fyst` extra 对应的
FYST trajectories 0.10.x；writer / loader 不 import FYST。
已验证环境为 TOAST 3.0.5、FYST 0.10.0（`f095c51` 加本地 UTC 补丁）；
版本范围不代表所有版本均已测试。补丁、实际导入路径与 pip metadata 的区别见
[依赖重建说明](docs/patches/README.md)，本 repo 不自动修改依赖。

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
python tools/check_writer_roundtrip.py "$RUN_DIR/fyst_trajectory.ecsv" "$RUN_DIR/hdf5"
```

生成结果为 `full_timeline.ecsv`、science-only `schedule.ecsv`、逐样本
`fyst_trajectory.ecsv` 和 `report.json`。writer 默认读取同目录的 schedule，核对其
SHA-256，每 native block 写一个 H5；`files.json` 列出本轮文件，
`writer_report.json` 记录写入设置。roundtrip 才验证原生 `LoadHDF5` 读回结果。
改为 `daisy` 并另选目录可运行 Daisy；已有输出默认拒绝覆盖。
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
python examples/01_ces_baseline.py --output-dir "$RUN_DIR/ces_baseline"
```

最后一条是独立的 30 s、20 Hz、两 detector CES 教学对照：比较 `SimGround` 与
`SimFYST` 的 Observation，不使用生成的 FYST trajectory。数据约定与限制见
[architecture](docs/architecture.md)。

## 单 block SimFYST 教学示例

直接使用已提交 fixture，无需重新生成轨迹：

```bash
export MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python examples/02_simfyst_trajectory.py
python tools/h5reader.py output/simfyst/fyst_ces_000_001/fyst_ces_000_001.h5 --tree
# 也可输入 Daisy；输出目录按 source Observation name 分开。
python examples/02_simfyst_trajectory.py data/fyst_daisy_003_000.h5
python -m pytest tests -q
```

示例显式建立 source site 坐标上的 caller Telescope（两个 boresight detectors、标称
10 Hz），打印字段 dtype / shape / 契约单位、metadata 与 rate check，再通过原生
`SaveHDF5` 保存；输出目录存在时必须传 `--overwrite` 才能重跑。
输出默认含 16 个 shared arrays（12 个原始输入 + 4 个派生 geometry）与清零的 detector
signal / flags；文件里的五个 demo detectors 不用作运行仪器。

library 用法为 `SimFYST(telescope=caller_telescope, trajectory_file=str(path)).apply(data)`，
要求空 `Data`、串行、caller site 与 source site 的 Cartesian 距离 ≤1 mm；caller sample
rate 与实际时钟相对误差 ≤1e-6，不重采样。`trajectory_boresight_angle=None` 使用来源
`roll_deg`，显式 angle Quantity 可覆盖并记录来源。weather、HWP、detector filtering /
splitting、修改运动的参数暂不支持；原 JSON 与 source hashes 单独保留。

## 目录与阅读入口

| 位置 | 用途 |
|---|---|
| `src/fyst_to_toast/` | `generate.py` 生成；`trajectory_h5.py` 共用字段、writer / loader；`fyst_flags.py` enum；`sim_fyst.py` CES / 单 block trajectory operator |
| `scripts/` | generate / save 两个 pipeline CLI |
| `examples/` | `01_ces_baseline.py` CES 对照；`02_simfyst_trajectory.py` caller instrument 教学 demo |
| `tools/`、`tests/`、[data/](data/README.md) | 手动检查、pytest、两份已提交 H5 fixtures |
| [docs/architecture.md](docs/architecture.md) | 当前数据流与稳定契约 |
| `docs/steps/`（仅本地） | 学习笔记与阅读顺序，不上传远端 |
| `docs/records/`（仅本地） | 2026-10-18 handoff / validation 历史快照，不上传远端 |
| [docs/patches/](docs/patches/README.md) | FYST 依赖补丁与重建说明；来源 commit / 文件 hash 另记于本地 `docs/UPSTREAM.json` |
| `output/` | 忽略的本机生成物和执行日志 |

本地另有 `AGENTS.md` / `CLAUDE.md`（给 AI 的开发规则）、`docs/steps/`、`docs/records/` 和 `docs/UPSTREAM.json`，
已在 `.gitignore` 中排除，只留在本机、不上传远端。

## 第三方代码与许可

- [`src/fyst_to_toast/sim_fyst.py`](src/fyst_to_toast/sim_fyst.py) 来自 TOAST `SimGround`（hpc4cmb/toast `9595449a`，`src/toast/ops/sim_ground.py`），
  保留 BSD-2-Clause：[许可](LICENSES/TOAST-BSD-2-Clause.txt) / [作者](LICENSES/TOAST-AUTHORS.txt)。
- `docs/patches/*.patch` 含 FYST 源码，保留 [FYST BSD-3-Clause](LICENSES/fyst-trajectories-BSD-3-Clause.txt)；
  补丁不自动应用，当前与历史 base 的区别见补丁说明。
- writer 源自用户的本地、当时未跟踪的 working-tree example，并非 FYST upstream
  已发布贡献；来源及原文件 hash 记录在本地 `docs/UPSTREAM.json`（不上传）。TOAST / FYST 仍是独立依赖。
  **用户自己的代码暂未声明许可：仓库保持私有期间保留所有权利（all rights reserved），分享前再决定许可。**
