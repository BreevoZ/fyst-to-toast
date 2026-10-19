# W2：生成器进 library，默认生成多个 science blocks

2026-10-18；下列结果在本机实际执行。FYST 为 `f095c51` + 已有 UTC 补丁，
Python 使用现有 `fyst` 环境；没有修改外部 repo、安装依赖或下载 IERS。

## 为什么要改

以前在另一个 repo 生成后复制 ECSV，现在在 `fyst_toast` 内调用安装的 FYST，直接
保存 native scheduler / trajectory 的结果。library 提供可复用函数，thin CLI 只管
参数、进程环境和 IERS；notebook / tests 也可调用同一份核心。`examples/` 留给教学 demo。

## 改了什么

G1 未跟踪的 `examples/00_generate_trajectory.py` 已用普通 `mv` 搬至
[`scripts/generate_trajectory.py`](../../scripts/generate_trajectory.py)。核心在
[`src/fyst_toast/generate.py`](../../src/fyst_toast/generate.py)：

| 函数 / 位置 | 职责 |
|---|---|
| `generate.py:32`，`build_legacy_patch()` | 原样建立 legacy CES / Daisy ObservingPatch |
| `generate.py:64`，`generate_observing_timeline()` | `OverheadModel()` 全默认，scheduler step=30 s |
| `generate.py:82`，`write_timeline_ecsv()` | 原生 `write_timeline` 写 full / science ECSV |
| `generate.py:99`，`reconstruct_trajectories()` | 重新读 schedule，再调用 `schedule_to_trajectories`，检查没有漏块 |
| `generate.py:113`，`build_trajectory_table()` | 原样保存数组、units、`[first,last)` block metadata |
| `generate.py:222`，`generate_trajectory()` | 串起各步骤、覆盖保护、写 trajectory ECSV / report，返回 report |

唯一的逐样本 ECSV 字段定义在
[`src/fyst_toast/trajectory_h5.py:25`](../../src/fyst_toast/trajectory_h5.py#L25)：
`ECSV_*` 常量及 `TRAJECTORY_COLUMNS` / `TRAJECTORY_META_KEYS` /
`TRAJECTORY_BLOCK_KEYS`；生成器与 writer 共用它们。writer 只替换 key 引用，计算不变。

移植来源仍是本地 legacy working-tree examples 的 `01_fyst_schedule.py`、
`02_fyst_trajectory.py`、`_support/environment.py`，不冒称 upstream 发布代码；
[G1 来源 hashes](../../output/codex/G1_generate/source_hashes.json) 保留追溯。

## 目的

在自己的 repo 内一条命令完成一个阶段：FYST inputs → full/science schedule →
逐样本 trajectory ECSV → TOAST H5 → 手动查看 / 验证，不再跨 repo 复制生成物。
每次生成四个文件：`full_timeline.ecsv`、`schedule.ecsv`、`fyst_trajectory.ecsv`、
`report.json`。无 science 时只留 full timeline / report，CLI 非零退出。

## 关键概念

- `generate_timeline` 决定何时做什么；`schedule_to_trajectories` 生成 science block
  内每个 sample 的运动。后者不生成 calibration / slew 的轨迹，science 内仍有 turnaround。
- `full_timeline.ecsv` 保留所有 block types；`schedule.ecsv` 只选 science，保留原窗口。
  `scan_index` 是一次 visit，`subscan_index` 是其子段；使用 FYST 编号，不另切块。
- `time_s` 从窗口起点计时；`block_time_s` 保留原始 `Trajectory.times`；绝对时间为
  `timestamp_unix_s`。nominal block 起止和实际首末 sample 不同，不补点、不重采样。
- G1 的 **CES 30 min 实测**：retune 5 min → pointing 3 → focus 5 → skydip 5 →
  slew 约 22 s → retune 5 → science 约 **6.6 min / 3982 samples**，只有一个 science block。
  当前默认 retune=300 s、min scan=60 s、max scan=3600 s；长窗口给后续子段 / visits
  留出时间，但 science 数量也受可见性与 overhead 影响，不能仅按窗口长度除以 3600。
- 只有 `generate.py` 在函数内 import FYST。ECSV → H5 → loader 不依赖 FYST；
  `import fyst_toast`、`import fyst_toast.trajectory_h5` 均不导入 FYST / TOAST。
  library 不改 MPI、线程或 IERS 全局设置，由调用者配置；CLI 已配置离线 IERS。

**新共同默认值是 10800 s（3 h）**，为给定候选中两例均至少三个 science blocks 的最小值。
四档都实际运行了 generator + writer，counts 保留 scheduler 原样（未合并 idle）：

| 窗口 s | case | calibration / slew / science / idle | science samples |
|---:|---|---|---:|
| 7200 | CES | 6 / 1 / 2 / 0 | 54,983 |
| 7200 | Daisy | 7 / 2 / 2 / 0 | 52,959 |
| **10800** | **CES** | **7 / 1 / 3 / 0** | **87,983** |
| **10800** | **Daisy** | **10 / 3 / 3 / 0** | **80,970** |
| 14400 | CES | 12 / 1 / 4 / 23 | 99,928 |
| 14400 | Daisy | 13 / 4 / 4 / 0 | 108,960 |
| 21600 | CES | 14 / 2 / 5 / 177 | 120,890 |
| 21600 | Daisy | 18 / 6 / 6 / 0 | 167,950 |

[完整窗口对比](../../output/codex/W2_generate_library/duration_sweep.md) 列出每档、每例的
scan/subscan、UTC start/stop、samples、各 ECSV/H5 大小及运行时间；原始数值在
[B_runs.json](../../output/codex/W2_generate_library/B_runs.json)。
CES 起点仍为 `2026-08-02T01:25:00`，Daisy 为 `2026-03-15T04:00:00` UTC；
patch、velocity、timestep=0.1 s 等参数和 `OverheadModel()` 均未调整。

## 怎么跑完整条流水线

在 `fyst` 环境、repo 根目录运行；`ces_demo` 换成一个尚不存在的输出目录，
Daisy 则将 case 改为 `daisy` 并另选目录。前两条 CLI 会先设置串行环境并禁止 IERS 下载。

```bash
python scripts/generate_trajectory.py ces --output-dir output/generated/ces_demo \
  --iers-file ../fyst-trajectories/tests/data/finals2000A.all
python scripts/save_trajectory_h5.py output/generated/ces_demo/fyst_trajectory.ecsv \
  --output-dir output/generated/ces_demo/hdf5 \
  --iers-file ../fyst-trajectories/tests/data/finals2000A.all
MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python tools/ecsv_reader.py output/generated/ces_demo/fyst_trajectory.ecsv
MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python tools/h5reader.py output/generated/ces_demo/hdf5/fyst_ces_000_000.h5
```

ECSV viewer 也能直接打开 `schedule.ecsv` / `full_timeline.ecsv`；操作见
[tools/README](../../tools/README.md)。省略 `--output-dir` 时仍定位本 repo 的
`output/generated/<case>/`，与 cwd 无关。已有文件默认拒绝覆盖；优先换目录。
`--overwrite` 只替换生成器的四个文件，不刷新已有 H5；`--duration-s 1800` 可复现 G1。

## 怎么验证的

所有数值进程先设 `MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`，
生成与 writer 使用同一张显式离线 IERS 表。结果与命令在
[`output/codex/W2_generate_library/`](../../output/codex/W2_generate_library/)：

- **Part A**：1800 s CES / Daisy 的三个 ECSV、`report.json` 和两个 H5 与
  `output/generated/{ces,daisy}/` 基线整文件 byte-identical。report 无路径或字段差异。
  writer / loader 的 AST 在常量替换回字面量后与 HEAD 完全相同。
- **Part B 默认窗口**：`output/generated/{ces,daisy}_10800_w2_b/` 中的全部六个 H5
  通过 `tools/check_writer_roundtrip.py`，17 个 shared arrays 的存储 hash 与 12 个输入
  对应字段逐点核对通过；随后每块实际调用 `_load_trajectory_block` 和
  `_check_sample_rate(10.0, ...)`，均通过，没有调整任何容差。
- 最大 nominal-rate relative error：CES `6.020978118e-7`，Daisy `9.556497814e-11`，
  阈值 `1e-6`。六块 clock phase residual 均为 `2.384185791e-7 s`，
  各自 tolerance 均为 `4.768371582e-7 s`；[逐块数值](../../output/codex/W2_generate_library/default_blocks.json)。
- `pytest tests/`：**52 passed**。新测试只生成 1800 s CES，读回 ECSV 后核对全部 columns /
  metadata keys，再实际传给 `write_trajectory_blocks`。屏蔽 FYST：**45 passed、2 skipped**
  （两个 FYST-dependent test modules），writer 仍成功，H5 byte-identical。
- repo 外 import 边界检查、FYST imports 搜索、CES baseline、Ruff check/format、
  `git diff --check` 通过。baseline 报告 byte-identical；原 H5 / G1 outputs 保持不变。

手动读回验收：`python tools/check_writer_roundtrip.py output/generated/ces_demo/fyst_trajectory.ecsv output/generated/ces_demo/hdf5`。
完整执行记录见 [verification.json](../../output/codex/W2_generate_library/verification.json)。

## 还没做什么

生产 patch / 仪器 / roll 参数、Pong、SimFYST 的 H5 分支和 sky/noise simulation 尚未做。
Daisy 高 El 的上游 `PointingWarning` / `AccelerationLimitWarning` 仍出现（本轮峰值警告 2.83、2.43 > 1.50 deg/s²）；未调参，原文保存在 [Daisy 生成日志](../../output/codex/W2_generate_library/B_daisy_10800_generate.log)。
