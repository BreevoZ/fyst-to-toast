# 验证记录

执行日期：**2026-10-18**。保留原 writer 迁移验证；依赖更新后的执行结果另见下方 R2 小节。
生成数据和完整本机报告位于被 Git 忽略的 `output/`，这里只保存可读结论与复现方法。

## 环境

原迁移验证：Python 3.12.14；TOAST 3.0.5；FYST trajectories 0.8.0；NumPy 2.5.2；SciPy 1.18.1；
Astropy 8.0.1；h5py 3.16.0；setuptools 84.0.0；Ruff 0.16.5。
这是本机已有环境，没有为本次验证下载或更新依赖。全部数值验证串行关闭 MPI，
`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`，IERS 自动下载关闭。

## FYST 本地兼容修改

当前依赖为 FYST 0.10.0（`f095c51` + [当前补丁](patches/fyst-trajectories-utc-offset_f095c51.patch)），
实际通过 env `fyst` 的静态 `.pth` 导入新 checkout；pip 0.8.0 metadata 已过时。
新机器重建见 [补丁说明](patches/README.md)。以下保留 **0.8.0 历史记录**，用于 fixture 来源追溯。

FYST 0.8.0 来自 base commit `2d0a3d395486fd4a43d83c8b38138e9b7bb6159b` 的本地
working tree，并非未修改的 release。相关修改是 `src/fyst_trajectories/overhead/io.py`
中已有的 `_iso_utc` / `_read_time`：输出显式 UTC offset，并通过 datetime fallback
读取这些时间。当前输入包含例如 `2026-08-02T01:25:00.000000+00:00`；未修改的代码
直接 `Time(value, scale="utc")`，该解析方式在本次 Astropy 环境中会失败。

已将这份既有文件 diff 原样保存在
[fyst-timeline-utc_2d0a3d3.patch](patches/fyst-timeline-utc_2d0a3d3.patch)，没有改写 reader 或自动 patch
依赖。`UPSTREAM.json` 记录 base commit、实际文件 hash 和 patch hash。
在临时目录用 base 文件执行 `git apply --check` / `git apply` 成功，应用后的文件与
本机实际使用的 `io.py` 逐字节一致；原 FYST checkout 未改变。

以下仅供复现历史 base `2d0a3d3`，不是当前依赖的安装步骤；应先检查目标是否已有修复，
在对应的历史 FYST checkout 内使用实际补丁路径：

```bash
git apply --check /path/to/fyst-to-toast/docs/patches/fyst-timeline-utc_2d0a3d3.patch
git apply /path/to/fyst-to-toast/docs/patches/fyst-timeline-utc_2d0a3d3.patch
```

已有修复的 working tree 不要重复应用。其他版本应先核对当前实现；补丁只验证过上述
base。普通发行版、新环境和完整 FYST working-tree 的其他改动未由这份补丁覆盖。

## R2：新依赖上的 writer 复跑

实际在 `f095c51` + 当前 UTC 补丁上执行：本地临时 clone 应用补丁后，4 个涉及文件与
新 checkout 逐字节相同；临时 clone 已删除。导入路径为 workspace 的
`fyst-trajectories/src/fyst_trajectories/__init__.py`，`__version__=0.10.0`，
`overhead/io.py` SHA-256 为 `32d9041d1af683579152f1ed7f7c4a86c92d110dc932c30cf4fbf356124458d3`。

使用下方相同 ECSV hashes 和新 clone 中字节相同的 IERS 表，CES / Daisy writer 及
`tools/check_writer_roundtrip.py` 均通过。与升级前 `output/{ces,daisy}/hdf5/` 比较：
3 / 4 个 Observations，样本分别为 `850 / 2611 / 2460` 和 `3000 / 3000 / 3000 / 399`；
名称、每个 Observation 的全部 17 个 shared arrays（dtype / shape / bytes）、
`trajectory_info`、`shared_array_sha256`、focalplane 与 site 全部相同。递归比较的
所有 H5 datasets / attributes 也相同；**7 个 H5 整文件 bytes / SHA-256 均相同，无差异**。
两个 case 的 `files.json`、`writer_report.json` 也逐字节相同。

`pytest tests/` 使用新输出运行，**43 passed，0 skipped**；CES baseline 的 600 samples、
8 个 shared arrays、13 组 intervals、detdata 与 Observation / Session 对照通过。
`git diff --check` 通过。此次没有重新生成 FYST schedule / trajectory，也没有重跑旧的
CLI / wheel 构建检查。命令和限制见 [R2 学习笔记](steps/02-fyst-dependency-patch.md)，
原始日志 / 完整对比报告在 `output/codex/R2_fyst_dependency/`。

## CES baseline

```bash
python examples/01_ces_baseline.py
```

实际另从 repo 外的临时工作目录运行脚本，默认报告路径仍定位到 repo 的 `output/`。
30 s × 20 Hz，600 samples，2 detectors。`SimFYST` 与原生 `SimGround` 的
8 个 shared arrays、13 组 interval lists（含 8 个 turnaround intervals）、两个 detdata
fields 和关键 Observation / Session 信息完全一致。代码 AST 对照也通过：仅 imports
路径和类名不同，注释不影响 AST。这不能证明 trajectory reader 已实现。

## 迁移 writer 与原生读回

在已有研究 workspace 中，从本 repo 根目录运行。将 `ces` 换为 `daisy` 可复现另一组：
输入已从旧 `../fyst-trajectories/examples/...` 移至下列归档路径；命令使用独立 R2 输出目录，原输出保留。

```bash
python scripts/save_trajectory_h5.py \
  ../legacy_repo/fyst-trajectories-sandbox-2026-10/examples/fyst_to_toast/output/ces/fyst_trajectory.ecsv \
  --iers-file ../fyst-trajectories/tests/data/finals2000A.all \
  --output-dir output/codex/R2_fyst_dependency/ces/hdf5

python tools/check_writer_roundtrip.py \
  ../legacy_repo/fyst-trajectories-sandbox-2026-10/examples/fyst_to_toast/output/ces/fyst_trajectory.ecsv \
  output/codex/R2_fyst_dependency/ces/hdf5
```

重复写入已有目录时需加 `--overwrite` 或选择新的 `--output-dir`。
这些输入来自既有本地 example，不随本 repo 提交；新 clone 需自行提供输入。
`--schedule` 在本次命令中采用默认的同目录 `schedule.ecsv`。

| case | native block 编号 `(scan, subscan)` | 每块 samples | 总 samples | 结果 |
|---|---|---|---:|---|
| CES | `(0,0)`, `(0,1)`, `(0,2)` | 850 / 2611 / 2460 | 5,921 | 写入与读回通过 |
| Daisy | `(0,0)`, `(1,0)`, `(2,0)`, `(3,0)` | 3000 / 3000 / 3000 / 399 | 9,399 | 写入与读回通过 |

逐 Observation 检查：

- native block、名称、样本数、metadata、sample rate、5 个示意 detectors 正确；没有 detdata。
- 全部 17 个 shared arrays 的 SHA-256 与写入前保存的 hash 一致，且数值有限。
- 12 个输入对应字段逐点精确相同，包括 timestamps、两种相对时间、Az/El 的 rad 与 deg、
  两轴速度、原 flags 和 scan/subscan 编号；flag 映射另行核对。
- boresight quaternions 归一化；从 `boresight_azel` 还原 Az/El 后，两组数据最大误差均为
  `2.842170943040401e-14 deg`，检查阈值为 `1e-9 deg`。
- 读回检查前后 H5 文件 hash 不变。Daisy 读回也从 repo 外的工作目录执行通过。

迁移前后逐 block 的计算 loop 通过 AST 对照：仅将旧 CLI 的 `args.case` 换为 metadata
中的 `case`，并缓存输入文件 hash。时间、pointing、flags、示意仪器和数组组织逻辑保留。
原文件 hash 通过 `UPSTREAM.json` 核对。没有将“存储 hash 一致”误称为独立天体测量验证；
RA/Dec 仍复用 TOAST 计算，本轮没有引入独立 astrometry reference。

输入内容标识（SHA-256）：

```text
CES trajectory
  a67ea4913709cf31454e9cbf279d4945e53c9dcf59bc818bc24fc6221faf50fd
CES schedule
  e0cbc5f80bd974ccb6870fdbf8d97ede5009149438bddadf7ca5f548e2b6194a
Daisy trajectory
  d4efb21502ec60aad144157367489299515b696f41df411eee5aae79c03faad0
Daisy schedule
  fd61a338253cdbe9c778967e969c9f2f0fb0330b1a86e942ee37dffc8a493dd8
Explicit IERS-A file (both cases)
  df9b19b7a6055871346909beb8ba127c46d464b36c47d5cb02a02ad0e9a2be06
```

## CLI 与打包

- `--help`、缺少输入时的错误提示、已有输出时拒绝覆盖、schedule hash 不匹配时拒绝写入，
  均已执行检查。拒绝写入后已有输出的文件 hash 未改变。
- 在临时目录以显式输入/输出路径运行、采用 Astropy 自带 IERS 数据、再次使用
  `--overwrite`，均通过；覆盖后也通过原生读回检查。
- 新 writer 和 roundtrip checker 通过 Ruff 检查；未为格式化而改写复制的 `SimFYST`。
- 从 repo 外单独 `import fyst_to_toast`，确认没有导入 TOAST / MPI。
- 使用以下离线命令构建 wheel 成功，并检查 wheel 内保留全部 license / authors / NOTICE、
  正确 license expression 和 extra dependency（当时名为 `writer`，现名为 `fyst`）：

```bash
python -m pip wheel --no-deps --no-build-isolation --no-index --no-cache-dir . \
  --wheel-dir /tmp/fyst-to-toast-release-check
```

wheel 是 package 安装检查；运行本文 example 的方式仍是从 Git checkout 执行脚本。
`.gitignore` 已实际核对：`output/` 中数据与报告、`.env`、本地 Claude settings 被忽略；
没有用 `*.h5` 等规则禁止未来明确需要版本管理的小型 fixtures。

## 尚未验证 / 尚未实现

- `SimFYST` 直接读 H5；真正的 FYST focalplane / Nasmyth roll；trajectory 输入的完整 intervals。
- MPI、多进程、全部支持范围内的依赖版本，以及新机器的全新依赖安装。
- 本轮数据的 FYST flags 只有 `1/2`，未用 `0/3` 样本做 roundtrip；Pong 本轮未运行。
- detector sky / atmosphere / noise simulation 和不规则采样策略。

后续最小任务见 [交接记录](HANDOFF.md)；上述未完成部分不影响本轮 writer 搬迁的验收。
