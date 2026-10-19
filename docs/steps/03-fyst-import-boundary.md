# D1：FYST import boundary

执行日期：2026-10-18；起点：`main@8d1f070`。本步未 commit。

## 为什么要改、目的是什么

用户只控制 `fyst_toast`。FYST 内部 API 会随版本变化：例如 v0.10 的 changelog
记录移除了 `overhead.compute_nasmyth_rotation`，改用
`Coordinates.get_field_rotation_from_altaz`。这是源码记录，不是本步运行的测试。
已经生成的文件不应因为 trajectory generator 的内部 API 变化而无法处理。

边界是：未来的轨迹生成脚本可以 import FYST；下游 **ECSV → TOAST H5 →
SimFYST** 依赖文件约定。可选 contract test 仍 import FYST，用来检查双方约定。
本步解除 writer/checker 的最后两处运行时依赖，没有实现新的 SimFYST H5 分支。

## 改了什么

以下行号对应本步完成后的文件。

| 文件与行号 | 改动 |
|---|---|
| `src/fyst_toast/fyst_flags.py:1` | 新增四个 plain-int enum：UNCLASSIFIED=0、SCIENCE=1、TURNAROUND=2、RETUNE=3；docstring 记录源文件、80–83 行及完整 `f095c51` commit |
| `examples/02_save_trajectory.py:18,100,158` | 小 helper 用 Astropy `Table.read(..., format="ascii.ecsv").meta` 读 site；使用本地 flags，再按原单位构造 GroundSite |
| `tools/check_writer_roundtrip.py:57` | flags 改从 `fyst_toast.fyst_flags` 导入 |
| `tests/test_fyst_contract.py:19,47,56` | `importorskip`、四项 enum 对照、CES/Daisy 两项 site 精确对照 |
| `pyproject.toml:21` | `writer` extra 改名 `fyst`，保留 `>=0.10,<0.11` pin 和 UTC patch 注释；仅供未来 generation 和 contract test |
| `README.md:37,57`、`docs/validation.md:139` | 最小依赖名称/用途更新；历史 wheel 检查仍标明当时名称 |
| `docs/steps/03-fyst-import-boundary.md:1` | 本学习笔记与验证入口 |

## 关键概念与 site caveat

**Code independence 不等于 semantic independence。** 不 import FYST 后，仍需理解
文件中 `scan_flag` 的 FYST enum 含义；它们不是 TOAST bitmask。
仅复制数字可能静默过时，因此 contract test 与实际安装的 FYST 比较。
没有 FYST 时整份 contract 模块明确 skip，普通下游测试仍须通过。

Site 不能未经测量就替换。源码 `overhead/io.py::_site_from_meta` 在经纬度四位小数
匹配 FYST 时使用 canonical site，忽略存储高度。本步先运行 `measure_site.py`，
对两个 legacy schedules 分别比较直接 metadata 与当前 `read_timeline(...).site`：

| case | name（两边） | latitude / deg（两边） | longitude / deg（两边） | height / m（两边） |
|---|---|---:|---:|---:|
| CES | FYST | -22.985639 | -67.740278 | 5611.8 |
| Daisy | FYST | -22.985639 | -67.740278 | 5611.8 |

三项数值差均为 `0.0`，float hex 也相同，才继续改 writer。
[原始测量](../../output/codex/D1_fyst_import_boundary/site_measurement.json) 保存输入 hash、
实际导入路径及完整精度。helper 要求这四个 metadata 字段及 Quantity 单位，直接使用
存储值，**没有 snapping 或 canonical fallback**。本结果只证明这两份 fixture 等价；
不声称任意 schedule 都与 FYST 的隐式 site 重建相同。

测试默认从 repo 位置解析
`../legacy_repo/fyst-trajectories-sandbox-2026-10/examples/fyst_to_toast/output/{ces,daisy}/schedule.ecsv`。
`FYST_TOAST_TEST_SCHEDULE_ROOT` 可覆盖包含 `ces/`、`daisy/` 的目录；相对值也相对
repo，绝对值直接使用。缺文件时逐 case skip，原因给出缺失路径和该环境变量。

## 怎么验证的

本步实际使用已有 `fyst` env 的 Python 3.12.14 / TOAST 3.0.5；contract 对照为
FYST 0.10.0、`f095c51` 加已有 UTC patch。未安装依赖、未联网、未下载 IERS。
所有 Python 进程在导入数值库前设 `MPI_DISABLE=1 OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1`；writer 使用指定的离线 `finals2000A.all`。

Import blocker 位于本次 output 的 `blocker/fyst_trajectories/__init__.py`，导入即
抛出 `ImportError`。将该目录设为 `PYTHONPATH` 后，先确认查找路径命中 blocker、
显式 import 确实失败，再运行下游。这使意外 FYST import 会直接暴露。

| 实际执行 | 结果 |
|---|---|
| blocker 下 writer + roundtrip checker，CES / Daisy | 3 / 4 blocks、5,921 / 9,399 samples；17 个 shared arrays 的存储 hash 和 12 个输入数组对照通过 |
| blocker 下 `pytest tests/ -ra` | **43 passed, 1 skipped**；skip 是整个可选 contract 模块，原因明确 |
| blocker 下单独 import `fyst_toast.fyst_flags` | 通过，未加载 FYST 或 TOAST |
| 无 blocker，`pytest tests/ -ra` | **49 passed**，含全部 6 项 contract checks |
| 从其他 cwd 用相对 env override 指向缺失 schedule 目录 | **4 passed, 2 skipped**，两份缺失路径都正确相对 repo 解析 |
| 无 blocker，`examples/01_ces_baseline.py` | 600 samples、8 个 shared arrays、13 组 intervals、detdata 完全一致；报告与原 baseline 字节相同 |
| `grep -rn fyst_trajectories src examples tools` + AST import 检查 | 只有常量模块的来源 docstring；无 FYST imports |
| writer 科学处理 `try/finally` 的 AST 与 HEAD 比较 | 完全相同；field set、flag mapping、focalplane、roll、输出组织未改 |
| changed/new Python 文件 Ruff check / format；`git diff --check` | 全部通过 |

原始 inputs、IERS、7 个原 H5 及 baseline/reader/operator 源码共 15 个受保护文件，
执行前后 SHA-256 不变。新 H5 与 `output/{ces,daisy}/hdf5/` 的完整文件逐字节比较
全部通过；以下是每对文件共同的 SHA-256：

| 文件 | 新旧共同 SHA-256 |
|---|---|
| `ces/hdf5/fyst_ces_000_000.h5` | `f026bc9872cdc6cc7c289a642e9f4d992370032bab7361134ded306d125676b3` |
| `ces/hdf5/fyst_ces_000_001.h5` | `2a46b18a6a9e3a6aa9f5d197b42c0ae29d0958a073dece5ad588ef8430035635` |
| `ces/hdf5/fyst_ces_000_002.h5` | `028f1198a294d6ada8c58a4ae319bddda90609b10779c978f59a525c01c14178` |
| `daisy/hdf5/fyst_daisy_000_000.h5` | `1de21fe50d78290e765aa2fd9f63590b04941de894803ef35b9412fc85727fad` |
| `daisy/hdf5/fyst_daisy_001_000.h5` | `fe2ed543bb0910a634cbd2153de56fe5613390de070463ff053660d8080c7c79` |
| `daisy/hdf5/fyst_daisy_002_000.h5` | `334aa7f1b5615be64007484b8c94086d0e938df266f4028135f9ed2b98cf6e7c` |
| `daisy/hdf5/fyst_daisy_003_000.h5` | `409931e4caeeda4fcf3307ddb3207f6bbc3a234900a0e657b5ca8260f3d42425` |

原始日志、测量/验证脚本、新 H5 都在
[`output/codex/D1_fyst_import_boundary/`](../../output/codex/D1_fyst_import_boundary/)。
[`commands.json`](../../output/codex/D1_fyst_import_boundary/commands.json) 记录逐条实际命令、
cwd、关键环境变量及 exit code；
[`h5_sha256.json`](../../output/codex/D1_fyst_import_boundary/h5_sha256.json) 记录每个新旧 hash。
复跑 writer 时使用新输出目录，以免触发已有输出保护。

## 还没做什么

未生成新轨迹、未更改相邻仓库、未 commit。未验证其他 site metadata 或其他 FYST
版本；当前 CES/Daisy samples 只有 flags 1/2，四项 enum contract 通过不等于 0/3
科学映射已验收。没有改 SimFYST 接入、真实 focalplane / Nasmyth roll 或 MPI。
README / architecture / handoff 的全面状态整理留给 R1；本步只调整依赖名称/用途。
