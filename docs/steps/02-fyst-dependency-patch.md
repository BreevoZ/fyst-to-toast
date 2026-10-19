# R2：记录新 FYST 依赖，并复跑 writer

执行：2026-10-18；`fyst_toast` 起点为 `main@19d5d1e`。这是依赖记录和兼容性验证，
没有修改 writer、loader、SimFYST 或已提交的 H5 fixtures。

## 为什么要改 / 目的

workspace 迁移后有四类记录过期：补丁仍针对旧 base `2d0a3d3`；`UPSTREAM.json`
仍把 0.8.0 当作当前依赖；writer extra 仍限制 0.8.x；运行示例仍指向已搬走的 ECSV。
FYST repo 不属于用户，不能在那里 commit；因此由 `fyst_toast` 保存当前未提交修复，
让别人能重建相同依赖，并用执行结果确认旧输入在新依赖上仍写出相同数据。

## 改了什么

| 文件 | 本步改动 |
|---|---|
| `docs/patches/fyst-trajectories-utc-offset_f095c51.patch` | 原样复制 C1 的 `change.patch`，包含代码、两份测试和文档，共 4 个文件 |
| `docs/patches/fyst-timeline-utc_2d0a3d3.patch` | `git mv` 旧补丁，内容和 hash 不变 |
| `docs/patches/README.md` | 区分当前 / 历史补丁，记录 base、hash、重建步骤和禁止重复应用 |
| `UPSTREAM.json` | 当前依赖改为 `f095c51` / 0.10.0 + 补丁；保留历史 fixture 来源；TOAST SimGround 来源字段未变 |
| `pyproject.toml` | writer extra 改为 `>=0.10,<0.11`，注释说明还需要补丁 |
| `README.md` | 当前依赖、实际导入方式和迁移后的输入路径 |
| `docs/validation.md` | 保留原验证历史，补充 R2 执行结果和当前命令 |
| `docs/HANDOFF.md` | 当前补丁入口、历史补丁区别和旧文件的新位置 |
| `data/README.md` | 历史补丁改名，补充迁移位置 / R2 链接，保留 fixtures 的 0.8.0 来源 |
| 本文 | 学习要点、执行命令、比较结果和边界 |

## 关键概念

**Patch file** 是相对于某个 base commit 的文本差异，不是完整源码或已提交的 commit。
`git apply --check` 先检查适用性，`git apply` 才修改目标 working tree；不要重复应用。
当前补丁针对 `f095c5176ab5f8b53aa19a549dc871ef2e91feeb`，SHA-256 为
`38b0926aac5f4ff49bd6b7043bb33a5ff8448f660ec3760a29dc896c4fa93eec`。
它输出带 offset 的微秒 UTC 时间，避免 TOAST 把 naive datetime 当成本地时区；
同时让 FYST reader 接受这些带 `+00:00` 的旧输入。

**Editable install** 让环境导入 checkout 中的源码，因此 working tree 的未提交改动也是
运行依赖的一部分。本机当前用静态 `.pth` 直接指向新 checkout 的 `src/`，不是本轮重新
执行安装；效果同样是读 working tree。实测 `__file__` 指向新 checkout、`__version__`
为 0.10.0，pip metadata 的 0.8.0 已过时。新机器安装步骤见 [补丁说明](../patches/README.md)。

**旧补丁要保留**：`data/` fixtures 和升级前输出产生于 0.8.0 + 旧补丁。新依赖可以读旧
输入，不代表历史数据已经由 0.10.0 重新生成；删除旧补丁会丢失生成来源的记录。

## 怎么验证的（实际执行）

以下从 `fyst_toast/` 根目录执行；本轮 Python 为 env `fyst` 的 Python 3.12.14，
TOAST 3.0.5。关闭 MPI / 数值多线程及 bytecode 写入；writer 自身关闭 IERS 下载。

```bash
conda activate fyst
export MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
R2_OUTPUT=output/codex/R2_fyst_dependency
export MPLCONFIGDIR="$PWD/$R2_OUTPUT/mplconfig"
export XDG_CACHE_HOME="$PWD/$R2_OUTPUT/cache"
mkdir -p "$R2_OUTPUT"
```

1. 从**本地** FYST repo 创建临时 clone，只修改临时副本：

```bash
git clone --no-checkout ../fyst-trajectories "$R2_OUTPUT/temporary_fyst_clone"
git -C "$R2_OUTPUT/temporary_fyst_clone" checkout f095c51
git -C "$R2_OUTPUT/temporary_fyst_clone" apply --check "$PWD/docs/patches/fyst-trajectories-utc-offset_f095c51.patch"
git -C "$R2_OUTPUT/temporary_fyst_clone" apply "$PWD/docs/patches/fyst-trajectories-utc-offset_f095c51.patch"
```

实际用 Python `read_bytes()` 比较 4 个 touched files，全部相同，随后删除临时 clone。
导入并打印 `fyst_trajectories.__file__`、`__version__` 和 `overhead/io.py` hash：
路径确为 workspace 的 `fyst-trajectories/src/fyst_trajectories/`，版本为 0.10.0，
hash 为 `32d9041d1af683579152f1ed7f7c4a86c92d110dc932c30cf4fbf356124458d3`。
完整命令 / 输出见 `patch_and_environment.log`，结构化结果见同名 `.json`。

2. 用归档输入写入独立目录，然后原生读回。再次执行应换新的输出目录，以保留本次结果：

```bash
R2_INPUT=../legacy_repo/fyst-trajectories-sandbox-2026-10/examples/fyst_to_toast/output
for case in ces daisy; do
  python scripts/save_trajectory_h5.py "$R2_INPUT/$case/fyst_trajectory.ecsv" \
    --schedule "$R2_INPUT/$case/schedule.ecsv" \
    --iers-file ../fyst-trajectories/tests/data/finals2000A.all \
    --output-dir "$R2_OUTPUT/$case/hdf5"
  python tools/check_writer_roundtrip.py "$R2_INPUT/$case/fyst_trajectory.ecsv" \
    "$R2_OUTPUT/$case/hdf5"
done
python "$R2_OUTPUT/compare_writer.py"
```

`compare_writer.py` 是本次保存在日志目录的只读核对脚本；比较对象为升级前
`output/{ces,daisy}/hdf5/` 与本次目录中的**每个** H5。输入 ECSV hashes 与
[原记录](../validation.md) 全部匹配；新旧 checkout 的 IERS 文件逐字节相同。

| case | Observations | 每块 samples | 总 samples | writer / roundtrip / 新旧对比 |
|---|---:|---|---:|---|
| CES | 3 | 850 / 2611 / 2460 | 5,921 | 全部通过 |
| Daisy | 4 | 3000 / 3000 / 3000 / 399 | 9,399 | 全部通过 |

名称、样本数、每块 17 个 shared arrays 的 dtype / shape / bytes、`trajectory_info`、
`shared_array_sha256`、focalplane（含 table metadata）和 site 全部相同。递归核对了
每文件 27 个 HDF5 objects、47 个 attributes、全部 19 个 datasets，没有内容差异。
**7 个 H5 整文件 bytes / SHA-256 也完全相同**；两组 `files.json`、`writer_report.json`
同样逐字节相同。本轮没有需要归因的文件层差异；不能预设重写 HDF5 一定会改变 bytes。
日志路径和依赖源码位置已改变，科学数据未变。详细结果在 `writer_comparison.json`。

3. 回归和变更范围检查：

```bash
FYST_TOAST_TEST_OUTPUT="$PWD/$R2_OUTPUT" python -m pytest tests/ -q \
  -o cache_dir="$R2_OUTPUT/pytest_cache" --basetemp="$R2_OUTPUT/pytest_tmp" \
  --junitxml="$R2_OUTPUT/pytest.xml"
python examples/01_ces_baseline.py --output-dir "$R2_OUTPUT/ces_baseline"
git diff --check
git diff --cached --check
```

结果为 **43 passed / 0 skipped**；测试的完整 writer 输入指向本轮新 H5，已提交 fixtures
也参与测试。CES baseline 的 600 samples、8 个 shared arrays、13 组 intervals、detdata
及 Observation / Session 对照均通过。diff 检查通过，受保护源码 / fixtures / 旧输出 / 输入
hashes 和原 FYST 的 Git 状态 / diff 未变。原始日志、比较脚本和运行脚本都在
`output/codex/R2_fyst_dependency/`（Git 忽略）。

## 还没做什么

没有重新生成 FYST schedule / trajectory，不据此声称 0.8 与 0.10 的 scheduler 行为一致；
没有覆盖 Pong、flags 0/3、MPI 或真实仪器 / sky / noise。没有重跑 FYST 全量测试、网络
clone、pip 安装或 wheel 构建；补丁 README 中的新机器命令是说明，不是本机执行记录。
未改 FYST / legacy / TOAST repos、源码、examples、tools、tests 或 `data/*.h5`；未 commit。
文档整体整合留给 R1。
