# R4：让 loader 测试有随仓库保存的小输入

执行日期：2026-10-18；起点为 `main` / `912a2ea`。本步未执行 Git commit。

## 为什么要改

原测试的正例、损坏副本和清理检查依赖被 Git 忽略的 `output/`。新 clone 没有这些
H5 时，大部分测试会 skip。随仓库保存两个小 fixture 后，不必先跑 scheduler 或
writer，也能执行核心 loader 检查。

## 改了什么、目的

- `data/fyst_ces_000_001.h5`、`data/fyst_daisy_003_000.h5`：直接复制用户指定的
  writer 输出，总计 636,076 bytes；与源文件 bytes / SHA-256 相同，没有重生成。
- [data/README.md](../../data/README.md)：记录用途、精确大小、hash、时间范围、
  来源命令和历史依赖。实测大小为 533,001 / 103,075 bytes，与原提示的
  “524 KB / 104 KB”不同；其他指定特征吻合。H5 metadata 未发现绝对文件路径。
- [tests/test_trajectory_h5.py](../../tests/test_trajectory_h5.py)：核心正例固定
  从 `data/` 读，逐字段对照独立 h5py 读取的 dtype / shape / bytes；负例只修改
  `tmp_path` 下的副本。保留原 CES 首块时钟检查和四个 Daisy blocks，扩展可选
  array 检查到全部 7 个 `output/` H5。`FYST_TOAST_TEST_OUTPUT` 只控制可选输入；
  缺失时给出 skip 原因，缺少 `data/` fixture 则失败。

目的在于提供稳定的 loader 回归输入，以及后续 SimFYST B2/B3 examples 的起点。
没有改 `src/`、examples、writer、原始 H5 或 R3 正在修改的文件。

## 为什么选这两个

CES 有 2611 samples，flags `{1:2531, 2:80}`；5 段 science sweeps 的 Az velocity
符号为 `-+-+-`，4 段完整 turnaround 各 20 samples，包含两种转向方向。首末均为
science，El 恒为 50°，适合后续从 flags 建立 CES intervals。

Daisy 有 399 samples，flags `{1:394, 2:5}`；block 开头处于 turnaround，只有
前 5 个样本为 flag 2，之后为 science。它覆盖 block 起点截断转向段的情形，El
在约 79.0446–80.5142° 之间变化，也能检查 loader 没有偷偷依赖 CES 的固定 El。
这些 fixture 性质检查不等于要求任意 trajectory 重新匹配 CES generator。

## 它们不能代表什么

每份只有一个 native block，没有真实 FYST flags 0/3，不覆盖完整 schedule、
多 block 空档或全部 patterns。数据来自升级前 FYST 0.8.0 working tree 和已有
UTC patch，写入使用 TOAST 3.0.5；本次没有用 v0.10 重生成。归档 ECSV 和 IERS
hash 本次只读核对通过，历史生成过程依据 `docs/validation.md`，并非本次执行。
五个示意 detectors 和固定 roll 不能代表已验收的 FYST focalplane / Nasmyth 姿态。

## 怎么验证的

本次使用用户指定的 `fyst` Python 3.12 环境。下列命令在 repo 根目录、激活该环境后
可重跑；实际完整 argv 和结果保存在 `output/codex/R4_sample_inputs/`。

```bash
export MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
AUDIT="$PWD/output/codex/R4_sample_inputs"
export MPLCONFIGDIR="$AUDIT/cache/matplotlib"
export XDG_CACHE_HOME="$AUDIT/cache" NUMBA_CACHE_DIR="$AUDIT/cache/numba"
unset FYST_TOAST_TEST_OUTPUT
python -m pytest tests/test_trajectory_h5.py -q -s -rs -p no:cacheprovider
R4_EMPTY_OUTPUT="$(mktemp -d)"
FYST_TOAST_TEST_OUTPUT="$R4_EMPTY_OUTPUT" \
  python -m pytest tests/test_trajectory_h5.py -q -s -rs -p no:cacheprovider
rmdir "$R4_EMPTY_OUTPUT"
python -m ruff check tests/test_trajectory_h5.py
python -m ruff format --check tests/test_trajectory_h5.py
git check-ignore -v data/fyst_ces_000_001.h5 data/fyst_daisy_003_000.h5
python examples/01_ces_baseline.py
```

本次执行结果：

- `output/` 在场：**43 passed，0 skipped**；空目录 override：**35 passed，8 skipped**。
  8 个 skips 是 7 块可选 array 对照与原 CES 首块时钟测试，均显示缺文件原因。
- 两个 fixture 的 12 个输入字段逐字节相同，样本数和 flag counts 符合上述实测；
  时钟均通过，nominal 10 Hz 通过、20 Hz 拒绝。10 类损坏分别在两个临时副本上
  验证拒绝；保留无 manifest、合成 enum 0/3、Data 清理与无效 rate 参数检查。
- 两轮 pytest 各自前后及 baseline 后，**2 个 fixture + 全部 7 个 output H5**
  的 SHA-256 均未改变。`git check-ignore -v` 无输出、exit 1，两个 fixture 不被忽略。
- Ruff check / format-check 通过。首次 check 的 `RUF007` 已改为 `pairwise`；
  首次只读 inspection 因 bytes 无法 JSON 编码而失败，递归解码后成功，未改 H5。
- CES baseline 实测通过：1 Observation、600 samples、2 detectors；8 个 shared
  arrays、13 组 interval lists、signal / flags 及所检查 metadata 与 SimGround 一致。
  全程串行、关闭 IERS 自动下载；测试禁止网络连接，没有安装或下载依赖。

原始记录：`input_inspection.json`、`provenance_hashes.json`、`copy_verification.json`、
`sha256_before.json` / `sha256_after*.json`、`pytest_output_present.log`、
`pytest_empty_override.log`、`ruff_check_final.log`、`ruff_format_check.log`、
`git_check_ignore.log`、`ces_baseline.log`。baseline 另生成
`output/ces_baseline/report.json`；日志目录内 `run_checks.py` 保存本次验证过程。

## 还没做什么

`SimFYST` 尚未接入 loader；本步不实现 flags 映射、interval 重建或新 examples。
没有验证 MPI、真实仪器、v0.10 重新生成的一致性，或 detector sky / noise simulation。
空目录 override 验证的是无需 `output/` 输入的测试行为，并非全新依赖安装测试。
