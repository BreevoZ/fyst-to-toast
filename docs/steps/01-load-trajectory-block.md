# B2.1：读入一个 native trajectory block

执行日期：2026-10-18。范围只有独立 loader、它的测试和文档；未修改 `sim_fyst.py`。

## 为什么要改

从当前源码看，`SimFYST` 仍由 schedule 生成 CES，没有读取 FYST trajectory 的入口。
已有 writer 保存的是 TOAST `SaveHDF5` schema，因此复用安装版 TOAST 3.0.5 的
`LoadHDF5` 解码 Observation、metadata 和 site；库代码无需直接依赖 h5py 布局，
也无需再设计 schema。测试单独使用 read-only h5py，作为数组字节的独立对照。

## 改了什么

新增私有模块 `src/fyst_toast/trajectory_input.py`，没有从 package 顶层导出 API：

| 位置 | 职责 |
|---|---|
| `src/fyst_toast/trajectory_input.py:38`，`_TrajectoryBlock` | 小型 frozen dataclass，保存数组副本、Observation name、解析后的 `trajectory_info` 及原 JSON、source hash manifest、site quantities 和时钟统计 |
| `src/fyst_toast/trajectory_input.py:57`，`_load_trajectory_block(path)` | 延迟 import TOAST；以 `files=[path], detdata=[], force_serial=True` 读临时 `Data`；验证后 `np.copy` 所需数组，始终在 `finally` 中 `clear()` |
| `src/fyst_toast/trajectory_input.py:151`，`_validate_clock(times)` | 在结构及 finite 检查后，检查严格递增和均匀时钟，返回 actual rate、最大 phase residual 和时间容差 |
| `src/fyst_toast/trajectory_input.py:171`，`_check_sample_rate(nominal_rate_hz, actual_rate_hz, rtol=1e-6)` | 独立比较两个 Hz 数值；不匹配抛 `ValueError`，通过则返回含两种 rate、relative error 和 rtol 的 dict，供调用者记录 metadata |

另新增 `tests/test_trajectory_input.py`，并在 `docs/architecture.md` 增加一行模块说明。
`frozen` 只禁止替换 dataclass 属性；其中 arrays / dicts 仍可编辑，但已经独立于临时
TOAST 内存。返回的 `source_site` 只有 `lat` / `lon` / `height` Astropy quantities，
没有文件中的 Telescope 或示意 focalplane。

## 目的

现在可以独立读取一个 writer H5，得到已经验证的原始轨迹，供 B2.2 接入使用：

```python
from fyst_toast.trajectory_input import _check_sample_rate, _load_trajectory_block

block = _load_trajectory_block("output/ces/hdf5/fyst_ces_000_000.h5")
times = block.arrays["times"]  # 原始 Unix seconds；不是重新生成的时间轴
rate_info = _check_sample_rate(10.0, block.actual_rate_hz)
```

执行前须按下方命令配置串行环境。loader 要求恰好一个 Observation、12 个字段完整且
都是等长有限一维数组、`N >= 2`、合法 JSON 及 `block.samples == N`、raw enum 属于
`{0,1,2,3}`。若存在 source manifest，校验**全部已加载 shared arrays** 的 hash，
当前文件是 17 个；只返回所需的 12 个。轨迹校验错误抛清晰 `ValueError`，native IO
错误原样传播。没有 CES 专属运动检查，四个 Daisy blocks 也可使用同一入口。

## 关键概念

- **TOAST shared arrays**：Observation 中与 detector data 分开的共享数组；这里的
  `times`、Az/El、FYST flags 等沿 sample 轴排列。`detdata=[]` 跳过 detector data；
  默认 `shared=[]` 则加载所有 shared arrays。临时载入的 derived arrays 只校验 hash。
- **为什么逐字节保留 times**：大 Unix epoch 的浮点量化使相邻差值略有不同。
  重新生成时间、rounding 或单位来回转换会改变输入证据。12 个字段保持 dtype、shape、
  `tobytes()`，包括原 radians 和 degrees 副本；不 wrap、不重采样、不再次切 block。
- **actual 与 nominal rate**：`actual_rate = (N-1)/(t[-1]-t[0])`，单位 Hz。
  验证参考轴是 `t[0] + arange(N)/actual_rate`，只用于测量 residual，不作为输出。
  最大绝对 residual 必须不超过 `max(1e-9, 2*max(spacing(abs(times))))` seconds。
  nominal 比较另用 `abs(nominal-actual)/actual <= rtol`，默认 `1e-6`，可配置。
  这落实本次用户决定，替代 B1 对 nominal rate 也做严格 phase 检查的旧建议。
  本次 CES actual 为 `9.999997349033228 Hz`，10 Hz 的相对差约 `2.65e-7`，通过；
  20 Hz 拒绝。允许 nominal 的微小频差不表示允许时间轴有 gaps / jitter。
- **FYST enum 与 TOAST bitmask**：本步只保留并验证原 enum，尚未执行映射，也没有从
  flags 建立 intervals。不能把 `fyst_scan_flag` 数值直接当 TOAST shared flags。

## 怎么验证的

以下命令在 repo 根目录、已有 `fyst` Python 环境中重跑；没有安装或下载依赖。
本次实际使用用户指定的 `fyst` 解释器，TOAST 为 site-packages 中的 3.0.5。

```bash
conda activate fyst
export MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
AUDIT="$PWD/output/codex/B2_1_trajectory_input"
mkdir -p "$AUDIT"
export MPLCONFIGDIR="$AUDIT/cache/matplotlib"
export XDG_CACHE_HOME="$AUDIT/cache" NUMBA_CACHE_DIR="$AUDIT/cache/numba"

python -m pytest tests/test_trajectory_input.py -q -s -p no:cacheprovider \
  --basetemp="$AUDIT/pytest_tmp"
python examples/01_ces_baseline.py
(cd /tmp && python -c "import fyst_toast, sys; assert 'toast' not in sys.modules")
python -m ruff check src/fyst_toast/trajectory_input.py tests/test_trajectory_input.py
python -m ruff format --check src/fyst_toast/trajectory_input.py tests/test_trajectory_input.py
git diff --check
```

本次执行结果：

- **pytest：25 passed，0 skipped**。CES 首块与四个 Daisy blocks 均逐字段与 h5py
  比较 dtype / shape / bytes；CES 为 850 samples、raw flag counts `{1:830, 2:20}`。
  五块的最大 phase residual 均为 `2.384185791015625e-7 s`，容差为
  `4.76837158203125e-7 s`。
- 临时 H5 副本中的缺字段、非递增时间、插入 gap、flag 7、原字段 / derived array hash
  不匹配、`block.samples` 不匹配、NaN、二维字段、坏 JSON 均按预期拒绝。
  同时验证无 manifest 可读、合成 enum 0/3 原样保留、可配置 rtol 及临时 Data 清理。
  0/2 Observation 与加载异常路径通过测试注入验证，未声称真实 writer 会产生这些情况。
- 测试 session 前后 **全部 7 个原始 H5 SHA-256 相同**。测试禁用网络连接与 IERS
  自动下载。默认输入根目录相对本 repo 定位为 `output/`；可用
  `FYST_TOAST_TEST_OUTPUT` 指向另一份含 `ces/hdf5`、`daisy/hdf5` 的目录；缺文件时明确 skip。
- **CES baseline 通过**：1 Observation、600 samples、2 detectors；SimGround / SimFYST
  的 8 个 shared arrays、13 组 intervals、signal / flags 和所检查的 metadata 一致。
  repo 外 import 未载入 `toast` / `mpi4py`。Ruff 最终 check / format-check 均通过。
  首轮 Ruff 提示 `TRY004`；为文件内容错误保留约定的 `ValueError`，加局部说明后通过。

原始日志在 `output/codex/B2_1_trajectory_input/`：`pytest_first.log`、
`ces_baseline.log`、`import_outside_repo.log`、`ruff_check_final.log`、
`ruff_format_final.log`；另保留首次 Ruff 日志和输入 SHA-256 快照。

## 还没做什么

`SimFYST` 尚未调用 loader；B2.2 才接入 caller Telescope、创建输出 Observation。
本步没有重建 intervals / TOAST flags、重新计算 pointing、处理多 blocks，或验证 MPI、
真实 FYST focalplane / Nasmyth 姿态、sky / atmosphere / noise simulation。
`force_serial=True` 只选择串行 HDF5 IO，不能代替 import 前的 `MPI_DISABLE=1`。
原始 0/3 flags 的科学含义也未因合成测试通过而确定。
