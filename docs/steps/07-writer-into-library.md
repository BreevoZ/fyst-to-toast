# W1：writer 搬入 library

日期：2026-10-18；基线 `main / 462a51d`。本步未 commit。

## 为什么要改

同一种 H5 格式原来由 example 中的 producer 和 library 中的 consumer 分别定义。
把两者放到 `trajectory_h5.py`，共用一份 12-field 清单及两个 Observation metadata
key，修改格式时就能在同一处看到写入端和读取端的约定。

## 改了什么

以下为本步完成后的源码入口（file:line）：

| 入口 | 职责 |
|---|---|
| `src/fyst_to_toast/trajectory_h5.py:24`，`_TRAJECTORY_FIELDS` | 唯一定义 12 个 trajectory 字段名；writer 按原顺序写，loader 按同一清单读。第 40–41 行定义两个共享 metadata key |
| `src/fyst_to_toast/trajectory_h5.py:209`，`write_trajectory_blocks()` | 接收内存 `QTable`（含 `meta['blocks']`、case、start_utc）、站点 name/deg/m、source hash、输出目录和 roll；创建原示意 Observations，调用原 `SaveHDF5`，返回有序 H5 文件名；`finally` 清理 `Data` |
| `src/fyst_to_toast/trajectory_h5.py:64`，`_load_trajectory_block()` | 从旧 `trajectory_input.py` 搬来，保留函数名、返回结构和校验行为 |
| `src/fyst_to_toast/trajectory_h5.py:158,178`，`_validate_clock()` / `_check_sample_rate()` | 保留时钟与采样率检查，不重采样 |
| `scripts/save_trajectory_h5.py:17,31,63`，`read_site_metadata()` / `parse_args()` / `main()` | 从旧 example 搬来；保留参数及默认值、imports 前环境设置、IERS、ECSV 读取、hash / overwrite / 输入安全检查，以及两个 JSON 文件的写入 |
| `tests/test_trajectory_h5.py:166`，`test_committed_write_load_roundtrip()` | 用已提交 CES / Daisy fixtures 重建输入，写到 `tmp_path` 再读回；逐字节核对 12 fields，并独立验证全部 17 个 shared arrays 的 hash manifest |

文件路径、命令、输入文件保护属于 CLI；内存数据到 H5 的适配属于 library。
IERS 由调用者配置，library 不改进程环境、不 import FYST；TOAST 仍只在函数内延迟导入。
CLI 默认输出目录不变，`read_site_metadata()` 仍留在 CLI 读取 schedule header。

使用三次 `git mv` 搬迁 writer、loader 和测试。旧文档只更新路径/名称，README 目录树
同步新位置；`tests/test_fyst_contract.py` 更新 `runpy` 路径。
`tools/README.md`、`docs/steps/04-data-readers.md` 和 `tools/check_writer_roundtrip.py`
没有引用被搬文件，检查后无需修改。来源记录只更新目标路径。

## 目的

让 pipeline CLI 和后续调用者复用同一 writer；此步仅改变代码归属。
字段、单位、flags 映射、五个 demo detectors、roll、pointing、metadata 内容、命名及
`SaveHDF5(volume=..., volume_index=None, detdata=[], force_serial=True)` 均保留。

## 关键概念

- **library / CLI / example**：library 提供可调用函数；CLI 组织文件与运行配置；example
  展示如何使用。教学用 CES baseline 仍在 `examples/`。
- **git mv**：记录旧路径删除和新路径加入，便于 Git 识别 rename、沿历史追踪；没有创建 commit。
  当前 `RM` 表示 rename 已进入 index，搬迁后的内容修改尚未 staged。
- **round-trip**：证明这两个 committed fixtures 的原始字段在 write→load 后保真，且 manifest
  与实际存储一致；不独立证明天体测量正确、完整 FYST 仪器正确、所有 patterns 或 MPI 可用。
- **重建的限制**：H5 保留了本块所需原始 degree / time / velocity / enum arrays，能精确恢复
  writer 的科学输入；没有完整多块 ECSV、schedule 文件或原 IERS 表。测试只把 block 的
  `first/last` 改为局部 `[0,N)`，保留其他 metadata，并用 Astropy bundled IERS 重算 pointing。
  因此此测试要求 12 fields 的 bytes 相同，不要求整个 H5 与 fixture 相同。
- **byte-identical**：纯搬迁不应改变任何输出。另用相同完整 ECSV 和 IERS 复跑，以整文件
  SHA-256 / bytes 相同为验收条件，覆盖 metadata、仪器及 derived arrays，而非仅数值近似。

## 怎么验证的

以下都是本步实际执行结果。使用已有 `fyst/bin/python`、TOAST 3.0.5；所有进程在数值库
导入前设置 `MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`，不下载 IERS。
CLI 显式使用 `../fyst-trajectories/tests/data/finals2000A.all`。

legacy 输入来自归档 `examples/fyst_to_toast/output/`；generated 输入来自本 repo
`output/generated/`。各组新文件与原目录的逐文件结果如下：

| 输入组 | H5 文件 | 整文件 bytes / SHA-256 |
|---|---|---|
| legacy CES | `fyst_ces_000_000.h5` | 相同 |
| legacy CES | `fyst_ces_000_001.h5` | 相同 |
| legacy CES | `fyst_ces_000_002.h5` | 相同 |
| legacy Daisy | `fyst_daisy_000_000.h5` | 相同 |
| legacy Daisy | `fyst_daisy_001_000.h5` | 相同 |
| legacy Daisy | `fyst_daisy_002_000.h5` | 相同 |
| legacy Daisy | `fyst_daisy_003_000.h5` | 相同 |
| generated CES | `fyst_ces_000_000.h5` | 相同 |
| generated Daisy | `fyst_daisy_000_000.h5` | 相同 |

四组的 `files.json`、`writer_report.json` 也逐字节相同，无路径或内容差异。
输出目录只出现在终端日志中，两个 JSON 的原格式未记录该目录。

- 正常环境 `pytest tests/`：**51 passed**。
- `PYTHONPATH` 放入抛 `ImportError` 的 FYST shadow package：writer 成功，3 个 CES H5 和
  两个 JSON 仍 byte-identical；pytest **45 passed, 1 skipped**，仅 FYST contract 模块跳过。
- 从 repo 外导入 `fyst_to_toast` 和 `trajectory_h5`，均未加载 TOAST / FYST。
- 用 `git archive HEAD` 加本步改动构成独立源码快照，无本地 output、无 W2 文件，确认导入
  来自快照；FYST 屏蔽时 **37 passed, 9 skipped**。两个新增 round-trip 都执行；跳过的是
  8 个已有可选 output 用例和 1 个 contract 模块。未重新安装依赖，也未冒称全新环境安装测试。
- `examples/01_ces_baseline.py`：600 samples、8 shared arrays、13 interval lists、detdata
  与原生 SimGround 全部一致，生成报告也与旧 baseline 报告 byte-identical。
- `tools/check_writer_roundtrip.py` 验收新 CES 输出通过：17 个 hashes、12 个输入字段及
  flags / boresight 等检查通过。Ruff check / format、Git staged / unstaged diff check 通过。
- AST 对照：解析 metadata 常量后，旧 loader/dataclass/clock/rate 与新实现相同；CLI 的
  `parse_args()` 和 `read_site_metadata()` 相同。774 个受保护文件 hashes 未变，包括两份
  W2 文件、committed H5、旧 output 及 baseline 脚本。

原始日志、新输出、完整 hashes 和执行命令均在
[`output/codex/W1_writer_library/`](../../output/codex/W1_writer_library/)：
`byte_identity.json` / `.log` 逐文件记录比较；`commands.jsonl` 记录命令；
`pytest_normal.log`、`pytest_blocked.log`、`pytest_fresh_snapshot.log`、`suite.log`
记录运行结果；`source_and_protection_audit.json` 和 `git_status_final.log` 保存最终审计。

单组 CLI 的复跑形式（请选择新的输出目录）：

```bash
export MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/save_trajectory_h5.py /path/to/fyst_trajectory.ecsv \
  --iers-file ../fyst-trajectories/tests/data/finals2000A.all \
  --output-dir output/codex/W1_writer_library/new-run
python -m pytest tests/ -q -rs -p no:cacheprovider
```

## 还没做什么

生成器搬迁和加长窗口属于 **W2**；两份 W2 未提交文件原样保留。
**SimFYST B2.2** 的 H5 分支尚未实现，`SimFYST` 仍是 CES baseline。
文档内容合并留给 R1；没有改变科学逻辑、生成新的 FYST 轨迹、修改相邻 repo 或提交代码。
