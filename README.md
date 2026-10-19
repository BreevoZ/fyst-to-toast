# fyst_toast

连接 FYST trajectory 与 TOAST Observation 的独立研究 package。
目标是让 FYST 决定望远镜怎样运动，让 TOAST 使用该轨迹进行后续模拟。

**当前有两个可运行入口：CES baseline，以及从旧 example 迁移的 HDF5 writer。**
`SimFYST` 仍是 `SimGround` 的 CES 副本，尚未接入 trajectory reader。
writer 已能保存原生 TOAST Observation HDF5，但它独立于 `SimFYST`。
本次实测结果和依赖版本见 [验证记录](docs/validation.md)。

## 目录与阅读顺序

```text
fyst_toast/
├── src/fyst_toast/
│   ├── __init__.py                 # 版本；不会主动 import TOAST / MPI
│   └── sim_fyst.py                 # 当前 CES baseline；未来的轨迹输入 operator
├── examples/
│   ├── 01_ces_baseline.py          # SimGround 与 SimFYST 的小规模行为对照
│   └── 02_save_trajectory.py       # 复用已有 writer；ECSV → TOAST Observation H5
├── tools/check_writer_roundtrip.py # 原生 LoadHDF5 读回，与输入数组逐点比较
├── docs/
│   ├── architecture.md            # 数据流、字段与各层职责
│   ├── validation.md              # 已运行的检查与已知限制
│   └── HANDOFF.md                 # 下一步任务与 Claude / Codex 交接
├── AGENTS.md                      # 共同开发约定
├── CLAUDE.md                      # Claude 接手入口
├── UPSTREAM.json                  # 被复制实现的来源与 SHA-256
└── output/                        # 本机生成物，不提交 Git
```

首次了解项目：本文 → [架构说明](docs/architecture.md) → 对应 example。
继续开发：先读 [AGENTS.md](AGENTS.md) 和 [交接记录](docs/HANDOFF.md)，再核对当前 diff。

## 环境与安装

需要 Python ≥ 3.10 和已安装的 TOAST。运行 writer 另需 FYST trajectories 0.10.x；
TOAST 本身有编译依赖，不由这些 example 自动安装。本机验证使用已有的 `fyst` conda
环境、TOAST 3.0.5 和 FYST trajectories 0.10.0（`f095c51` 加下述补丁），版本声明不等于
整个范围均已测试。

**当前使用的是带本地时间解析修复的 FYST 0.10.0 working tree。** 现有 schedule 使用
`+00:00` UTC offset；所测试 FYST commit 的未修改 reader 无法直接解析这种字符串。
当前 [兼容补丁](docs/patches/fyst-trajectories-utc-offset_f095c51.patch) 的来源、重建命令和
历史补丁区别见 [补丁说明](docs/patches/README.md)。仅安装同一版本号不保证能读取相同
输入；本 repo 不会自动修改安装的 FYST。当前 env 通过静态 `.pth` 导入新 checkout，
`__version__` 为 0.10.0；pip metadata 中残留的 0.8.0 不代表实际运行代码。

在已有依赖和 setuptools ≥ 77 的环境中，从本 repo 根目录运行：

```bash
conda activate fyst
python -m pip install --no-deps --no-build-isolation -e .
```

这是复用本机依赖的 editable install，不会更新它们。若使用其他环境，激活该环境即可，
环境名称不必叫 `fyst`。`pyproject.toml` 的 `writer` extra 声明额外的 FYST 依赖；
上面的 `--no-deps` 命令要求它已经安装。

串行 example 在 import 前关闭 MPI、限制数值线程，并关闭 IERS 自动下载；
package 本身不修改这些进程设置。不要在已初始化 MPI 的 notebook kernel 中依靠
修改环境变量关闭 MPI；请在首格配置后重启 kernel。

## 1. 验证当前 SimFYST baseline

```bash
python examples/01_ces_baseline.py
```

脚本建立同一个原生 `GroundSchedule`，分别用 `SimGround` 和 `SimFYST` 创建独立
`Data`，比较 shared arrays、intervals、detdata 和 Observation / Session 信息。
输入为 30 秒、20 Hz CES，两个示意 detectors；报告写到
`output/ces_baseline/report.json`。这是复制行为的对照，不使用 FYST trajectory。

当前 notebook 接口仍与 `SimGround` 相同：

```python
from fyst_toast.sim_fyst import SimFYST

sim = SimFYST(telescope=telescope, schedule=schedule, weather=None)
sim.apply(data)  # 新建的空 TOAST Data；需要先设置好运行环境
```

## 2. 将已有 FYST trajectory 写成 H5

准备同一轮产生的 `fyst_trajectory.ecsv` 和 `schedule.ecsv`。输入是本地 FYST example
第 02 步保存的逐样本表，包含 units、native block 清单及 schedule hash；
不是任意 ECSV，也不是 TOAST v5 schedule。具体字段见 [输入约定](docs/architecture.md#输入约定)。

```bash
python examples/02_save_trajectory.py /path/to/fyst_trajectory.ecsv \
  --schedule /path/to/schedule.ecsv \
  --output-dir output/ces/hdf5
```

将 `/path/to/` 替换为实际路径。`--schedule` 默认取 trajectory 同目录的 `schedule.ecsv`；
`--output-dir` 默认本 repo 的 `output/<case>/hdf5`，case 来自输入 metadata。
CES 和 Daisy 使用同一个脚本，切换输入即可，不需要改变 writer 参数或运动算法。

在当前研究 workspace 中，例如：

```bash
python examples/02_save_trajectory.py \
  ../legacy_repo/fyst-trajectories-sandbox-2026-10/examples/fyst_to_toast/output/ces/fyst_trajectory.ecsv \
  --iers-file ../fyst-trajectories/tests/data/finals2000A.all
```

这些相对路径只是已有 workspace 的运行示例，代码不依赖 sibling repo 的位置。
旧 `examples/fyst_to_toast` 是本地研究代码，不能假定普通上游 clone 含有它或它的输出。
本 repo 不附带研究数据；新机器需要自行提供符合约定的输入。已有旧 example 时可先
运行其中的 `01_fyst_schedule.py ces`、`02_fyst_trajectory.py ces`，Daisy 同理。

| 参数 | 含义 |
|---|---|
| `--iers-file FILE` | 显式离线 IERS-A 表；省略则用 Astropy 自带数据。复现 pointing 数值时使用同一张表 |
| `--roll-deg 0` | 固定 local boresight roll，默认 0°，仅为诊断设置 |
| `--overwrite` | 允许覆盖本轮同名输出；不删除目录中的其他文件 |
| `--help` | 查看完整 CLI，不加载 TOAST |

每个已有 native science block 生成一个 Observation 和一个 H5，保留 timestamps、
Az/El、速度与 FYST flags，不重新采样、不补块间空档。`files.json` 列出本轮文件，
`writer_report.json` 记录输入 hash 和写入设置；读取时使用清单，避免混入旧文件。

这沿用旧 writer 的 TOAST `SaveHDF5` schema：五个示意 detectors、固定 roll，
没有 detector signal，也没有扫描 interval lists。原始 flags 已保存，未来可据此重建
intervals。它还不是正式 FYST 仪器模型或完整 sky / noise simulation。

## 3. 用原生 TOAST 读回核对

```bash
python tools/check_writer_roundtrip.py /path/to/fyst_trajectory.ecsv output/ces/hdf5
```

此检查调用原生 `LoadHDF5`，验证每个 block、全部 17 个 shared arrays 的存储 hash，
并将原始轨迹字段、转换后的 Az/El 和 flags 与 ECSV 逐点比较。它是 writer 的验收工具，
不是 `SimFYST` trajectory reader。

下一步先实现单 native CES block 的轨迹输入，再扩展多 blocks 和其他 patterns。
无需先重造 HDF5 schema；接入点和验收条件见 [交接记录](docs/HANDOFF.md)。

## 来源与提交范围

`SimFYST` 保留原 `SimGround` 的结构与注释；writer 从既有本地 example 复制。
[UPSTREAM.json](UPSTREAM.json) 记录各自来源和文件 hash，
[NOTICE](NOTICE)、[LICENSE](LICENSE)、[AUTHORS](AUTHORS) 与 [LICENSES](LICENSES)
保留来源说明及许可。不要把本地未提交 example 误称为已发布的 FYST upstream 实现。

`.gitignore` 排除生成输出、Python 缓存、临时环境和本地 Claude settings；
代码、文档、来源记录可提交。推送前检查 `git status --short` 和实际待提交 diff。
Claude 可从 [CLAUDE.md](CLAUDE.md) 接手；这里提供交接约定，没有配置跨模型自动调用。
