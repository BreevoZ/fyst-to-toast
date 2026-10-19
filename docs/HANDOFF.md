# SimFYST 交接记录

状态日期：**2026-10-18**。这是可随 repo 带走的开发交接，不依赖此前聊天。
执行前请重新检查当前代码、Git diff 和 [验证记录](validation.md)。

## 研究目标与用户偏好

目标是让 FYST 负责实际运动，`SimFYST` 读取逐样本 trajectory，结合传入的 TOAST
`Telescope` 创建标准 `Observation`，供后续 pointing、sky、atmosphere、noise 等
operators 使用。TOAST 的数据结构可以承载非 CES 运动；目前限制来自
`SimGround` 的运动生成入口。

用户正在理解 repo，要求慢慢推进、结构清楚、可以手动在 terminal 重跑。
默认用中文解释，保留英文 class / API。优先复用已有 scheduler、trajectory generator、
writer 和 TOAST classes，不为了 demo 重造算法或人为改变 block 边界。

## 当前实现边界

| 部分 | 当前行为 | 尚未完成 |
|---|---|---|
| `src/fyst_to_toast/sim_fyst.py` | 复制 `SimGround`，调整类名、imports 和阶段说明；仍由 schedule 生成 CES | 无 `trajectory_file` 参数，无 trajectory reader |
| `examples/01_ces_baseline.py` | 同一短 CES 分别运行 `SimGround` / `SimFYST`，比较 Observation 内容 | 不是 FYST trajectory 保真测试，不证明 MPI 兼容 |
| `scripts/save_trajectory_h5.py` | 从既有 FYST trajectory ECSV 创建示意 Observation，再调用原生 `SaveHDF5` | 不经过 `SimFYST`，不实现新的 reader，不运行 sky / noise |

writer 来自旧 workspace example
`fyst-trajectories/examples/fyst_to_toast/hdf5/03_save_trajectory.py`。
该历史路径现位于 workspace 的 `legacy_repo/fyst-trajectories-sandbox-2026-10/` 下；输入 ECSV 同在其 `examples/fyst_to_toast/output/{ces,daisy}/`。
迁移时保留原有数组适配及原生 `SaveHDF5`，解除旧 example 路径依赖。
来源、复制时文件 hash 和 license 以 repo 的来源记录为准；该旧文件是本地
working-tree example，不能仅凭 FYST repo 的 HEAD 声称它是上游已提交的实现。

当前复用的是 **TOAST Observation HDF5 schema**，不是新设计的 raw trajectory
schema。后续可以从这些文件取原始 time / Az / El / FYST flags 作为开发输入。
如果目的只是恢复已保存的完整 Observation，原生 `LoadHDF5` 已有此职责。
`SimFYST` 的目标则是根据轨迹和调用者传入的仪器构建 Observation，不应把文件中
的示意 focalplane 当成未来运行时固定配置。

## 从哪里运行

使用已具备所需依赖的环境；完整安装说明见 [README](../README.md)。
以下命令在本 repo 根目录运行，输入路径由操作者指定：

```bash
python examples/01_ces_baseline.py
python scripts/save_trajectory_h5.py /path/to/fyst_trajectory.ecsv
```

writer 接口：

- positional argument：已有的 trajectory ECSV。
- `--schedule`：对应的 schedule ECSV，默认在输入文件同目录寻找 `schedule.ecsv`。
- `--output-dir`：默认本 repo 的 `output/<case>/hdf5`。
- `--iers-file`：可选的显式离线 IERS 表；不要依赖另一个 checkout 的 `tests/data` 路径。
- `--roll-deg`：默认 `0`，诊断用固定 local boresight roll。
- `--overwrite`：显式允许覆盖已有输出。

输入来自已有 FYST example 的原生 scheduler / trajectory 生成流程，不是任意
列名相似的 ECSV。reader / writer 需要的字段、metadata 和来源检查见
[架构说明](architecture.md) 与脚本的 `--help`。

**依赖的本地修改也需要交接：**当前 FYST 0.10.0（base `f095c51`）working tree 含
UTC-offset / microsecond 修复，使 `read_timeline` 可解析输入中的 `+00:00` 时间。
普通同版本安装不保证等价。当前 [兼容补丁](patches/fyst-trajectories-utc-offset_f095c51.patch)
及历史 fixture 的 [0.8.0 补丁](patches/fyst-timeline-utc_2d0a3d3.patch) 均由本 repo 保存；
base commit、hash、重建命令见 [补丁说明](patches/README.md)，实际 writer 对比见
[R2 学习笔记](steps/02-fyst-dependency-patch.md)。env 通过静态 `.pth` 导入新 checkout，
不要以残留的 pip 0.8.0 metadata 判断代码版本。本 repo 不自动修改依赖；不要在已有
修复的 checkout 重复应用，也不要悄悄删掉时间的 UTC offset。

串行 example 在 import 前关闭 MPI、限制数值线程；package 本身不改环境。
本机曾出现只 import MPI 即 idle 200% CPU 的情况，这属于本机已观察到的问题，
不推广为所有 TOAST / MPI 环境都如此。

## 数据约定和易混淆的概念

- **时间与角度：**TOAST `times` 使用 UTC Unix seconds；`azimuth` / `elevation`
  使用 radians，原 FYST degree arrays 另存。不得在读取时悄悄重采样或重建时钟。
- **block / Observation：**writer 使用输入 metadata 中已有的 native science block
  切片，每 block 一个 Observation、一个 H5。scan / subscan indices 保留为数据；
  不按名字重新猜测或拆分。块间空档不能直接连线伪装成真实 slew。
- **FYST flag / TOAST flags：**当前 adapter 保留 `fyst_scan_flag`。FYST science `1`
  对应无新增无效位，turnaround `2` 对应 TOAST `shared_mask_unstable_scanrate`
  （当前值 `4`），其他类别先映射为 invalid。这个诊断策略不是 retune / 未分类状态
  的最终科学定义。TOAST flags 可组合，判断某个位应使用 bitwise mask，不能只比较
  `flags == 4`。
- **intervals：**当前 writer 不保存扫描 intervals；旧流程在 load 后从逐样本 flags
  重建，避免此前观察到的 interval 时间边界 IO 歧义。当前 TOAST sample spans 是
  半开 `[first, last)`；升级版本应重核。原生 `SimGround` 是先生成 intervals 再写 flags，
  未来 trajectory 输入则需要从已有分类建立 intervals。
- **仪器与姿态：**当前 writer 用五个示意 detectors；固定 roll 只是 IO / pointing
  诊断设置。`boresight_azel` 通过 TOAST 的 `-Az` longitude 约定构造，随后转换为
  `boresight_radec`。真实 FYST / Nasmyth roll、focalplane 和噪声参数仍需确认。
- **速度：**TOAST `position` / `velocity` 是站点的运动向量，不能当作 mount Az/El
  角速度；原始 FYST axis velocities 是另外保存的字段。
- **模拟范围：**保存的 boresight 和示意仪器不代表已经完成 detector sky / atmosphere /
  noise simulation。writer 使用 `detdata=[]`，不能把这些文件称为最终 detector timestream。

## 已有证据与下一步

2026-10-15 的串行 baseline 已在本地 TOAST 3.0.5 下验证：30 s、20 Hz、600 samples、
两个 detectors；全部 shared arrays、13 组 interval lists、detdata 和关键 Observation /
Session 信息与原 `SimGround` 一致。此结果只证明当时复制出的 CES baseline；
依赖版本范围不代表所有版本均已测试。

**本次迁移 writer 的执行命令、结果与限制以 [validation.md](validation.md) 为准。**
不能把旧目录的测试结果或上述 baseline 当作新 writer 已通过的证据。

下一个建议任务是单 native CES block 的 trajectory 读入，不需要先改全部 operator：

1. 检查复用 H5 的实际字段和版本约定，确定如何取出 timestamps、Az/El、原始 FYST flags
   及 block metadata。保留已有 writer / baseline 作为输入和对照。
2. 审查 `SimFYST._exec()` 的 schedule、时间和样本数量假设，再确定最小接入点；
   不仅仅替换 `_simulate_scanning()` 后就假定外层逻辑适用。
3. 用调用者传入的 `Telescope` 创建一个 Observation；显式检查 sample rate、单位和时间
   有效性。输入中的五个示意 detectors 不应成为隐藏的运行时约束。
4. 验收输入 timestamps / Az / El / 原 flags 被逐样本保留，TOAST flags 与 intervals 一致，
   boresight 有限且符合坐标约定。验收目标是忠实于 FYST 输入，不是重新匹配 CES generator。

随后再扩展多 blocks、Daisy / Pong、完整仪器和 MPI。未决项包括：真实 boresight roll、
不规则采样或不同采样率的处理、FYST `0` / retune 等状态的最终映射，以及块间空档的
语义。不要未经确认在 reader 中填充、插值或生成这些空档。

## Claude / Codex 协作方式

预期由 Claude 协助用户理解与规划、拆分小任务并 review；Codex 执行具体修改与验证。
这里没有配置跨模型自动调用，也不假设双方共享会话。可以手动复制以下信息；
源码、diff 和可复现命令比聊天中的完成声明更可靠。

给执行者的任务模板：

```text
目标：这一步要得到什么可观察行为？
范围：允许修改哪些文件；哪些科学参数或接口需保留？
输入：文件、schema、版本、相关现有实现。
验收：要运行的命令，以及应比较的实际数组 / 属性。
交付：diff 摘要、命令和结果、产物位置、失败或尚未验证的部分。
```

review 时核对当前 diff、是否复用原生实现、是否保留输入信息、是否把示意配置误当
正式仪器配置，以及检查是否覆盖本次修改。继续工作前先看 `git status`；commit / remote
状态会变化，不把交接时的状态当成永久前提。用户若调整方向，以最新明确指令为准。
