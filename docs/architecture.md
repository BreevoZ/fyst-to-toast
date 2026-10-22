# 架构与数据契约

这里维护当前接口；字段名以 [trajectory_h5.py](../src/fyst_to_toast/trajectory_h5.py)
中的常量为唯一来源。各步执行证据在本地学习笔记 `docs/steps/` 与历史快照 `docs/records/`（均不上传远端）。

## 数据流与职责

```text
fyst_to_toast.generate → FYST native scheduler
    ↓ full_timeline.ecsv（全部 blocks）+ schedule.ecsv（science-only）
FYST read_timeline → schedule_to_trajectories
    ↓ fyst_trajectory.ecsv（逐样本数组 + native block 清单）
scripts/save_trajectory_h5.py → trajectory_h5.write_trajectory_blocks
    ↓ TOAST SaveHDF5：每 native block 一个 Observation H5 + files.json
trajectory_h5._load_trajectory_block → 校验并复制原始轨迹
    ↓ SimFYST(trajectory_file=..., telescope=caller)：单 block geometry / motion classification
TOAST Data.obs / Observation → 后续模拟 operators
```

另外，`examples/01_ces_baseline.py` 用同一 `GroundSchedule` / `Telescope` 分别运行
原生 `SimGround` 与当前 `SimFYST`；两者仍自己生成 CES，不经过上述 trajectory 链。
读取 schedule 成功不能证明逐样本运动保真。原生 `LoadHDF5` 恢复文件中的示意仪器与
指向；`SimFYST` trajectory 分支使用调用者的仪器并重新计算 geometry，二者职责不同，
无需先另造 raw H5 schema。

| 层 | 职责 |
|---|---|
| `generate.py` | 唯一 import FYST 的运行模块；调用原生调度与重建，保存 ECSV 和 generation report |
| `trajectory_h5.py` | 共用 ECSV / H5 定义、内存 QTable → H5 writer、私有单 block loader 与时钟检查 |
| `fyst_flags.py` | 本地 plain-int enum；可选 contract tests 与 FYST 核对 |
| `sim_fyst.py` | 保留 CES baseline；`trajectory_file` 提前分支，使用 caller Telescope 建 Observation |
| `scripts/` | 参数、路径、覆盖保护、进程环境、离线 IERS 和 JSON 清单 |
| `tools/` | 只读浏览及原生 LoadHDF5 roundtrip 验收，见 [工具说明](../tools/README.md) |

运行库只有 `fyst_to_toast.generate` 在函数内 import `fyst_trajectories`；下游只读文件。
contract tests 是核对外部约定的例外。library 不改 MPI / 线程 / IERS 设置；
导入顶层 package 不主动导入 TOAST / MPI，也不运行模拟。

## 生成与 ECSV 输入约定

生成器目前提供 CES / Daisy / Pong，默认 scheduling window 为 10800 s，scheduler step
为 30 s，使用 `OverheadModel()` 默认值及原 legacy patch 参数；详见
本地笔记 `docs/steps/06-generate-with-fyst.md`。science 数量受 overhead / 可见性影响，
不能只按窗口时长推断。`scan_index` / `subscan_index` 保留 FYST 编号，不重新切块。
无 science 时只写 full timeline 和 `no_science` report，CLI 非零退出。

writer 接收生成器（或兼容的历史 exporter）保存的逐样本 QTable 和同轮 schedule，
不是任意列名相似的 ECSV，也不是 TOAST v5 schedule。
`ECSV_*`、`TRAJECTORY_COLUMNS`、`TRAJECTORY_META_KEYS`、`TRAJECTORY_BLOCK_KEYS`
定义完整列名与 metadata 清单；两端共用，不在文档维护第二份长表。

- `timestamp_unix_s` 是绝对 UTC Unix seconds；`time_s` 从 scheduling window 起点计时；
  `block_time_s` 保留原 `Trajectory.times`，不能假定每块从零开始。
- `az_deg` / `el_deg` 是 mount AltAz degree；两轴 `az_velocity` / `el_velocity`
  单位 deg/s。`scan_flag` 是 FYST enum，scan/subscan 为原始整数编号。
- RL-B 的 `boresight_roll_deg` 为 deg、float64、shape `(N,)`：直接调用 FYST 得到
  `ρ(t) = −(nasmyth_sign·El(t) + instrument_rotation)`，端口取自 schedule；
  `instrument_rotation_deg` / CLI `--instrument-rotation-deg` 默认 0，尚待仪器团队确认。
  ECSV metadata 记 `roll_source="fyst_focal_plane_rotation"`、`roll_nasmyth_port`、
  `roll_instrument_rotation_deg`；convention 仍为 `toast_q0_rz_local_z_v1`。
  不使用 schedule 的 `boresight_angle`；机械角不含视差旋转，由 TOAST 转天球时处理。
- metadata 的 `case` 用于命名，`start_utc` 为窗口起点，`source_schedule_sha256`
  必须与实际 schedule 文件字节一致。`blocks` 保留原顺序及完整 metadata：
  `[first,last)`、编号、native/requested timestep、nominal block 起止和实际首末 sample 等。
- CLI 要求 blocks 连续覆盖输入、无空块 / 重复文件名、全表 timestamps 严格递增；
  writer 检查块内 `block_time_s` 步长。不补样本、不插值，也不生成未提供的 slew /
  calibration 运动。schedule nominal duration 与实际 sample coverage 可以不同。
- schedule header 的 `site_name`、`site_lat`、`site_lon`、`site_alt` 由 CLI 用 Astropy
  直接读入，转换为 deg / deg / m，不调用 FYST canonical-site fallback 或 snapping。
  CES / Daisy 历史输入的精确 site 对照见 本地笔记 `docs/steps/03-fyst-import-boundary.md`。

FYST v0.11.0 起 schedule 时间为毫秒加 `+00:00` 的 UTC 文本（修复已进入 upstream）；
旧补丁、与 0.10 补丁版输出的差异和重建方式见 [补丁说明](patches/README.md)。

## H5 与 loader 契约

writer 调用原生 `SaveHDF5`，每 block 写
`fyst_<case>_<scan_index:03d>_<subscan_index:03d>.h5`。
这是本 adapter 的组织方式，不是 HDF5 只能装一个 scan 的限制。

`_TRAJECTORY_FIELDS` 唯一定义 loader 返回的 12 个输入字段：原始绝对 / 相对时间、
Az/El 的 radian 与 degree 副本、deg/s 速度、FYST enum 及 scan/subscan 编号。
writer 另外保存 `flags`、`boresight_azel`、`boresight_radec`、`position`、`velocity`，
以及 `boresight_roll`（rad、float64、shape `(N,)`），合计 **18 个 shared arrays**。
`_TRAJECTORY_ARRAY_FIELDS` 为原 12 字段加 roll，原 `_TRAJECTORY_FIELDS` 顺序保持不变。
TOAST `times` 为 Unix s，`azimuth` / `elevation` 为 rad；
`position` / `velocity` 是站点运动向量，不是 mount 角速度。

roll 的字段名、metadata keys、来源标签与 convention 在 `trajectory_h5.py` 统一定义。
`toast_q0_rz_local_z_v1` 表示 `Q0 = qa.from_lonlat_angles(-az, el, 0)`，右乘 local +Z
的 `Rz(roll)`；必须明确单位、正号与乘法侧，不能直接把其他库的 rotation 当成同一角度。
writer 有 ECSV roll 列时使用整段数组，要求上述 convention 与非空 `roll_source`；
此时非零 `--roll-deg` 被 CLI / library 拒绝。无列时广播 scalar `--roll-deg` 并记为
`legacy_scalar`。H5 `trajectory_info` 保存来源与 convention，legacy scalar 输入另保留
`roll_deg`；新 per-sample 输入不保存一个可能误导的 scalar 值。
`boresight_roll` 与其他 shared arrays 一起进入 manifest。

`_TRAJECTORY_INFO_KEY` 对应 JSON `trajectory_info`：输入 hash、block、roll、sample rate
等适配信息；`_SHARED_ARRAY_SHA256_KEY` 对应各 shared array 连续内存字节的 SHA-256。
CLI 的 `files.json` 列出本轮 H5，读取时用清单避免混入旧文件；`writer_report.json`
只表示写入完成，不等于读回验收通过。generator 的 `--overwrite` 只替换自己的四个
产物，不刷新 H5；writer 的覆盖选项不删除其他文件。

`_load_trajectory_block(path)` 使用原生 `LoadHDF5(detdata=[], force_serial=True)`：

- 要求恰好一个 Observation、12 个字段齐全且有限 / 一维 / 等长、`N >= 2`，合法 JSON
  与 `block.samples == N`，raw enum 属于 `{0,1,2,3}`；存在 hash manifest 时核对
  **全部已加载 shared arrays**。内容错误抛 `ValueError`，native IO 错误原样传播。
- 额外返回 `boresight_roll`；已有字段必须为 float64、一维、长度 N、有限，并通过
  manifest（若有）及 metadata convention / source 检查。旧 H5 无该字段时，从有限
  `trajectory_info.roll_deg` 广播生成，不改原始 JSON / manifest；返回的 `roll_source`
  为 `legacy_scalar`、`roll_convention` 为上述约定。无效 legacy scalar 由 loader 拒绝，
  SimFYST override 不绕过文件校验。
- 原样复制字段的 dtype / shape / bytes，不 wrap、重采样或再次切 block；返回
  `_TrajectoryBlock` 含数组、metadata / 原 JSON、hash manifest、site quantities 和时钟统计，
  不返回示意 focalplane。`finally` 清理临时 Data；frozen dataclass 不冻结数组内部。
- actual rate 为 `(N-1)/(times[-1]-times[0])` Hz；严格递增之外，参考均匀时钟的
  最大 phase residual 须 ≤ `max(1e-9, 2*max(spacing(abs(times))))` s。
  参考轴只作检查，不替换输入。
- `_check_sample_rate(nominal, actual, rtol=1e-6)` 另查
  `abs(nominal-actual)/actual <= rtol`，返回两种 rate、relative error 和 rtol。
  允许微小频差不表示允许 gaps / jitter；不规则采样策略尚未实现。

## SimFYST 单 block geometry / motion classification（B2.2–B2.3）

`trajectory_file=None` 保持原 schedule/CES 路径。提供文件时，`_exec()` 在任何 schedule
检查之前进入独立方法；既不读取 GroundSchedule，也不调用 CES / elnod generator。
geometry 不按 pattern 分支，当前用已提交 CES（2611 samples）和 Daisy（399 samples）
fixtures 验收。运行环境由 caller 配置，示例与 tests 禁止 IERS 下载。

- 必须显式提供有非空 Focalplane 与 GroundSite 的 caller Telescope；与 schedule /
  schedule_file 互斥，不支持 telescope_file。source demo instrument 只用于 loader 读取，
  不替换 caller 的 detector 名单、quaternions、sample rate、FOV、site 或 telescope 名称。
- caller 与 source EarthLocation 的 Cartesian 距离须 ≤1 mm；名称可以不同。
  `_check_sample_rate(caller_rate_Hz, actual_rate_Hz, rtol=1e-6)` 通过才继续，保留返回的
  两种 rate、relative error、rtol；标称 10 Hz 在两个 fixtures 上通过，20 Hz 被拒绝。
- 只支持空 `data.obs`、无 MPI communicator、world/group size=1、一个 group。
  禁止 distribute_time、detset/session split、任何 detector selection；weather（包括
  caller site 自带 weather）与任何 HWP 设置、非默认 motion 参数都明确拒绝。
  motion 包括 elnods、El modulation / steps、randomize_phase、track_azimuth、cosecant、
  scan rate / acceleration 和 fix_rate_on_sky；默认值在此分支不参与生成运动。
- 使用 `Observation(comm, caller_telescope, N, process_rows=1, sample_sets=[[N]], ...)`。
  name 来自 source Observation，Observation / Session UID 都用 `name_UID(name)`；Session
  首末是原 timestamp 转 UTC datetime，shared times 保留完整原始 float64 精度。
  校验 / 构造失败不会 append 半成品；已分配 Observation 用 native `clear()` 清理。
- `_TRAJECTORY_FIELDS` 的全部 12 列保留原名 / dtype / shape / bytes，包括 times、Az、El，
  不 wrap、不 round、不转换原始 degree / radian 副本。标准 output name traits 可设别名，
  原始列仍保留；拒绝同名碰撞。默认新增两个 boresight quaternions 和 position / velocity，
  加上 B2.3 的 shared flags 和 RL-A 的 roll，合计 18 个 shared columns。detdata 由 caller detectors 决定：
  默认 float64 signal（K）和 uint8 flags，native allocator 全部清零；对应 name 为 None 时跳过。
- boresight 使用 `qa.from_lonlat_angles(-az, el, 0)`；逐样本 roll **右乘**本地 +Z rotation。
  writer 与 SimFYST 共用 `_boresight_with_roll`：整段全零时跳过乘法以保持 Q0 字节，
  否则一次向量化处理全部 samples。`trajectory_boresight_angle=None` 使用 loader 返回的
  roll 数组；显式有限 scalar angular Quantity 广播替换整段，shared `boresight_roll`
  保存实际使用值，来源记为 `trajectory_boresight_angle`。不改 source JSON。
- `azel_to_radec` 接收原 times、caller site 和 operator 的 use_ephem / use_qpoint /
  so3g_compat_mode。已执行默认 PyEphem 与 Astropy backend 测试；qpoint / so3g 未验收。
  `site.position_velocity(times)` 的单位是 **km、km/s**，与 mount rates 不同。
  TOAST 3.0.5 GroundSite 当前返回 Earth-center barycentric motion，尚未补站点平移。

metadata 分开记录来源和当前计算：`trajectory_info` 为来源 JSON 原文；
`source_shared_array_sha256` 为来源 manifest 的 JSON，不冒充当前输出 hashes；
`simfyst_info` 是 JSON，`contract_version="RL-A"`，记录 source basename / 整文件 SHA-256、N / 首末
timestamp、rate check / clock residual、`roll_source` / `roll_convention`、caller telescope/site 名称、coordinate
backend 选项；`flags/intervals` 标明 motion 已派生、solar / directions 尚未派生，
`flag_mapping` 记录 enum → 实际整数 mask，`class_counts` 记录 science / turnaround / invalid
样本数。`roll_deg` 仅为常量数组的 scalar 摘要，随时间变化时为 null，实际值以 shared
`boresight_roll` 为准。Shared arrays 本身无单位属性；示例打印契约单位，detdata 则带 native units。

trajectory 分支创建 shared flags 及三个基础 motion intervals；不派生 solar / direction
intervals，也不运行 AzimuthRanges。`_provides()` 声明 shared flags 和三个基础 intervals，
保留实际提供的其他 shared / detdata；Observation 构造器自带的
`ALL_OBSERVATION_SAMPLES` 仅描述全样本结构，不是扫描分类。见
[教学示例](../examples/02_simfyst_trajectory.py) 与 [验收测试](../tests/test_sim_fyst_trajectory.py)。

## Flags、区间与科学边界

FYST 是互斥 enum：UNCLASSIFIED=0、SCIENCE=1、TURNAROUND=2、RETUNE=3。
writer 保留原 enum，同时将 science 映射为 TOAST `0`、turnaround 映射为
`shared_mask_unstable_scanrate`（已验证 TOAST 3.0.5 中为 `4`），其余映射为
`shared_mask_invalid`（`1`）。TOAST flags 是可组合 bitmask，应按位判断，不能用
`flags == 4` 代替位测试。`0/3` 的最终科学处理尚未决定；合成测试通过不代表真实状态验收。

B2.3 SimFYST 直接读取原 `fyst_scan_flag`，同样映射 science → 0、turnaround → caller
配置的 `turnaround_mask`、unclassified / retune → `defaults.shared_mask_invalid`，不复用
writer 保存的 flags。输出使用 `shared_flags` trait 指定的名称和 uint8 dtype；该名称必须
非空且不与其他 shared columns 冲突，turnaround_mask 必须在 uint8 范围内。
`scanning_interval` 对应 raw science、`turnaround_interval` 对应 raw turnaround，新增
`fyst_invalid_interval`（默认 `fyst_invalid`）对应 raw unclassified / retune；区间名称必须
非空、互异，且不能占用 `ALL_OBSERVATION_SAMPLES`。三类互斥并覆盖所有样本。
shared flags 用于下游按 bitmask 排除样本，intervals 用于 `obs.view[name]` 分段访问。

sample spans 使用半开 **`[first,last)`**，选取 `first` 到 `last-1`，长度为 `last-first`。
2026-10-18 已复现安装版 TOAST 3.0.5 `IntervalList` 端点 bug：
`samplespans=[(1,5)] | 空集` 得到 `[(1,6)]`；`~[(0,2),(3,6)]` 在六样本轴上得到空集，
预期却是 `[(2,3)]`。SimFYST 的 `_mask_to_sample_spans` 从原 enum 的布尔 masks 找连续
True runs，直接传给 `IntervalList(times, samplespans=...)`，不做时间端点 round trip，
不使用 `|`、`~` 或 `IntervalList(intervals=...)`。原生 SimGround 的区间合并也可能受影响，
升级版本需重核。writer / loader 不派生这些 motion intervals；由 SimFYST trajectory 分支完成。

writer 用 native `fake_hexagon_focalplane` 的五个示意 detectors，按 TOAST longitude
`-Az` 构造 boresight 并应用上述 roll，再调用 `azel_to_radec(..., use_ephem=True)`
与 `site.position_velocity()`。IERS 表影响 pointing；复现时应选同一张表。
真实 FYST focalplane 尚未验收；Nasmyth 端口与 instrument_rotation 待仪器团队确认。
`detdata=[]`，没有 detector signal、sky / atmosphere / noise simulation，也没有用独立
astrometry reference 验证 RA/Dec。`tools/check_writer_roundtrip.py` 兼容新（18 个）和
legacy（17 个）shared arrays。

后续是 **solar intervals** 与 **B3：方向区间**；多 block、MPI 和真实 focalplane 另行推进。
Daisy 高 El 的上游 acceleration warnings 仍未解决，历史现象及限制见 W2；
CES / Daisy 串行 IO 通过不代表所有 patterns 或运动约束已经验收。
