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
    ↓ 未来 B2.2：结合调用者的 Telescope 接入 SimFYST（尚未实现）
TOAST Data.obs / Observation → 后续模拟 operators
```

另外，`examples/01_ces_baseline.py` 用同一 `GroundSchedule` / `Telescope` 分别运行
原生 `SimGround` 与当前 `SimFYST`；两者仍自己生成 CES，不经过上述 trajectory 链。
读取 schedule 成功不能证明逐样本运动保真。原生 `LoadHDF5` 恢复文件中的示意仪器与
指向；未来 `SimFYST` 则要使用调用者的仪器，二者职责不同，无需先另造 raw H5 schema。

| 层 | 职责 |
|---|---|
| `generate.py` | 唯一 import FYST 的运行模块；调用原生调度与重建，保存 ECSV 和 generation report |
| `trajectory_h5.py` | 共用 ECSV / H5 定义、内存 QTable → H5 writer、私有单 block loader 与时钟检查 |
| `fyst_flags.py` | 本地 plain-int enum；可选 contract tests 与 FYST 核对 |
| `sim_fyst.py` | CES baseline；尚无 `trajectory_file` 输入分支 |
| `scripts/` | 参数、路径、覆盖保护、进程环境、离线 IERS 和 JSON 清单 |
| `tools/` | 只读浏览及原生 LoadHDF5 roundtrip 验收，见 [工具说明](../tools/README.md) |

运行库只有 `fyst_to_toast.generate` 在函数内 import `fyst_trajectories`；下游只读文件。
contract tests 是核对外部约定的例外。library 不改 MPI / 线程 / IERS 设置；
导入顶层 package 不主动导入 TOAST / MPI，也不运行模拟。

## 生成与 ECSV 输入约定

生成器目前提供 CES / Daisy，默认 scheduling window 为 10800 s，scheduler step
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
- metadata 的 `case` 用于命名，`start_utc` 为窗口起点，`source_schedule_sha256`
  必须与实际 schedule 文件字节一致。`blocks` 保留原顺序及完整 metadata：
  `[first,last)`、编号、native/requested timestep、nominal block 起止和实际首末 sample 等。
- CLI 要求 blocks 连续覆盖输入、无空块 / 重复文件名、全表 timestamps 严格递增；
  writer 检查块内 `block_time_s` 步长。不补样本、不插值，也不生成未提供的 slew /
  calibration 运动。schedule nominal duration 与实际 sample coverage 可以不同。
- schedule header 的 `site_name`、`site_lat`、`site_lon`、`site_alt` 由 CLI 用 Astropy
  直接读入，转换为 deg / deg / m，不调用 FYST canonical-site fallback 或 snapping。
  CES / Daisy 历史输入的精确 site 对照见 本地笔记 `docs/steps/03-fyst-import-boundary.md`。

FYST UTC offset 修复的当前 / 历史 base、hash 和重建方式见
[补丁说明](patches/README.md)；同版本号不保证相同工作树行为。

## H5 与 loader 契约

writer 调用原生 `SaveHDF5`，每 block 写
`fyst_<case>_<scan_index:03d>_<subscan_index:03d>.h5`。
这是本 adapter 的组织方式，不是 HDF5 只能装一个 scan 的限制。

`_TRAJECTORY_FIELDS` 唯一定义 loader 返回的 12 个输入字段：原始绝对 / 相对时间、
Az/El 的 radian 与 degree 副本、deg/s 速度、FYST enum 及 scan/subscan 编号。
writer 另外保存 `flags`、`boresight_azel`、`boresight_radec`、`position`、`velocity`，
合计 **17 个 shared arrays**。TOAST `times` 为 Unix s，`azimuth` / `elevation` 为 rad；
`position` / `velocity` 是站点运动向量，不是 mount 角速度。

`_TRAJECTORY_INFO_KEY` 对应 JSON `trajectory_info`：输入 hash、block、roll、sample rate
等适配信息；`_SHARED_ARRAY_SHA256_KEY` 对应各 shared array 连续内存字节的 SHA-256。
CLI 的 `files.json` 列出本轮 H5，读取时用清单避免混入旧文件；`writer_report.json`
只表示写入完成，不等于读回验收通过。generator 的 `--overwrite` 只替换自己的四个
产物，不刷新 H5；writer 的覆盖选项不删除其他文件。

`_load_trajectory_block(path)` 使用原生 `LoadHDF5(detdata=[], force_serial=True)`：

- 要求恰好一个 Observation、12 个字段齐全且有限 / 一维 / 等长、`N >= 2`，合法 JSON
  与 `block.samples == N`，raw enum 属于 `{0,1,2,3}`；存在 hash manifest 时核对
  **全部已加载 shared arrays**。内容错误抛 `ValueError`，native IO 错误原样传播。
- 原样复制字段的 dtype / shape / bytes，不 wrap、重采样或再次切 block；返回
  `_TrajectoryBlock` 含数组、metadata / 原 JSON、hash manifest、site quantities 和时钟统计，
  不返回示意 focalplane。`finally` 清理临时 Data；frozen dataclass 不冻结数组内部。
- actual rate 为 `(N-1)/(times[-1]-times[0])` Hz；严格递增之外，参考均匀时钟的
  最大 phase residual 须 ≤ `max(1e-9, 2*max(spacing(abs(times))))` s。
  参考轴只作检查，不替换输入。
- `_check_sample_rate(nominal, actual, rtol=1e-6)` 另查
  `abs(nominal-actual)/actual <= rtol`，返回两种 rate、relative error 和 rtol。
  允许微小频差不表示允许 gaps / jitter；不规则采样策略尚未实现。

## Flags、区间与科学边界

FYST 是互斥 enum：UNCLASSIFIED=0、SCIENCE=1、TURNAROUND=2、RETUNE=3。
writer 保留原 enum，同时将 science 映射为 TOAST `0`、turnaround 映射为
`shared_mask_unstable_scanrate`（已验证 TOAST 3.0.5 中为 `4`），其余映射为
`shared_mask_invalid`（`1`）。TOAST flags 是可组合 bitmask，应按位判断，不能用
`flags == 4` 代替位测试。`0/3` 的最终科学处理尚未决定；合成测试通过不代表真实状态验收。

sample spans 使用半开 **`[first,last)`**，选取 `first` 到 `last-1`，长度为 `last-first`。
2026-10-18 已复现安装版 TOAST 3.0.5 `IntervalList` 端点 bug：
`samplespans=[(1,5)] | 空集` 得到 `[(1,6)]`；`~[(0,2),(3,6)]` 在六样本轴上得到空集，
预期却是 `[(2,3)]`。不要依赖 `|` / `~` 合并或反转这些区间；未来从 flags 直接计算
整数 spans，再建立 intervals。原生 SimGround 的区间合并也可能受影响，升级版本需重核。
本步复现日志见 本地笔记 `docs/steps/09-repo-docs-and-licenses.md`。writer 尚未保存扫描 interval
lists，loader 也未重建；原生 SimGround 从 intervals 写 flags，未来轨迹输入方向相反。

writer 用 native `fake_hexagon_focalplane` 的五个示意 detectors，按 TOAST longitude
`-Az` 构造 boresight，`--roll-deg` 默认 0°，再调用 `azel_to_radec(..., use_ephem=True)`
与 `site.position_velocity()`。IERS 表影响 pointing；复现时应选同一张表。
真实 FYST focalplane / Nasmyth roll 尚未验收。`detdata=[]`，没有 detector signal、
sky / atmosphere / noise simulation，也没有用独立 astrometry reference 验证 RA/Dec。

下一步是 **B2.2：SimFYST 接入单 block loader 与调用者 Telescope**；完整 flags /
intervals、方向区间、多 block、Pong、MPI 和真实仪器另行推进。
Daisy 高 El 的上游 acceleration warnings 仍未解决，历史现象及限制见 W2；
CES / Daisy 串行 IO 通过不代表所有 patterns 或运动约束已经验收。
