# 当前架构与数据约定

状态日期：2026-10-18。代码和 [验证记录](validation.md) 是当前行为的依据。

## 三条数据流

**可运行的 baseline：**

```text
TOAST GroundSchedule + Telescope
          ↓
SimGround / 当前 SimFYST       自己生成 CES 运动
          ↓
Data.obs → Observation         shared / intervals / 初始 detdata
```

**可运行的独立 writer：**

```text
FYST 原生 scheduler / schedule_to_trajectories（本 repo 外）
          ↓
既有 example 第 02 步：逐样本 trajectory ECSV + 对应 schedule ECSV
          ↓
examples/02_save_trajectory.py
    原样取 native blocks → 创建示意 Observation → TOAST SaveHDF5
          ↓
每 block 一个 TOAST Observation H5 + files.json
          ↓
TOAST LoadHDF5 可恢复保存的 Observation（检查脚本已覆盖）
```

**尚未实现的目标：**

```text
已有 H5 中的原始轨迹 + 调用者传入的 Telescope
          ↓
SimFYST（读取轨迹；不重新生成 CES）
          ↓
标准 Data.obs / Observation → 后续 TOAST 模拟 operators
```

原 `SimGround` 的限制在于从 schedule 生成 CES 运动；TOAST Observation 的逐样本
arrays 可以承载其他 patterns。writer 已保存的 TOAST H5 可以提供开发输入，
当前不需要先设计另一种 H5 schema。原生 `LoadHDF5` 恢复示意仪器与已有指向；
未来 `SimFYST` 要结合调用者的仪器建立 observation，这两个入口的职责不同。

## 各层负责什么

| 层 | 职责 | 当前状态 |
|---|---|---|
| FYST repo | scheduler、science / overhead blocks、轨迹与原始分类 | 复用安装的原生实现，不在此重写 |
| `examples/02_save_trajectory.py` | ECSV 到示意 Observation 的字段适配，调用原生 IO | 已迁移，路径通过 CLI 指定 |
| `src/fyst_toast/sim_fyst.py` | 建立供后续 operators 使用的 Data / Observations | 仍是复制的 CES baseline |
| TOAST dependency | Data、Observation、Telescope、坐标变换、Save/LoadHDF5 等 | 通过正常 imports 复用 |
| `tests/check_writer_roundtrip.py` | 用原生 loader 检查存取和输入保真 | 验收工具，不是新的科学 reader |

writer 暂时保留为独立脚本，方便逐段阅读：它仍包含教学用 focalplane、roll 与 flag 策略。
正式 package 接口在单 block 接入时再确定。

## 输入约定

writer 接收本地旧 `02_fyst_trajectory.py` 导出的 `QTable`，保留其现有字段。
它不是任意 FYST ECSV 的通用 loader；调用原生 `read_timeline` 读取配套 schedule 的站点。

| trajectory 列 | 单位 / 含义 |
|---|---|
| `timestamp_unix_s` | s；绝对 UTC Unix 时间 |
| `time_s` | s；相对于 scheduling window 起点 |
| `block_time_s` | s；原生 `Trajectory.times`，不能假定每块都从 0 开始 |
| `az_deg`, `el_deg` | deg；mount AltAz |
| `az_velocity`, `el_velocity` | deg/s；mount 两轴速度 |
| `scan_flag` | FYST enum；保留原值 |
| `scan_index`, `subscan_index` | 原始整数编号 |

必要 metadata：

- `case`：pattern 名称，用于文件命名；`start_utc`：timeline 起点。
- `source_schedule_sha256`：对应 schedule 文件字节的 SHA-256，writer 要求完全匹配。
- `blocks`：原生 block 顺序表。writer 使用每项的 `first`、`last` 半开切片，
  `scan_index`、`subscan_index` 和 `native_timestep_s`；完整 block metadata 原样保留。
  旧 exporter 另记录 nominal block 起止时间、实际首末样本、样本数和请求步长。

blocks 必须连续覆盖输入表，命名不能重复，timestamps 严格递增。writer 沿用旧实现，
检查每块 `block_time_s` 的步长；它不插值以满足 sample rate。块间实际时间空档保留，
不会生成未提供的 slew / calibration 轨迹。schedule 的 nominal duration 与实际提供的
sample coverage 不一定相同，不能用 schedule 时长推断样本数。

## HDF5 中保存什么

使用 TOAST 原生 `SaveHDF5` schema，不直接用 h5py 设计新的 dataset 布局。
每个 block 一个文件是这个 example 的组织方式，并非 HDF5 格式只能装一个 scan。
文件名：`fyst_<case>_<scan_index:03d>_<subscan_index:03d>.h5`。

每个 Observation 包含以下 17 个 shared arrays：

| 名称 | 内容 |
|---|---|
| `times` | 原始绝对 Unix seconds |
| `azimuth`, `elevation` | 原始 Az/El 转换为 radians |
| `flags` | FYST enum 映射后的 TOAST uint8 bitmask |
| `boresight_azel`, `boresight_radec` | TOAST quaternion；由 Az/El 和固定 roll 计算 |
| `position`, `velocity` | TOAST 站点运动向量；不是 mount 角速度 |
| `fyst_time_s`, `fyst_block_time_s` | 原始两种相对时间 |
| `fyst_az_deg`, `fyst_el_deg` | 原始角度，deg |
| `fyst_az_velocity_deg_s`, `fyst_el_velocity_deg_s` | 原始两轴角速度，deg/s |
| `fyst_scan_flag` | 原始 FYST enum |
| `scan_index`, `subscan_index` | 原始编号 |

Observation metadata 中，`trajectory_info` 是 JSON 字符串，记录输入 hash、block、
roll 和采样率；`shared_array_sha256` 保存各 shared array 的连续内存字节 hash，
供 IO 往返核对。目录的 `files.json` 是本轮文件列表，`writer_report.json` 是本轮
写入报告；后者表示写入完成，不能单独作为读回验证通过的证据。

## 科学与实现边界

- 使用 native `fake_hexagon_focalplane`，取五个示意 detectors。不是实际 FYST focalplane。
- 按 TOAST longitude = `-Az` 构造 boresight；`--roll-deg` 为固定 local roll。
  真正的 FYST / Nasmyth 姿态尚待确认。
- `azel_to_radec(..., use_ephem=True)` 与 `site.position_velocity()` 复用 TOAST 实现。
  IERS 表会影响坐标结果；显式提供相同表可避免默认数据版本差异。
- 当前映射：FYST science `1` → TOAST `0`；turnaround `2` → unstable-scanrate 位
  （当前 `4`）；其他值 → invalid 位（当前 `1`）。使用 TOAST 常量，不将原始 enum
  直接当作 bitmask。FYST `0` / `3` 的最终科学处理尚未确定。
- `detdata=[]`，没有 sky、atmosphere、noise 或 detector signal。未创建扫描 interval
  lists；未来从原 flags 建立与 samples 一致的 intervals，不能误称此文件已拥有
  `SimGround` 的全部输出结构。
- 串行验证不代表 MPI 或真实仪器已验证；CES / Daisy 样本的通过也不代表全部 patterns。

下一步的最小范围和验收条件见 [HANDOFF](HANDOFF.md)。
