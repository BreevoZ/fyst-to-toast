# 小型 trajectory 样本输入

这两个文件供 loader 测试和后续 `SimFYST` examples 使用。新 clone 在已有 TOAST
依赖环境中即可运行核心测试，无需先生成被 Git 忽略的 `output/`，也不需要安装 FYST
来重新生成轨迹。它们保存的是 TOAST `SaveHDF5` Observation schema。

2026-10-18 从下列既有 writer 输出**逐字节复制**；没有重写 metadata、重新切 block
或重采样。副本与源文件的 bytes 及 SHA-256 均已实测相同：

| 本目录文件 | 复制来源（相对 repo 根目录） | 大小 | samples | 原始 FYST flag counts |
|---|---|---:|---:|---|
| `fyst_ces_000_001.h5` | `output/ces/hdf5/fyst_ces_000_001.h5`（现位于 `output/legacy/ces/hdf5/`） | 533,001 bytes（520.509 KiB） | 2611 | `{1: 2531, 2: 80}` |
| `fyst_daisy_003_000.h5` | `output/daisy/hdf5/fyst_daisy_003_000.h5`（现位于 `output/legacy/daisy/hdf5/`） | 103,075 bytes（100.659 KiB） | 399 | `{1: 394, 2: 5}` |

总大小为 636,076 bytes。实测大小与任务描述中的“524 KB / 104 KB”不同，以以上精确
bytes 为准；样本数、flag counts 和下述运动特征与任务描述相符。

```text
fyst_ces_000_001.h5
  2a46b18a6a9e3a6aa9f5d197b42c0ae29d0958a073dece5ad588ef8430035635
fyst_daisy_003_000.h5
  409931e4caeeda4fcf3307ddb3207f6bbc3a234900a0e657b5ca8260f3d42425
```

## 各自覆盖什么

- **CES `(scan=0, subscan=1)`**：后续 SimFYST B2/B3 的代表性 CES 输入。首末样本均为
  science（enum `1`），5 个连续 science runs 的 Az velocity 符号为 `- + - + -`；
  中间 4 个完整 turnaround runs（enum `2`），各 20 samples，覆盖 `- → +` 和
  `+ → -` 两种方向。El 恒为 `50 deg`。这些是当前样本性质，不是通用 loader
  应强制要求的 CES generator 行为。
- **Daisy `(scan=3, subscan=0)`**：非 CES 输入及 block 起点截断 turnaround 的边界
  情况。样本 `[0, 5)` 为 turnaround，随后 `[5, 399)` 为 science；block 内没有
  开头转向段之前的 science，不能假定每个 block 都以完整 sweep 开始。El 从
  `79.04457108202241` 到 `80.51415846553718 deg`，science 中 Az velocity 可变号，
  不能套用 CES 的固定 El / 等速 sweep 假设。

以下时间由原始 `shared/times` 实测，单位为 UTC Unix seconds；UTC 文本仅显示到
微秒。span 定义为末样本减首样本，不是 nominal schedule duration 或 `N / rate`。

| 测量 | CES | Daisy |
|---|---|---|
| 首样本 Unix s | `1785635187.8999493` | `1773548960.101578` |
| 末样本 Unix s | `1785635448.9000187` | `1773548999.901578` |
| 首样本 UTC | `2026-08-02T01:46:27.899949+00:00` | `2026-03-15T04:29:20.101578+00:00` |
| 末样本 UTC | `2026-08-02T01:50:48.900019+00:00` | `2026-03-15T04:29:59.901578+00:00` |
| span / s | `261.0000693798065` | `39.799999952316284` |
| actual rate `(N-1)/span` / Hz | `9.999997341770571` | `10.000000011980832` |

两块的最大 clock phase residual 均为 `2.384185791015625e-7 s`，小于当前 loader
容差 `4.76837158203125e-7 s`；nominal 10 Hz 在 `rtol=1e-6` 下通过，20 Hz 拒绝。

## 来源链：历史生成记录与本次核对

生成过程依据 本地历史快照 `docs/records/2026-10-18-validation.md` 的记录：原生 FYST scheduler /
trajectory generator → 旧本地 example 的 `fyst_trajectory.ecsv` 与 `schedule.ecsv`
→ 本 repo 的 `scripts/save_trajectory_h5.py` → `output/<case>/hdf5/` → 本目录副本。

历史输入产生于升级前的 **FYST trajectories 0.8.0 working tree + 本地 UTC patch**，
base commit 为 `2d0a3d395486fd4a43d83c8b38138e9b7bb6159b`；不是未修改的 release，
**本次没有用 v0.10 重新生成**。`_iso_utc` / `_read_time` 的本地补丁支持带 `+00:00`
的时间文本。补丁和来源文件 hash 见 本地 `docs/UPSTREAM.json` 与
历史 UTC patch。当前依赖 FYST 0.11.0
已内置修复，不改变这些 fixtures 的历史来源。writer 源自旧本地
working-tree example，并非 FYST upstream 已提交文件；原有 license / authors 记录
见 [第三方代码与许可](../README.md#第三方代码与许可) 和 [LICENSES](../LICENSES)。

验证记录中的 writer 命令如下（从本 repo 根目录执行；这是 **R4 记录的历史命令，R4 未运行**）：
迁移后输入位于 `../legacy_repo/fyst-trajectories-sandbox-2026-10/examples/fyst_to_toast/output/{ces,daisy}/`；R2 的复跑与旧输出比较见 本地学习笔记 `docs/steps/02-fyst-dependency-patch.md`。

```bash
python scripts/save_trajectory_h5.py \
  ../fyst-trajectories/examples/fyst_to_toast/output/ces/fyst_trajectory.ecsv \
  --iers-file ../fyst-trajectories/tests/data/finals2000A.all

python scripts/save_trajectory_h5.py \
  ../fyst-trajectories/examples/fyst_to_toast/output/daisy/fyst_trajectory.ecsv \
  --iers-file ../fyst-trajectories/tests/data/finals2000A.all
```

当时 `--schedule` 默认取各 trajectory 同目录的 `schedule.ecsv`；`--output-dir`
当时默认分别为本 repo 的 `output/ces/hdf5`、`output/daisy/hdf5`（2026-10-19 起移至 `output/legacy/`；writer 的默认输出现改为输入 ECSV 同目录的 `hdf5/`）；固定 `--roll-deg 0`。
写入版本为 **TOAST 3.0.5**，两份 H5 的根属性 `toast_version` 也实测为 `3.0.5`，
`toast_format_version` 为 `2`。

这些 ECSV 目前归档于 workspace 相对路径：

```text
legacy_repo/fyst-trajectories-sandbox-2026-10/examples/fyst_to_toast/output/ces/
legacy_repo/fyst-trajectories-sandbox-2026-10/examples/fyst_to_toast/output/daisy/
```

这里的 workspace 指本 repo 的父目录；从本 repo 出发在上述路径前加 `../`。
对应 IERS 文件位于同一归档树的 `tests/data/finals2000A.all`，新 clone 的
`../fyst-trajectories/tests/data/finals2000A.all` 与它逐字节相同，可用于当前复跑。这些位置只说明
历史输入现存位置，loader 测试不依赖 sibling checkout。

下列 SHA-256 原载于现归档的 `docs/records/2026-10-18-validation.md`；本次只读检查归档文件，全部吻合，
并与现有 `writer_report.json` 中的 source / schedule / IERS hashes 交叉核对：

```text
CES fyst_trajectory.ecsv
  a67ea4913709cf31454e9cbf279d4945e53c9dcf59bc818bc24fc6221faf50fd
CES schedule.ecsv
  e0cbc5f80bd974ccb6870fdbf8d97ede5009149438bddadf7ca5f548e2b6194a
Daisy fyst_trajectory.ecsv
  d4efb21502ec60aad144157367489299515b696f41df411eee5aae79c03faad0
Daisy schedule.ecsv
  fd61a338253cdbe9c778967e969c9f2f0fb0330b1a86e942ee37dffc8a493dd8
Explicit IERS-A finals2000A.all（两种 case 共用）
  df9b19b7a6055871346909beb8ba127c46d464b36c47d5cb02a02ad0e9a2be06
```

**绝对路径检查：**本次遍历两份 H5 的全部 attributes，以及字符串 / compound
datasets（包括 focalplane table metadata），未发现嵌入的绝对文件路径。
`trajectory_info.source_sha256` 保存 trajectory 的内容 hash，不保存输入路径；
schedule 与 IERS hashes 来自外部 writer 报告和验证记录，并未嵌入这两份 H5。

## 不能代表什么

每份文件只有一个 native block，不代表完整 schedule、块间空档或多 block 拼接。
原始数据均**没有 FYST flags `0/3`**；测试副本里合成的 `0/3` 只检查 enum 保留，
不构成真实 retune / 未分类状态的科学验证。FYST enum 与 TOAST shared bitmask
含义不同，不能直接互换。

文件中的 5 个 detectors 和固定 boresight roll 是诊断设置，未验收真实 FYST
focalplane / Nasmyth 姿态。没有 detector signal 或扫描 interval lists；不覆盖
Pong、不规则采样、MPI、sky / atmosphere / noise simulation。当前 `SimFYST`
仍为 `SimGround` 的 CES baseline，尚未接入 H5 loader。

本次测试结果与重跑方法见 本地 R4 学习笔记 `docs/steps/05-sample-inputs.md`；原始日志
在忽略目录 `output/codex/R4_sample_inputs/`，不作为 fixture 一同提交。
