# R3：把自己的终端 data viewers 带入 repo

## 为什么要改、目的

能运行 writer / loader，还需要能亲自查看它们之间的数据。这里迁移用户已有的
ECSV / HDF5 viewer，保留彩色表格、原按键、完整值和 metadata 查看方式，
方便从 schedule → trajectory → TOAST H5 逐层核对；浏览结果不替代科学验收。
依据是研究记录中 2026-09-25 / 09-28 的终端可读性与屏幕恢复要求。

## 改了什么

来源是 legacy archive 的 `examples/fyst_to_toast/` 本地工具：847 行的
`ecsv_reader.py`（2026-09-29 新版）、`h5reader.py` 和 `_support/hdf5_table.py`。
没有采用旧 repo 根目录的 841 行版本，也没有修改 archive。

- `tools/ecsv_reader.py` 与 `tools/_hdf5_table.py` 保持来源文件逐字节相同。
- [tools/h5reader.py:17](../../tools/h5reader.py#L17)：将
  `from _support.hdf5_table ...` 改为 `from _hdf5_table ...`。
  [tools/h5reader.py:19](../../tools/h5reader.py#L19) 保留同目录的 `ecsv_reader` import；
  删除原第 9 行的 `# ruff: noqa: E402`。直接执行脚本时 Python 自然把 `tools/`
  放在 `sys.path[0]`，无需路径注入或其他 checkout。
- `git mv tests/check_writer_roundtrip.py tools/check_writer_roundtrip.py`；代码逐字节不变，
  参数路径本来就按 cwd 解析，无内部路径需要修正。README 与 `docs/*.md` 只替换旧路径。
- 新增 [tools/README.md](../../tools/README.md)，汇总命令与按键。

源码观察：H5 用 `h5py.File(..., "r")` 打开；ECSV 只读表，原有 `--html` 仅用于另存预览。
两个 viewer 不 import TOAST。单独的 roundtrip 验收脚本按原用途调用原生 `LoadHDF5`。

## 怎么用它们核验中间数据

以下在 repo 根目录、已有 `fyst` 环境运行。`CES_INPUT` 指向本机旧数据，仅是输入示例，
可换成自己的 ECSV 目录；工具代码不依赖 legacy repo。

```bash
export MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CES_INPUT=../legacy_repo/fyst-trajectories-sandbox-2026-10/examples/fyst_to_toast/output/ces
```

1. 检查 schedule 的 UTC 字符串，而不是只看缩略表格：

   ```bash
   python tools/ecsv_reader.py "$CES_INPUT/schedule.ecsv" --view text --pager never \
     --columns start_time,stop_time --all --no-metadata --no-schema --full
   ```

   本次实查三个 schedule rows 的起止时间全部带 `+00:00`，例如
   `2026-08-02T01:42:01.727932+00:00`。viewer 保留字符串，不做时区转换。

2. 找 CES H5 的 turnaround（行号从 0 起）：

   ```bash
   python tools/h5reader.py data/fyst_ces_000_001.h5 --view text \
     --node /shared/fyst_scan_flag --start 475 --rows 5
   python tools/h5reader.py data/fyst_ces_000_001.h5 --view text \
     --start 475 --rows 5 --columns time_s,fyst_flag,toast_flags --full
   ```

   本次第 477 行首次出现 FYST enum `2`；前后五行是 `1,1,2,2,2`，
   对应 TOAST flags `0,0,4,4,4`。全文件 2,611 点中有 80 个 turnaround samples。
   交互时打开 `--node /shared/fyst_scan_flag`，用 `/` 搜索 `2`、`n` 找下一个，
   Enter 查看完整值，Tab 回文件树。FYST 是 enum；一般 TOAST flags 应按 bitmask 判断。

3. 检查 trajectory 样本、单位与精度：

   ```bash
   python tools/ecsv_reader.py "$CES_INPUT/fyst_trajectory.ecsv" --view text --pager never \
     --start 675 --rows 10 --columns time_s,az_deg,el_deg,scan_flag --full
   ```

   去掉文本选项可进入全屏；`u` 看列单位、`m` 看 metadata、Enter 看完整值。
   H5 默认 trajectory 视图的 Az/El 为 degree，`--node /shared/azimuth` 则显示原始 rad，
   不要把界面转换误认为写回文件。

## 怎么验证的（2026-10-18 实际执行）

使用指定 `fyst/bin/python`，关闭 MPI、限制 NumPy 线程，并禁写 Python bytecode。
原始日志、完整命令及来源 hash 在
[`output/codex/R3_data_readers/`](../../output/codex/R3_data_readers/)（Git ignored）。

- 两份 ECSV、指定 CES / Daisy H5：19 种非交互案例分别从 repo 内外运行，
  共 38 组 legacy / ported stdout、stderr 逐字节相同，diff 全空，无路径差异。
- 120×32 PTY：四份输入各跑新旧版本，共 8 个会话，实际移动、Enter、Tab、`m`、`q`；
  均退出 0，四组完整 ANSI 屏幕记录逐字节相同。alternate screen 和光标恢复，
  termios 除 macOS 临时 PENDIN 位外相同。逐阶段 `.ansi` 与可回放 `.cast` 存于 `pty/`。
- 首轮 PTY 断言误要求增量输出重发完整 `HDF5 tree` 标题；两版均仅重画 `tree`。
  修正验证脚本后通过，未改 viewer；首轮日志保留为 `initial_*`。
- CES roundtrip 在 repo 内外各通过一次：3 blocks、5,921 samples、每块 17 shared arrays，
  12 个输入对应数组逐点相同。viewer 的实际 import trace 没有 TOAST、MPI 或 FYST imports。
- 11 个受保护文件 SHA-256 前后相同：两份 ECSV、4 份 H5、`files.json`、3 份来源代码及
  legacy `_support/__init__.py`。ECSV 的同路径 HTML 输出也实际被拒绝，输入未变。
- reader 按来源的 Ruff 规则（E/F/W/I/UP/D、NumPy docstring、100 列）check / format 通过；
  checker 按当前默认配置通过。用显式 `known-first-party` 标记两个同目录模块。
  未运行 repo-wide formatter。首次全局 Ruff 的 7 项旧代码风格 / import 分组诊断保留在
  `ruff_global_initial.log`，本次不为这些额外规则改写原实现；复跑参数见 `ruff.json`。

## 还没做什么

没有重写 UI、修改科学数组、安装依赖、运行模拟或接入 SimFYST，也未 commit。
本轮只做上述输入与固定窗口的 smoke test；没有重跑旧记录中的 resize、Ctrl+C / SIGTERM、
超大文件、detector 数据与 HTML 浏览器验收。ECSV 仍整表读取；H5 的现有缓存策略未改。
