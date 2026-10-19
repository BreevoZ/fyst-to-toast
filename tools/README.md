# 手动检查中间数据

在已有 `fyst` 环境、repo 根目录运行：

```bash
export MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/ecsv_reader.py /path/to/schedule.ecsv
python tools/ecsv_reader.py /path/to/fyst_trajectory.ecsv
python tools/h5reader.py data/fyst_ces_000_001.h5
```

`ecsv_reader.py` 浏览 ECSV 的原始列、metadata、单位和完整 JSON。
`h5reader.py` 浏览 HDF5 表格、文件树和 attributes，复用同一套界面；
`_hdf5_table.py` 是它的内部 adapter（每列缓存 256 行，无单独 CLI）。
默认 H5 trajectory 视图把 Az/El 显示为 degree；打开原始 dataset 时仍显示 rad。
两个 viewer 只读输入，不 import TOAST；使用现有 NumPy、Astropy、h5py、PyYAML。
ECSV 原有 `--html preview.html` 可另存预览，请使用独立输出文件。

| 按键 | 用途 |
|---|---|
| 方向键 / `hjkl` | 移动单元格；`Space` / `b` 或 PgDn / PgUp 翻页 |
| `[` / `]`、`H` / `G`、`g` | 横向翻页、首末行、跳到指定行（0-based） |
| `/`、`n` | 搜索当前列完整值、下一个匹配 |
| Enter、`m`、`u`、`?` | 完整单元格 / JSON、metadata、列类型与单位、帮助 |
| Tab | ECSV 下一列；H5 表格返回文件树，树中回到 trajectory |
| H5 树的 ← / →、Enter、`m`、`d` | 收起 / 展开、打开 dataset、attributes、选择 detector index |
| `q` / Esc | 关闭弹窗；在表格或文件树退出并恢复 terminal（Ctrl+C 也可退出） |

用于保存日志的非交互命令（终端中也不会进入全屏界面）：

```bash
python tools/ecsv_reader.py /path/to/schedule.ecsv --view text --pager never \
  --rows 5 --color never
python tools/ecsv_reader.py /path/to/fyst_trajectory.ecsv --view text --pager never \
  --start 675 --rows 10 --columns time_s,az_deg,el_deg,scan_flag --full
python tools/ecsv_reader.py /path/to/schedule.ecsv --view text --pager never \
  --row 0 --details --full
python tools/h5reader.py data/fyst_ces_000_001.h5 --view text --tree
python tools/h5reader.py data/fyst_ces_000_001.h5 --view text \
  --node /shared/fyst_scan_flag --start 475 --rows 5
```

ECSV 可加 `--no-metadata --no-schema` 缩短输出；`--full` 保留数值精度，
`--columns` 选列，`--help` 查看全部选项。默认数值为 6 位有效数字，仅影响显示。

`check_writer_roundtrip.py` 是原生 TOAST 的只读 CLI 验收脚本（会 import TOAST），
逐点比较 writer 的 H5 与输入 trajectory，结果 JSON 输出到 stdout：

```bash
python tools/check_writer_roundtrip.py /path/to/fyst_trajectory.ecsv output/ces/hdf5
```

从其他 cwd 运行时，给脚本和输入传绝对路径即可，无需修改 `sys.path`。
本次迁移来源、具体核验例子与结果见 [R3 学习笔记](../docs/steps/04-data-readers.md)。
