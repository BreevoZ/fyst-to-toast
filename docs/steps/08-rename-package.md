# N1：统一 package 名称

2026-10-18；历史名称：distribution `fyst-toast`，import 与本机 repo 目录 `fyst_toast`（仅此处保留作迁移说明）。
起点为干净的 `main@9470535`；以下验证已在现有 `fyst` 环境实际执行。

## 为什么要改

远程 repo 已叫 `fyst-to-toast`。让 repo、安装名称和 Python import 使用同一组词，
避免漏写中间的 `to`；连字符与下划线按各自语法选择。

## 改了什么

- 用 `git mv` 将源码包改为 `src/fyst_to_toast/`，同步代码、测试、命令和文档中的名称。
- distribution 改为 `fyst-to-toast`；version、dependencies、extras、license metadata 不变。
  setuptools 仍从 `src/` 自动发现 package，wheel 验证确认只包含新包。
- 临时缓存目录改为 `fyst-to-toast`；测试变量统一为 `FYST_TO_TOAST_TEST_OUTPUT`、
  `FYST_TO_TOAST_TEST_IERS`、`FYST_TO_TOAST_TEST_SCHEDULE_ROOT`。
- 文档中的 repo 目录及补丁重建命令使用未来目录 `fyst-to-toast/`；历史结果和来源 hash 保留。
  新增的长环境变量表达式仅作换行；12 个 Python 文件的 AST 核对确认无逻辑变化。
- 清除已确认未跟踪且被忽略的旧 egg-info；旧 `build/` 保存在本轮日志目录的
  `build_before/`，防止它混入新 wheel。补丁内容和样例 H5 未改。

## 目的

以后从 remote clone 得到 `fyst-to-toast/`，安装的是 `fyst-to-toast`，代码写
`from fyst_to_toast.sim_fyst import SimFYST`，三处含义清楚且名称一致。

## 关键概念

| 名称 | 本项目写法 | 用途 |
|---|---|---|
| Repo / directory name | `fyst-to-toast` | Git remote 的项目名和约定的 clone 目录；本机目录将在下一步改名 |
| Distribution name | `fyst-to-toast` | `pyproject.toml` 的安装、构建 metadata 名称 |
| Import package name | `fyst_to_toast` | Python 在 `src/` 下找到的包，写在 `import` 语句中 |

Python 普通 `import` 语句要求名称是 identifier，`-` 是减号，不能放进 identifier，
所以使用 `_`。wheel 文件名也将 distribution 的连字符规范化为下划线。

本机 editable install 的静态 `.pth` 保存绝对路径 `<repo>/src`，Python 启动时将它
加入搜索路径。因此只改 `src/` 内的 package 目录，新 import 就能找到代码；它没有给
旧包名保留 alias。**整个 repo 目录改名后，旧 `.pth` 路径会失效**，需重新 editable
install，并更新 distribution metadata；这些由 orchestrator 下一步完成，本轮未安装。

这一步只改寻址和名称，没有改变 scheduler、采样、block 边界、数组或 writer schema。
相同输入、参数、依赖和 IERS 表下，科学文件应保持 byte-identical；这里另用实际
生成结果逐字节比较验证，不能仅凭“改名不改算法”作结论。

## 怎么验证的

完整命令及日志在 [`output/codex/N1_rename/`](../../output/codex/N1_rename/)；
执行清单见 [verification.json](../../output/codex/N1_rename/verification.json)，
字节、metadata、wheel 和源码核对见 [artifact_audit.json](../../output/codex/N1_rename/artifact_audit.json)。
所有数值进程导入前设 `MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`，
writer / generator 使用 `../fyst-trajectories/tests/data/finals2000A.all`，禁止 IERS 下载。

| 检查 | 本轮结果 |
|---|---|
| 正常 `pytest tests/ -ra` | **52 passed，0 skipped**；一次缓存目录尚未创建的 Astropy warning，日志保留 |
| `PYTHONPATH` 中仅增加 FYST import blocker | **45 passed，2 skipped**；仅 `test_fyst_contract.py`、`test_generate.py` 整模块跳过 |
| `examples/01_ces_baseline.py` | 600 samples、2 detectors，8 个 shared arrays、13 组 intervals 和 detdata 与 SimGround 一致；报告与旧 baseline 逐字节相同 |
| repo 外导入新包及 `trajectory_h5` | 成功，`sys.modules` 中无 `toast` / `fyst_trajectories`；旧包名导入以 `ModuleNotFoundError` 失败 |
| legacy CES → writer | 3 个 H5 与 `output/ces/hdf5/` 整文件相同；两个 JSON 也相同 |
| 默认 10800 s CES generation | 3 个 ECSV 与 `output/generated/ces_10800_w2_b/` 相同，`report.json` 也相同；3 blocks、87,983 samples |
| 离线 wheel | 下方命令成功；包含 5 个新包源码文件和全部 4 个 license / authors / NOTICE 文件，metadata 保留 |
| 输入及原产物保护 | 17 个文件的 SHA-256 未变，包括两份 `.patch` 和两份 `data/*.h5` |
| Ruff check / format-check | 全范围未通过：40 条 lint（example 2、SimFYST 38）及 3 个文件的格式问题均在 HEAD 已存在；本次未新增问题，未扩大范围修复 |
| `git diff HEAD --check` | 通过 |

```bash
python -m pip wheel --no-deps --no-build-isolation --no-index --no-cache-dir . \
  --wheel-dir output/codex/N1_rename/wheel
```

产物为 `fyst_to_toast-0.1.0.dev0-py3-none-any.whl`。Ruff 的 HEAD 对照在
[ruff_head.json](../../output/codex/N1_rename/ruff_head.json) / [ruff_current.json](../../output/codex/N1_rename/ruff_current.json)；
既有格式问题涉及 example、package `__init__.py`、`sim_fyst.py`。
旧名称搜索在受跟踪文件中无命中；把本篇尚未跟踪的新笔记纳入搜索后，仅命中开头的历史名称说明。

## 还没做什么

未移动 repo 目录、重新安装、commit、push 或修改 remote；没有网络请求或 IERS 下载。
没有改动外部 FYST / legacy / TOAST repo，也没有实现 SimFYST 的 H5 分支、MPI 或新的科学功能。
目录改名与 editable reinstall 留给 orchestrator，随后需再从 repo 外核对 import 路径。
