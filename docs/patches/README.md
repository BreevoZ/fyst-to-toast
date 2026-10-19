# FYST UTC 时间兼容补丁

TOAST 的 `datetime.fromisoformat(...).timestamp()` 会把没有 offset 的 naive datetime
按主机本地时区解释。当前补丁让 FYST 输出带 `+00:00` 的微秒 UTC 文本，并使 reader
兼容旧文本及带 offset 的输入。版本号无法表达这项未提交修改；FYST repo 不属于用户，
不得在那里 commit，因此 `fyst-to-toast` 保存可版本管理的补丁。

| 用途 | 补丁 | base commit / 版本 |
|---|---|---|
| **当前依赖** | [fyst-trajectories-utc-offset_f095c51.patch](fyst-trajectories-utc-offset_f095c51.patch) | `f095c5176ab5f8b53aa19a549dc871ef2e91feeb` / 0.10.0 |
| **历史 fixture 来源** | [fyst-timeline-utc_2d0a3d3.patch](fyst-timeline-utc_2d0a3d3.patch) | `2d0a3d395486fd4a43d83c8b38138e9b7bb6159b` / 0.8.0 |

SHA-256（补丁文件本身）：

```text
当前  38b0926aac5f4ff49bd6b7043bb33a5ff8448f660ec3760a29dc896c4fa93eec
历史  2e5e295a60239173bd1fa0dcb1366b26e038b7e6abde73f609e3805925b18c63
```

当前补丁涉及 `src/fyst_trajectories/overhead/io.py`、`tests/overhead/test_io.py`、
新增 `tests/overhead/test_io_timestamps.py`、`docs/overhead_io.rst`；历史补丁只涉及
`src/fyst_trajectories/overhead/io.py`。旧补丁仅改名，内容不变，用于追溯 `data/` fixtures。
当前补丁是 workspace `research_notes/fyst_trajectories_utc_fix_2026-10-18/change.patch`
的逐字节副本。两个补丁针对不同 base，**不要叠加，也不要重复应用（do not apply twice）**。

## 新机器重建当前依赖

以下是新机器操作说明，本轮未执行网络或安装命令。在 workspace 根目录执行；
已有 `fyst-to-toast/`，目标 `fyst-trajectories/` 必须尚不存在，不要对正在使用的 checkout 操作：

```bash
set -e
git clone https://github.com/ccatobs/fyst-trajectories.git fyst-trajectories
git -C fyst-trajectories checkout f095c5176ab5f8b53aa19a549dc871ef2e91feeb
git -C fyst-trajectories apply --check ../fyst-to-toast/docs/patches/fyst-trajectories-utc-offset_f095c51.patch
git -C fyst-trajectories apply ../fyst-to-toast/docs/patches/fyst-trajectories-utc-offset_f095c51.patch
```

在准备好运行依赖和 FYST build-system 依赖（包括 hatchling）的目标环境中做 editable
install，使 Python 导入这个 patched working tree，而不是另一份 site-packages 副本：

```bash
conda activate fyst
export MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pip install --no-deps --no-build-isolation --no-index -e ./fyst-trajectories
python - <<'PY'
import hashlib
from pathlib import Path
import fyst_trajectories as fyst
import fyst_trajectories.overhead.io as io
expected = Path("fyst-trajectories/src/fyst_trajectories").resolve()
digest = hashlib.sha256(Path(io.__file__).read_bytes()).hexdigest()
print(fyst.__file__, fyst.__version__, digest, sep="\n")
assert Path(fyst.__file__).resolve() == expected / "__init__.py"
assert fyst.__version__ == "0.10.0"
assert digest == "32d9041d1af683579152f1ed7f7c4a86c92d110dc932c30cf4fbf356124458d3"
PY
```

当前机器没有重新安装：env `fyst` 的静态 `.pth` 直接指向新 checkout 的 `src/`，同样
读取 working tree。pip metadata 仍为 0.8.0，实际 `__version__` 为 0.10.0；应以上述
导入路径和文件 hash 为准。验证结果见 本地 R2 学习笔记 `docs/steps/02-fyst-dependency-patch.md`。
