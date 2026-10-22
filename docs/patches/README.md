# FYST UTC 时间修复：历史补丁

TOAST 的 `datetime.fromisoformat(...).timestamp()` 会把没有 offset 的 naive datetime
按主机本地时区解释。FYST **v0.11.0**（`c933b7a`）已内置修复（upstream commit `07a36ac`）：
schedule 时间写成毫秒加 `+00:00` 的 UTC 文本，reader 兼容旧文本及带 offset 的输入。
fyst-to-toast 现依赖 `fyst-trajectories>=0.11,<0.12`，**不再需要任何补丁**。

下面两个补丁只用于追溯旧输出的来源，**不要应用到 0.11**：

| 用途 | 补丁 | base commit / 版本 |
|---|---|---|
| 0.10 过渡期本机依赖（2026-10-18 至 10-22） | [fyst-trajectories-utc-offset_f095c51.patch](fyst-trajectories-utc-offset_f095c51.patch) | `f095c5176ab5f8b53aa19a549dc871ef2e91feeb` / 0.10.0 |
| 历史 fixture 来源 | [fyst-timeline-utc_2d0a3d3.patch](fyst-timeline-utc_2d0a3d3.patch) | `2d0a3d395486fd4a43d83c8b38138e9b7bb6159b` / 0.8.0 |

SHA-256（补丁文件本身）：

```text
0.10  38b0926aac5f4ff49bd6b7043bb33a5ff8448f660ec3760a29dc896c4fa93eec
0.8   2e5e295a60239173bd1fa0dcb1366b26e038b7e6abde73f609e3805925b18c63
```

0.10 补丁涉及 `src/fyst_trajectories/overhead/io.py`、`tests/overhead/test_io.py`、
新增 `tests/overhead/test_io_timestamps.py`、`docs/overhead_io.rst`，时间写到微秒；
0.8 补丁只涉及 `src/fyst_trajectories/overhead/io.py`，用于追溯 `data/` fixtures。
两个补丁针对不同 base，不要叠加，也不要重复应用。

0.11 与 0.10 补丁版的生成结果不逐位相同：schedule 起止时间只保存到毫秒，Daisy / Pong
以读回的 block 起点重建轨迹，时间可平移 <0.5 ms；半开区间末端的样本数也可能差 1
（例：默认 3 h Pong 74402 → 74401）。三个默认 case 中 CES 完全不变。

## 新机器重建依赖

在 workspace 根目录执行；已有 `fyst-to-toast/`，目标 `fyst-trajectories/` 必须尚不存在：

```bash
set -e
git clone https://github.com/ccatobs/fyst-trajectories.git fyst-trajectories
git -C fyst-trajectories checkout --detach v0.11.0
```

在准备好运行依赖和 FYST build-system 依赖（包括 hatchling）的目标环境中做 editable
install，使 Python 导入这个 checkout，而不是另一份 site-packages 副本：

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
assert fyst.__version__ == "0.11.0"
assert digest == "07597f7ef0a642f47ee1a6053ad6075a5593d6f349aec68bc3011635ff6f146f"
PY
```

当前机器没有重新安装：env `fyst` 的静态 `.pth` 直接指向 `fyst-trajectories/src/`，
该 checkout 已切到 tag `v0.11.0`（detached）。pip metadata 仍为 0.8.0，实际
`__version__` 为 0.11.0；应以上述导入路径和文件 hash 为准。
