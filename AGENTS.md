# fyst_toast 开发约定

这个独立 package 连接 FYST trajectory 与 TOAST Observation。默认用中文讨论，
保留英文 API 和科学术语。先读 `README.md`、`docs/architecture.md`、
`docs/HANDOFF.md`，再核对当前代码和 `docs/validation.md`；交接记录有日期，
不能替代当前执行结果。

## 工作边界

- 修改前检查本 repo 的 `git status` 和相关 diff，保留用户已有改动。
  相邻的 FYST、TOAST 和模拟项目是独立 repo；跨 repo 修改须属于当前任务范围。
- 当前 `SimFYST` 仍是 `SimGround` 的 CES baseline，没有 trajectory reader。
  迁移的 writer 是独立 example，保存 TOAST `SaveHDF5` schema；不能把它的完成
  描述为 `SimFYST` 已支持 HDF5。
- 用户希望研究并复用原 repo 实现。优先复用 native scheduler、trajectory generator、
  TOAST classes 和 IO；不要为了 demo 重造 scheduler、另切 block 或悄悄重采样。
- 路径通过参数或文件自身位置解析；不得硬编码用户名、绝对 workspace 路径，
  或要求某个 sibling checkout 恰好存在。不要用 `sys.path` 注入替代安装依赖。
- 生成数据、H5、图和执行报告默认写入已忽略的 `output/`。小型 fixture 若需要提交，
  应有明确用途、来源及体积说明，不把用户研究数据顺手加入 Git。

## 数值与运行约定

- 保留输入 timestamps、原始 FYST flags 和 native block 边界；明确时间及角度单位。
  FYST flag 是 enum，TOAST shared flags 是 bitmask，不能直接复制数值当作同一含义。
- 当前 writer 的五个示意 detectors 和固定 boresight roll 是诊断设置，不能描述为
  已验收的 FYST focalplane / Nasmyth 姿态。参考 `docs/HANDOFF.md` 的未决问题。
- 在本地串行 example 入口、导入 NumPy / TOAST 之前设置 `MPI_DISABLE=1`、
  `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`。本机 MPI 曾造成 idle CPU 占用；
  不依赖导入后设置环境变量关闭已初始化的 MPI。
- package import 不修改进程环境，不自动开启 MPI、运行模拟或下载天文数据。
  MPI 验证应是明确安排的独立工作；串行结果不能证明 MPI 已通过。

## 任务与验证

- 每次完成一个可解释的小步骤。动手前说清本次输入、输出、改动边界和验收条件。
  用户已授权的可逆工作直接推进，不重复请求确认。
- 改动后运行相关且有意义的检查。baseline 的用途是验证复制行为；未来 trajectory
  reader 的用途是保留输入轨迹，不能要求它的运动重新匹配 CES generator。
- 汇报区分源码观察、设计目标和实际执行结果。给出修改文件、运行命令、结果及
  已知限制；失败或未运行的检查如实标注，不把旧报告当成本次结果。
- 改行为时同步 README / architecture / handoff 中相关说明；更新来源记录时保留
  upstream license、authors、commit / 文件 hash 与 local working-tree 来源的区别。

Claude / Codex 的协作方式见 `docs/HANDOFF.md`。本文件是共享开发约定，
`CLAUDE.md` 只提供入口，不维护另一套互相矛盾的规则。
