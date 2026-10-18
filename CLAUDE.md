# Claude 接手入口

请先读 [AGENTS.md](AGENTS.md)，它是本 repo 共同的开发约定。
接着读 [README.md](README.md)、[架构说明](docs/architecture.md)、
[交接记录](docs/HANDOFF.md) 和 [验证记录](docs/validation.md)，并检查当前 Git diff。

预期协作方式是：Claude 帮用户规划、解释、拆分小任务并 review；Codex 在获得的
任务范围内实现和验证，返回 diff、执行命令与证据。这里没有配置自动跨模型调用，
也不假定双方能看到对方的会话；可使用 handoff 中的任务与回报模板手动传递。

当前特别需要记住：`SimFYST` 仍是 CES baseline；独立 example 复用了已有的
TOAST HDF5 writer，不代表 operator 的 trajectory 读取已实现。先保留可运行对照，
再逐步接入真实 trajectory；优先复用 repo 现有实现。
