# Project instructions

## large-scale-evidence-extraction 的维护归属

- 用户指定本项目 `llm-evidence-triangulation` 为该 skill 的唯一维护项目，源目录为 `skills/large-scale-evidence-extraction/`。
- 其他项目只能读取和调用已发布版本；不得修改提示词、schema、脚本、测试、说明、全局安装副本或 GitHub 源码，也不得另建修改版候选副本、安装覆盖、回滚或发布版本。
- 其他项目可在 skill 目录之外配置研究问题、摘要输入、术语索引、运行参数和输出目录。调用 Python 时使用 `-B` 或 `PYTHONDONTWRITEBYTECODE=1`，避免在 skill 内产生缓存。
- 其他项目发现问题时只报告现象及修改建议，回到本项目统一维护。通过切换 shell 工作目录、克隆仓库或子代理执行不能绕过维护归属。
- 本项目获得相应维护任务后，先修改源目录、验证，再同步全局安装副本和 GitHub。只有用户明确要求改变本条维护归属时，才可调整该规则。
- 本节仅约束该 skill；不扩大其他项目文件的访问或修改权限。
