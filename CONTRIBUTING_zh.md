# 贡献指南

感谢你关注 BadcaseFlow。项目目前优先保证数据协议、训练适配器、评测门禁和飞轮记录的稳定性，欢迎围绕这些方向提交改进。

## 开发环境

项目需要 Python 3.11 或更高版本：

```bash
python -m pip install -e .
python -m badcaseflow version
python -m badcaseflow doctor --strict
```

如果要测试 GSM8K 的 verl parquet 数据：

```bash
python -m pip install -e ".[parquet]"
```

## 提交代码前

在仓库根目录执行：

```bash
python -m compileall src tests
python -m unittest discover -s tests
python -m badcaseflow doctor --root . --strict
git diff --check
```

## 代码约定

- 新增功能优先复用现有 adapter、recipe、run store 和 registry 结构。
- 对外暴露的 CLI、数据字段和状态值需要补充中文文档。
- 复杂的数据转换、门禁判断、命令生成和状态迁移要写简短中文注释。
- 代码默认使用 ASCII；只有面向用户的文档、命令输出或必要的中文注释使用中文。
- 不要把本地运行结果写进仓库，`runs/`、`artifacts/`、`.badcaseflow/` 已加入忽略规则。
- 新增数据格式时，同时补 schema 校验、失败路径测试和一个最小示例。
- 新增训练或评测后端时，同时补 adapter 列表、recipe 示例、dry-run 测试和文档。

## 测试原则

- 单元测试必须可以离线运行。
- 不要求测试环境安装 GPU、verl、LLaMA-Factory 或其他外部训练框架。
- 真实训练命令使用 dry-run、command harness 或 mock adapter 验证。
- 涉及文件、registry、任务状态的测试使用临时目录。

## 提交说明

提交信息建议说明行为变化，例如：

```text
feat: 增加 GSM8K 数据准备命令
fix: 修复训练 run 指标解析
docs: 补充 AutoDL 快速开始
test: 增加 registry 回评测测试
```

## Pull Request 建议

PR 描述中请说明：

- 解决了什么问题；
- 改动了哪些命令、数据字段或状态；
- 是否兼容已有 recipe；
- 执行了哪些测试；
- 是否需要安装额外依赖。
