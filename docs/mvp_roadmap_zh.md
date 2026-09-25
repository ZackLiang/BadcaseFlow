# 版本路线图

## 1. 首版目标

第一版不追求马上训练出最强模型，而是跑通一个可信、可追踪、可继续扩展的 Agent 训练飞轮：

```text
合成种子任务
  -> mock Agent rollout
  -> 规则 / judge 评测
  -> accepted / rejected 切分
  -> SFT 数据导出
  -> 训练 dry-run manifest
  -> badcase 报告
  -> 下一轮任务建议
```

它应该能在普通电脑上运行，不依赖 GPU、不依赖远程服务、不下载大模型。这样用户可以先理解协议和闭环。

## 2. v0.1.0: 首个可用版本

交付物：

- CLI：`bcf ingest`、`bcf rollout`、`bcf eval`、`bcf export`、`bcf analyze-cases`；
- 本地 run 目录：`runs/<run_id>/`；
- 数据协议：`TaskSample`、`AgentTrace`、`EvalResult`、`DatasetVersion`、`FlywheelRound`；
- mock agent：确定性工具调用和 final answer；
- mock tools：calendar/search/calculator/rag；
- rule evaluator：schema、tool success、answer contains evidence；
- SFT exporter：ShareGPT/OpenAI messages 两种格式；
- case analyzer：按错误类型聚类，生成下一轮任务建议。
- 训练 adapter：LLaMA-Factory、verl 和 command harness；
- 评测 adapter：builtin、command、promptfoo、OpenCompass 和 LightEval；
- registry：dataset、artifact、model、eval、promotion 和 deployment；
- deployment plan：vLLM、SGLang 和 OpenAI-compatible endpoint 的上线计划；
- doctor：首版示例、recipe、文档和 adapter 自检。
- job runner：把 launch plan 登记为任务，支持本地执行、SSH 计划、状态、日志和 collect。

成功标准：

- 一条命令跑完整链路；
- 每条样本可追踪来源、prompt/tool/reward 版本；
- accepted/rejected/SFT/case report 都能落盘；
- 所有示例数据都是 synthetic。

## 3. v0.2: Web Workbench

交付物：

- Dashboard：飞轮轮次、模型候选、关键指标；
- Datasets：样本浏览、过滤、版本对比；
- Evaluation：样本级结果、指标趋势、失败分桶；
- Training：recipe、日志、指标、artifact；
- Case Board：badcase 聚类、归因、下一步动作；
- Model Registry：checkpoint、adapter、eval report、部署状态。

成功标准：

- 用户能从一个失败 case 跳到原始 trace、评测原因、训练数据投影；
- 用户能把一批 badcase 标记为下一轮 SFT/DPO/RL 数据候选；
- 所有训练和上线动作都有确认入口。

## 4. v0.3: 执行编排

交付物：

- launcher plan 升级为可恢复任务；
- job runner 升级为队列化任务；
- 支持本地、SSH、Ray、Kubernetes、Slurm adapter；
- 失败重试、超时控制、资源队列、日志流和运行状态恢复；
- 训练/评测 run 与 Web 工作台联动。

成功标准：

- 用户能提交训练或评测任务，并在失败后继续追踪和恢复；
- 所有任务的输入、输出、日志、指标和状态都能追溯；
- 同一套 recipe 可以在不同执行环境中复用。

## 5. v0.4: Online Agent Loop

交付物：

- vLLM/SGLang OpenAI-compatible serving adapter；
- deployment plan：服务启动命令、canary、健康检查和回滚 runbook；
- trace collector；
- canary evaluation；
- A/B 或 shadow run；
- human feedback import；
- production trace sampling policy。

成功标准：

- 新模型必须通过冻结 eval suite 才能候选上线；
- canary 期间线上 trace 自动进入评测和 case board；
- 支持快速回滚；
- 线上样本进入训练前必须完成数据治理和质量门禁。

## 6. 推荐技术栈

后端：

- Python 3.11+；
- FastAPI；
- SQLModel 或 SQLAlchemy；
- SQLite for dev，PostgreSQL for production；
- local filesystem for dev，S3/MinIO for production。

前端：

- React + Vite 或 Next.js；
- TanStack Query；
- 表格和图表优先，避免做成营销站；
- 训练指标可先链接 TensorBoard/MLflow，后续再内嵌。

任务执行：

- v0.1.0 使用本地 harness 和 launcher plan；
- v0.2 先把运行状态接入 Web 工作台；
- v0.3+ 支持 Ray/Kubernetes/Slurm/SSH adapter。

## 7. 首版后的开发拆分

1. 做交互式 Web 工作台：筛选、审批、编辑、任务分派和多人协作。
2. 强化 case analysis：embedding 聚类、相似 case 合并、judge 复核、行动优先级。
3. 升级 launcher：把 plan 变成可恢复任务，支持日志流、失败重试和超时控制。
4. 接 trace 回流：导入线上 trace、canary eval、部署后复盘和回滚建议。
5. 补数据质量门禁：去重、覆盖率、污染检查、评测集冻结和数据准入。
6. 完善示例项目：给出从 trace 到训练、回评测、部署计划的端到端样例。

## 8. 待决策事项

- Web 工作台使用 React/Vite 还是 Next.js；
- 任务执行层优先支持 Ray、Kubernetes、Slurm 还是 SSH；
- registry 后续是否从 JSON 文件升级到 SQLite/PostgreSQL；
- trace adapter 优先接 Langfuse、Phoenix 还是 OpenTelemetry；
- judge/eval 优先支持本地模型、OpenAI-compatible endpoint 还是两者都保留。
