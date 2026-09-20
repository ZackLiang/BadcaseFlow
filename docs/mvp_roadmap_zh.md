# MVP 路线图

## 1. MVP 目标

第一版 MVP 不追求马上训练出最强模型，而是跑通一个可信的最小飞轮：

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

## 2. v0.1: Local Flywheel

交付物：

- CLI：`bcf ingest`、`bcf rollout`、`bcf eval`、`bcf export`、`bcf analyze-cases`；
- 本地 run 目录：`runs/<run_id>/`；
- 数据协议：`TaskSample`、`AgentTrace`、`EvalResult`、`DatasetVersion`、`FlywheelRound`；
- mock agent：确定性工具调用和 final answer；
- mock tools：calendar/search/calculator/rag；
- rule evaluator：schema、tool success、answer contains evidence；
- SFT exporter：ShareGPT/OpenAI messages 两种格式；
- case analyzer：按错误类型聚类，生成下一轮任务建议。

成功标准：

- 一条命令跑完整链路；
- 每条样本可追踪来源、prompt/tool/reward 版本；
- accepted/rejected/SFT/case report 都能落盘；
- 所有示例数据都是 synthetic。

## 3. v0.2: Training Adapters

交付物：

- LLaMA-Factory adapter：生成 SFT/DPO 配置，支持 dry-run 和真实启动；
- verl adapter：生成 GRPO/OPD 配置，支持 dry-run 和真实启动模板；
- preflight：检查 JSONL schema、token 长度、base model、输出路径；
- metrics adapter：TensorBoard/MLflow 日志目录登记；
- artifact registry：adapter、merged model、logs、eval report 统一登记。

成功标准：

- 用户可以先 dry-run 查看命令和资源计划；
- 训练命令由 launcher adapter 生成，支持按环境替换；
- 训练完成后能自动触发 eval run；
- train run 页面或 manifest 能看到 loss、reward、KL、throughput 等指标入口。

## 4. v0.3: Web Workbench

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
- 所有训练和上线动作都有人工确认入口。

## 5. v0.4: Online Agent Loop

交付物：

- vLLM/SGLang OpenAI-compatible serving adapter；
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

- v0.1 使用本地 worker；
- v0.2 抽象 launcher；
- v0.3+ 支持 Ray/Kubernetes/Slurm/SSH adapter。

## 7. 第一周任务拆分

1. 定义 Python package 和 CLI 框架。
2. 实现本地 run store：保存 JSONL、manifest、summary。
3. 实现 synthetic `TaskSample` ingest。
4. 实现 mock tool registry 和 mock agent rollout。
5. 实现 rule evaluator。
6. 实现 SFT exporter。
7. 实现 case analyzer 的最小错误分类。
8. 写端到端测试，确保 `ingest -> rollout -> eval -> export -> analyze` 可重复运行。

## 8. 需要尽早定下来的决策

- 项目名使用 `BadcaseFlow`，CLI 暂定 `bcf`；
- License：建议 Apache-2.0；
- 第一版 UI 用 React/Vite 还是先用 Streamlit；
- 训练后端优先接 LLaMA-Factory 还是 verl；
- 是否内置 Langfuse/Phoenix adapter，还是先只导出 OpenTelemetry/JSONL。
