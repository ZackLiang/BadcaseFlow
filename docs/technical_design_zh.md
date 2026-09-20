# BadcaseFlow 技术方案

## 1. 项目定位

BadcaseFlow 是一个面向 Agent 的受控自进化平台。它不是单一训练脚本，也不是单一评测工具，而是把数据生产、评测、SFT、偏好优化、RL/OPD、指标观测、case 分析、模型上线和线上反馈重新纳入同一个可追踪闭环。

目标用户是希望持续训练业务 Agent、工具调用 Agent、RAG Agent 或任务型小模型的团队。平台需要回答四个问题：

1. 训练数据从哪里来，是否干净、可回放、可追溯？
2. 训练是否真的提升了独立评测集，而不是只提升训练集？
3. 失败 case 为什么失败，下一轮应该补数据、改 prompt、改 reward，还是改工具/工作流？
4. 产出的模型如何经过门禁后上线，并把线上反馈带回下一轮迭代？

## 2. 核心闭环

```text
Data Sources
  production traces / seed tasks / documents / human labels / synthetic tasks
    |
    v
Data Factory
  ingest -> clean -> dedupe -> redact -> generate -> rollout -> trace store
    |
    v
Evaluation System
  rule checks -> reward model -> LLM judge -> human review -> eval report
    |
    +------ rejected / low-score cases ------+
    |                                       |
    v                                       v
Training Factory                     Case Analyzer
  SFT / DPO / GRPO / PPO / OPD        cluster -> attribute -> action proposal
    |                                       |
    v                                       |
Model Registry <------ iteration plan ------+
    |
    v
Serving + Monitoring
  vLLM/SGLang gateway -> canary -> online traces -> next data round
```

这个闭环里最重要的资产不是 checkpoint，而是带 lineage 的样本、轨迹、评测结果和训练 run。模型只是某一次 flywheel round 的产物。

## 3. 从本地参考项目抽象出的能力

已有项目里可以沉淀为平台通用能力的部分：

| 参考原型 | 可沉淀能力 | 平台抽象 |
|---|---|---|
| Agent data harness | seed 宽表清洗、Agent ReAct rollout、工具参数校验、rethink、reward gate、SFT 投影、自举 badcase | `DataFactory`, `TraceSchema`, `RewardGate`, `CaseEvolution` |
| Task pipeline | 任务协议、teacher annotation、QA/reward filter、SFT 导出、eval report、OPD/MOPD simulation | `TaskSpec`, `TeacherAdapter`, `EvalSuite`, `DistillRecipe` |
| Student distillation smoke | student SFT、teacher/student 资源池拆分、verl OPD、TensorBoard 日志、最小真实 smoke | `TrainingRecipe`, `ResourcePlan`, `MetricAdapter`, `SmokePlan` |
| Knowledge pipeline | 文档清洗、RAG 切片、检索评测、SFT/DPO 数据导出 | `KnowledgePipeline`, `RetrievalEval`, `PreferenceBuilder` |
| Controlled workflow agent | 受控 workflow、工具白名单、权限/引用校验、人工升级、审计轨迹 | `AgentRuntime`, `ToolPolicy`, `AuditTrace`, `HumanReview` |

平台需要把这些能力抽象成通用协议、通用 adapter 和可替换实现。

## 4. 产品模块

### 4.1 Workspace

Workspace 是项目级隔离单元。一个 workspace 包含：

- task schema：任务输入、输出、工具、约束和指标定义；
- datasets：seed、trace、SFT、preference、RL prompt、eval set；
- eval suites：评测集、judge 配置、指标口径、通过门槛；
- train recipes：SFT、DPO、GRPO、PPO、OPD 等训练模板；
- model registry：checkpoint、adapter、merge 产物、部署记录；
- case board：badcase、聚类、归因、处理动作和状态。

### 4.2 Data Factory

Data Factory 负责生产可训练、可评测、可回放的数据。

主要能力：

- ingestion：导入 JSONL/CSV/Parquet/API trace；
- cleaning：字段归一、去重、异常过滤、长度限制；
- governance：字段保留策略、采样策略、访问策略；
- synthetic generation：基于 seed 扩写 query/task，多轮用户模拟；
- agent rollout：调用当前 agent/policy 生成工具轨迹；
- quality gate：schema 检查、工具参数检查、reward/judge/human gate；
- projection：投影为 SFT、preference、RL prompt、OPD 数据格式；
- lineage：记录来源、版本、prompt hash、工具版本、reward 版本。

核心原则：低分样本不能直接进入训练集。它应该先进入 badcase 池，被归因后生成新的训练任务或评测任务。

### 4.3 Evaluation System

评测系统既要支持离线 benchmark，也要支持 Agent 轨迹级评测。

评测类型：

- exact/rule：格式、JSON schema、工具名、参数合法性；
- task metric：分类 F1、检索 Recall@k、pass@k、任务成功率；
- trajectory metric：工具调用顺序、重试次数、无效工具率、引用一致性；
- reward model：可插拔 reward function 或 reward server；
- LLM-as-judge：rubric 化评分，要求 judge 版本和 prompt 版本可追踪；
- human review：人工标注 chosen/rejected、错误类型和备注；
- regression gate：上线前必须通过冻结评测集和关键业务指标门槛。

评测结果需要保留样本级明细，不能只有平均分。case analyzer 依赖样本级失败原因。

### 4.4 Training Factory

Training Factory 不重新实现所有训练框架，而是做统一编排和适配。

第一批训练后端：

- SFT/DPO/KTO/ORPO：优先适配 LLaMA-Factory 或 Axolotl；
- PPO/GRPO/OPD/MOPD：优先适配 verl；
- 轻量本地实验：可适配 TRL；
- 高性能 RLHF：可选适配 OpenRLHF。

训练平台能力：

- recipe registry：训练配置模板、变量、默认资源；
- data resolver：按 dataset version 解析训练输入；
- preflight：检查数据 schema、tokenizer、模型路径、显存预算；
- launcher：本地、SSH、Slurm、Kubernetes、Ray job 适配；
- metrics：TensorBoard、MLflow、console log、W&B/SwanLab 可选；
- artifact sync：checkpoint、adapter、merged model、训练日志归档；
- failure capture：OOM、依赖不兼容、Ray resource mismatch、数据格式错误自动归因。

训练类型与输入输出：

| 类型 | 输入 | 输出 | 常见用途 |
|---|---|---|---|
| SFT | `sft.jsonl` | SFT adapter/checkpoint | 学会格式、工具协议、基础任务行为 |
| DPO | `preference.jsonl` | DPO adapter/checkpoint | 校正边界 case 的偏好 |
| GRPO/PPO | prompt set + reward | RL checkpoint | 优化可计算 reward 或 judge reward |
| OPD | student prompt + teacher model | student checkpoint | 缓解离线蒸馏分布不匹配 |
| MOPD | prompt + teacher router | student checkpoint | 多专家 teacher 能力迁移 |

### 4.5 Case Analyzer

Case Analyzer 是飞轮的核心。它把失败样本转成下一轮行动，而不是只列 badcase。

输入：

- eval sample result；
- agent trace；
- tool calls；
- reward/judge reason；
- human labels；
- model version and dataset version。

输出：

- error taxonomy：格式错、工具错、检索缺证据、推理错、过度拒答、幻觉、长上下文丢失等；
- cluster：按任务类型、工具、意图、失败原因聚类；
- action proposal：补 SFT 数据、构造 preference pair、加入 RL prompt、修改 reward、修改工具 schema、加入 regression eval；
- priority：按线上频率、业务风险、评测损失和修复成本排序。

自动化迭代必须保守：平台可以自动生成候选数据和实验计划，但进入训练集、上线门禁和 reward 变更默认需要人工确认。

### 4.6 Agent Runtime And Serving

训练好的 agent 需要能上线使用，并把反馈带回平台。

运行时原则：

- 工具权限在程序层白名单控制，不只靠 prompt；
- 每次请求记录 trace id、model id、prompt version、tool version；
- 高风险或不确定请求进入人工复核；
- serving 使用 OpenAI-compatible API，默认支持 vLLM/SGLang；
- 支持 canary、A/B、回滚和 shadow evaluation；
- 线上 trace 按采样策略进入 Data Factory。

## 5. 数据契约

### 5.1 TaskSample

```json
{
  "sample_id": "task-001",
  "workspace_id": "demo-agent",
  "input": {"query": "book a meeting room tomorrow afternoon"},
  "context": {"locale": "en-US", "user_tier": "standard"},
  "expected": {"success_criteria": ["valid_tool_call", "clear_final_answer"]},
  "tags": ["scheduling", "tool-use"],
  "lineage": {
    "source": "synthetic",
    "source_version": "v1",
    "created_at": "2026-09-20T00:00:00Z"
  }
}
```

### 5.2 AgentTrace

```json
{
  "trace_id": "trace-001",
  "sample_id": "task-001",
  "model_id": "qwen-agent-sft-v1",
  "prompt_version": "agent_prompt@sha256:...",
  "steps": [
    {
      "type": "tool_call",
      "tool_name": "calendar.search_slots",
      "arguments": {"date": "2026-09-21", "duration_minutes": 60},
      "validation": {"schema": "passed", "policy": "passed"},
      "observation": {"slots": ["14:00", "15:00"]}
    }
  ],
  "final_answer": "I found two available slots: 14:00 and 15:00.",
  "lineage": {"tool_bundle": "calendar-tools@v1"}
}
```

### 5.3 EvalResult

```json
{
  "eval_id": "eval-001",
  "trace_id": "trace-001",
  "suite_id": "tool-agent-regression@v1",
  "scores": {
    "format": 1.0,
    "tool_success": 1.0,
    "answer_quality": 0.8
  },
  "passed": true,
  "reasons": ["valid_tool_call", "answer_supported_by_observation"],
  "judge": {
    "type": "rule+llm",
    "judge_model": "open-model-or-configurable-provider",
    "rubric_version": "rubric@v1"
  }
}
```

### 5.4 TrainRun

```json
{
  "train_run_id": "train-001",
  "workspace_id": "demo-agent",
  "recipe": "sft.lora.llamafactory@v1",
  "base_model": "Qwen/Qwen3-1.7B",
  "datasets": ["sft-dataset@2026-09-20"],
  "status": "running",
  "metrics_uri": "runs/train-001/tensorboard",
  "artifacts": {
    "adapter": "s3://bucket/train-001/adapter",
    "logs": "s3://bucket/train-001/logs"
  }
}
```

## 6. 系统架构

```text
Web UI
  datasets / evals / train runs / case board / model registry / deployment
    |
    v
API Server
  auth, workspace, metadata, dataset registry, recipe registry
    |
    +------------------+
    |                  |
    v                  v
PostgreSQL        Object Store
metadata          jsonl/parquet, logs, checkpoints, reports
    |
    v
Job Orchestrator
  data jobs / eval jobs / train jobs / analysis jobs / deploy jobs
    |
    +-----------+------------+-------------+
    |           |            |             |
Data Worker  Eval Worker  Train Worker  Deploy Worker
```

推荐先实现单机版：

- SQLite/PostgreSQL 均可；
- 本地文件系统作为 object store；
- local worker 执行数据、评测和 smoke training；
- 训练后端先只生成 dry-run manifest，再接真实 LLaMA-Factory/verl。

之后再扩展到：

- MinIO/S3；
- Kubernetes/Ray/Slurm；
- 多租户 workspace；
- 分布式训练和模型服务。

## 7. UI 设计

第一版 UI 应该是工作台，不是营销页。

核心页面：

- Dashboard：当前 flywheel rounds、模型候选、关键指标、阻塞项；
- Datasets：seed、trace、accepted、rejected、SFT、preference、RL prompt；
- Evaluation：评测套件、样本级结果、指标趋势、门禁；
- Training：recipe、资源、实时日志、TensorBoard/MLflow 链接、产物；
- Case Board：badcase 聚类、错误归因、行动建议、处理状态；
- Model Registry：base model、adapter、merged model、eval report、部署状态；
- Deployment：canary、A/B、rollback、线上 trace 采样策略。

## 8. 自动化迭代策略

平台的自动迭代不应该直接无脑重训。推荐采用四级动作：

| 级别 | 动作 | 是否自动执行 |
|---|---|---|
| L0 | 重评测、case 聚类、生成报告 | 可以自动 |
| L1 | 生成候选数据、候选 preference pair、候选 eval case | 可以自动，但需标记未审核 |
| L2 | 发起训练实验、创建 train run | 默认人工确认 |
| L3 | 提升模型到线上、修改 reward 或工具权限 | 必须人工审批 |

每一轮迭代产生一个 `FlywheelRound`：

```text
round_id
base_model
input_datasets
eval_before
training_runs
eval_after
case_analysis
recommended_actions
promotion_decision
```

## 9. 开源实现路线

### Milestone 0: 方案和协议

- 完成技术设计；
- 定义 TaskSample、AgentTrace、EvalResult、TrainRun、FlywheelRound；
- 准备 synthetic demo dataset；
- 明确 adapter 机制和示例协议。

### Milestone 1: 本地最小闭环

- CLI 跑通 `ingest -> rollout(mock) -> eval -> sft export -> train dry-run -> case report`；
- 所有数据写入本地 `runs/<run_id>/`；
- 不依赖 GPU，不依赖远程服务。

### Milestone 2: 训练后端适配

- 接入 LLaMA-Factory SFT/DPO；
- 接入 verl OPD/GRPO 的 dry-run 和真实提交模板；
- 训练指标接入 TensorBoard/MLflow；
- 增加 tokenizer/data preflight。

### Milestone 3: Web 工作台

- 数据集浏览；
- 评测结果样本级查看；
- train run 日志和指标；
- case board；
- model registry。

### Milestone 4: 在线 Agent 和回流

- vLLM/SGLang OpenAI-compatible serving adapter；
- 受控工具白名单；
- trace collector；
- canary gate；
- 线上反馈进入下一轮 flywheel。

## 10. 初始仓库结构

```text
badcaseflow/
  README.md
  docs/
    technical_design_zh.md
  examples/
    datasets/
    tools/
    recipes/
  backend/
    app/
      api/
      core/
      data_factory/
      evaluation/
      training/
      case_analyzer/
      serving/
      registry/
    tests/
  frontend/
  scripts/
  docker/
```

## 11. 关键取舍

1. 平台做编排和元数据，不重复造训练框架。
2. 数据、评测、训练、上线都必须有 version 和 lineage。
3. Agent 工具权限必须在程序层控制。
4. 自动化迭代先做“建议和准备”，谨慎自动上线。
5. 开源仓库只保留 mock、adapter、schema 和通用 recipe。
