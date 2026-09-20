# 执行计划：从定位到第一个可用版本

## 1. 当前最该做什么

先不要急着做完整 Web 平台，也不要一开始接真实 GPU 训练。第一阶段最重要的是证明一个闭环：

```text
Agent trace / seed task
  -> 评测
  -> badcase 归因
  -> 生成训练数据候选
  -> 生成训练 recipe dry-run
  -> 训练后回评测接口
```

只要这条链路成立，产品价值就成立。Web、分布式训练、多租户、权限、复杂部署都可以后移。

## 2. 第一版产品原则

1. 先做 CLI，再做 Web。
2. 先做本地文件 run store，再接数据库。
3. 先做 mock Agent，再接真实 Agent trace。
4. 先做规则评测，再接 LLM judge。
5. 先做 LLaMA-Factory SFT dry-run，再做真实训练。
6. 先做 verl OPD dry-run，再做 GPU smoke。
7. 每一步都要保存样本级结果，不能只保存平均分。

## 3. 四周开发节奏

### 第 1 周：数据协议和本地 run store

目标：把平台的地基打稳。

要做：

- 定义核心 schema：
  - `TaskSample`
  - `AgentTrace`
  - `EvalResult`
  - `CaseAnalysis`
  - `DatasetVersion`
  - `TrainRecipe`
  - `TrainRun`
  - `FlywheelRound`
- 实现本地 run 目录：
  - `runs/<run_id>/manifest.json`
  - `runs/<run_id>/tasks.jsonl`
  - `runs/<run_id>/traces.jsonl`
  - `runs/<run_id>/eval_results.jsonl`
  - `runs/<run_id>/case_report.json`
  - `runs/<run_id>/sft.jsonl`
- 实现 CLI 骨架：
  - `bcf ingest`
  - `bcf inspect`
  - `bcf init-workspace`
- 加 3 到 10 条 synthetic seed task。

验收标准：

```bash
bcf ingest --workspace demo-agent --input examples/datasets/seed_tasks.jsonl --run runs/demo
bcf inspect --run runs/demo
```

能看到导入了多少样本、字段是否合法、run manifest 是否完整。

### 第 2 周：mock rollout、评测和 badcase 归因

目标：让平台能从任务跑到失败分析。

要做：

- 实现 mock tool registry：
  - `calendar.search_slots`
  - `calculator.evaluate`
  - `rag.search_policy`
- 实现 mock Agent rollout：
  - 生成工具调用；
  - 生成 observation；
  - 生成 final answer；
  - 保存完整 trace。
- 实现 rule evaluator：
  - schema 检查；
  - tool 是否调用正确；
  - tool 参数是否合法；
  - final answer 是否包含关键证据；
  - 是否出现未授权动作。
- 实现 case analyzer v0：
  - `tool_missing`
  - `tool_argument_error`
  - `answer_not_supported`
  - `format_error`
  - `task_failed`
- 输出 case report：
  - 失败样本列表；
  - 错误类型分布；
  - 每类建议动作。

验收标准：

```bash
bcf rollout --workspace demo-agent --run runs/demo --agent mock
bcf eval --workspace demo-agent --run runs/demo --suite examples/datasets/eval_tasks.jsonl
bcf analyze-cases --workspace demo-agent --run runs/demo
```

能看到哪些样本失败、为什么失败、下一轮建议补什么。

### 第 3 周：训练数据导出和 recipe dry-run

目标：让 badcase 能进入训练准备阶段。

要做：

- 实现 SFT exporter：
  - OpenAI messages 格式；
  - ShareGPT 格式；
  - 保留 trace id 和 lineage。
- 实现 preference candidate exporter：
  - chosen/rejected 先用规则构造；
  - 标记 `needs_review`。
- 实现 RL prompt exporter：
  - prompt + reward metadata；
  - 支持后续转 verl parquet。
- 实现 recipe dry-run：
  - LLaMA-Factory SFT；
  - LLaMA-Factory DPO；
  - verl OPD；
  - verl GRPO。
- 实现 preflight：
  - 文件是否存在；
  - JSONL 是否可解析；
  - 必要字段是否齐全；
  - 输出目录是否冲突；
  - recipe 参数是否缺失。

验收标准：

```bash
bcf export sft --workspace demo-agent --run runs/demo --output runs/demo/sft.jsonl
bcf export preference --workspace demo-agent --run runs/demo --output runs/demo/preference.jsonl
bcf train dry-run --workspace demo-agent --recipe examples/recipes/sft_llamafactory.example.yaml
bcf train dry-run --workspace demo-agent --recipe examples/recipes/opd_verl.example.yaml
```

能生成可读的训练命令、数据摘要和风险提示。

### 第 4 周：训练后回评测和项目演示

目标：把“训练不是终点，回评测才是门禁”做出来。

要做：

- 实现 `bcf eval-model` 的接口形态：
  - model endpoint；
  - eval suite；
  - output report。
- 实现模型版本 manifest：
  - base model；
  - train run；
  - checkpoint path；
  - eval report；
  - promotion decision。
- 实现 before/after 对比报告：
  - pass rate；
  - 每类错误变化；
  - 新增失败；
  - 修复失败；
  - 是否通过门禁。
- 写一个完整 demo 文档：
  - 从 seed task 到 badcase；
  - 从 badcase 到 SFT 数据；
  - 从 recipe dry-run 到回评测报告。

验收标准：

```bash
bcf compare --before runs/baseline/eval_report.json --after runs/demo/eval_report.json
bcf promote dry-run --model artifacts/demo-agent/sft-v0.1 --eval-report runs/demo/eval_report.json
```

能解释一个模型为什么可以或不可以进入候选上线。

### 第 5 周：远程训练 dry-run

目标：让 BadcaseFlow 可以自然接到自己的服务器或 AutoDL。

要做：

- `bcf init-workspace`：生成本地 workspace 配置；
- `bcf remote add`：登记远程目标；
- `bcf remote list`：查看远程目标；
- `bcf remote plan`：生成同步、训练和回收产物的 dry-run 命令；
- 远程计划先输出 JSON 和 PowerShell 脚本；
- 暂不直接执行 SSH 命令，避免一开始受网络、密钥、服务器环境影响。

验收标准：

```bash
bcf init-workspace --workspace demo-agent
bcf remote add --name autodl --host root@your-server --workdir /root/BadcaseFlow
bcf remote plan \
  --target autodl \
  --workspace demo-agent \
  --run runs/demo \
  --recipe examples/recipes/sft_llamafactory.example.yaml
```

能生成：

```text
runs/demo/remote_plan_autodl.json
runs/demo/remote_plan_autodl.ps1
```

## 4. 第一版不要做什么

暂时不要做：

- 完整 Web 工作台；
- 多租户和复杂权限；
- Kubernetes 控制面；
- 在线计费；
- 完整标注平台；
- 替代 Langfuse/Phoenix 的全量 observability；
- 替代 OpenCompass/LightEval 的 benchmark 生态；
- 自己实现 SFT/RL 训练算法。

这些都很重要，但不是证明产品价值的第一步。

## 5. 目录建议

```text
badcaseflow/
  pyproject.toml
  src/agent_flywheel/
    cli.py
    schemas/
      task.py
      trace.py
      eval.py
      train.py
      case.py
    storage/
      local_store.py
    rollout/
      mock_agent.py
      tools.py
    evaluation/
      rule_eval.py
    case_analyzer/
      analyzer.py
    exporters/
      sft.py
      preference.py
      rl_prompt.py
    training/
      recipe.py
      llamafactory.py
      verl.py
      preflight.py
  tests/
  examples/
  docs/
```

## 6. 第一批核心 schema

### TaskSample

代表一个任务输入。

关键字段：

- `sample_id`
- `workspace_id`
- `input`
- `context`
- `expected`
- `tags`
- `lineage`

### AgentTrace

代表一次 Agent 执行轨迹。

关键字段：

- `trace_id`
- `sample_id`
- `model_id`
- `prompt_version`
- `tool_bundle_id`
- `steps`
- `final_answer`
- `metadata`

### EvalResult

代表一条 trace 的评测结果。

关键字段：

- `eval_id`
- `trace_id`
- `suite_id`
- `scores`
- `passed`
- `failure_types`
- `reasons`

### CaseAnalysis

代表 badcase 归因。

关键字段：

- `case_id`
- `trace_id`
- `primary_failure_type`
- `root_cause`
- `recommended_action`
- `target_dataset`
- `needs_human_review`

### TrainRun

代表一次训练任务。

关键字段：

- `train_run_id`
- `recipe_id`
- `base_model`
- `dataset_versions`
- `status`
- `metrics_uri`
- `artifacts`

## 7. 最小差异化能力

第一版最值得打磨的是 case analyzer，不是训练按钮。

它应该能回答：

```text
这个 case 为什么失败？
应该进入 SFT、DPO、RL prompt，还是 eval set？
这个问题是否真的需要训练，还是应该改工具 schema / workflow / reward？
下一轮训练前需要人工确认什么？
```

如果这几个问题回答得好，这个产品就和普通训练脚本、普通观测平台拉开了距离。

## 8. 你现在可以马上做的三件事

1. 把 v0.1 的 CLI 和 schema 先写出来。
2. 用 synthetic 办公 Agent 跑通 10 条样本的闭环。
3. 把 case analyzer 的输出做得非常清楚：失败原因、建议动作、目标数据集、是否需要人工确认。

第一版 demo 不需要华丽，但一定要让人看到：

```text
失败样本不是停在日志里，而是真的被转化成下一轮改进动作。
```
