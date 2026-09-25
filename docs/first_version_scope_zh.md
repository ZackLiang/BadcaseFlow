# 首版范围说明

## 首版定位

BadcaseFlow v0.1.0 是一个面向 Agent 后训练的飞轮控制层。它不替代训练框架、评测框架或模型服务，而是把这些能力接到同一条可追踪链路里：

```text
任务 / trace
  -> 评测
  -> badcase 分析
  -> 训练数据导出
  -> SFT / RL / OPD recipe
  -> 训练 run
  -> 候选模型回评测
  -> promotion gate
  -> job runner
  -> 部署计划
  -> 静态工作台
  -> 行动状态跟踪
  -> 下一轮迭代计划
```

首版最重要的价值不是“自动把模型训到最好”，而是让每次改进都有来源、有证据、有产物、有门禁、有下一步。

## 当前已经能做

- 跑通本地 demo：导入任务、mock rollout、规则评测、case 分析和训练数据导出。
- 导入已有 Agent trace：把外部执行轨迹纳入同一套评测和分析流程。
- 管理数据版本：记录数据类型、记录数、`sha256`、schema 状态和来源信息。
- 生成训练计划：对接 LLaMA-Factory、verl 和 command backend，统一 dry-run、run、日志、指标和诊断。
- 生成评测计划：对接 builtin、command、promptfoo、OpenCompass 和 LightEval。
- 编排一轮飞轮：用 `bcf round run-local` 生成 `round.json`、`events.jsonl`、`data_run/` 和 `iteration_plan`。
- 登记候选模型：把 train run、dataset version、eval report 和 promotion decision 绑定到 registry。
- 生成部署计划：为 vLLM、SGLang 或 OpenAI-compatible endpoint 写出 JSON 计划和 PowerShell runbook。
- 导出静态工作台：用 `bcf workbench build` 生成 HTML 和 `workbench.json`，集中查看 round、case action、registry、训练和评测摘要。
- 跟踪迭代行动：用 `bcf action list/update` 管理行动项状态、负责人和备注。
- 执行启动任务：用 `bcf job submit/run/status/logs/collect` 把 launch plan 纳入可追踪任务。
- 自检首版完整性：用 `bcf doctor --strict` 检查示例、recipe、adapter 和文档是否齐全。

## 还做得不够好的地方

- 工作台还偏只读：首版已能导出静态 HTML，但还没有交互式筛选、审批、编辑和多人协作。
- case 分析还偏规则：能给出明确分类和行动建议，但还没有语义聚类、相似 case 合并和优先级排序。
- 自动迭代还偏轻量：`iteration_plan` 已支持行动状态、负责人和备注，但还没有任务队列、审批流和执行回写。
- 训练资源调度还偏轻量：当前重点是 recipe、harness、launcher plan 和 job runner，还没有统一管理 GPU 队列、并发和重试。
- 线上回流还没有闭环：部署计划已具备，trace 采样、灰度监控和回滚建议还需要继续开发。
- 数据治理还比较基础：已经有 schema 和版本记录，后续需要补去重、质量打分、污染检查和评测集冻结策略。

## 首版验收路径

```bash
python -m pip install -e .
bcf version
bcf doctor --strict
bcf demo --run runs/demo
```

进一步跑一轮包含训练 harness、registry 和门禁的本地飞轮：

```bash
bcf round run-local \
  --workspace demo-agent \
  --round runs/rounds/round-smoke \
  --train-mode run \
  --recipe examples/recipes/command_smoke.example.yaml \
  --model-id demo-agent-candidate \
  --attach-round-eval

bcf round inspect --round runs/rounds/round-smoke
bcf round plan-next --round runs/rounds/round-smoke
bcf action list --plan runs/rounds/round-smoke/iteration_plan.json
bcf registry model-lineage --model-id demo-agent-candidate
bcf launcher plan --kind train --launcher local --mode run --recipe examples/recipes/command_smoke.example.yaml --run-id job-smoke --output runs/job-smoke-plan.json
bcf job submit --plan runs/job-smoke-plan.json --job-id job-smoke
bcf job run --job runs/jobs/job-smoke
bcf workbench build --registry-root .
```

## 下一阶段优先级

1. 交互式 Web 工作台：在静态导出的基础上加入筛选、审批、编辑、任务分派和多人协作。
2. Case 智能分析：加入 embedding 聚类、judge 复核、相似 case 合并和行动优先级。
3. 执行编排：把 job runner 升级为队列化任务，支持并发、重试、资源调度和运行状态恢复。
4. 线上回流：接入 trace collector、canary eval、回滚建议和部署后复盘。
5. 数据质量：补数据去重、样本覆盖率、评测集冻结、训练集污染检查和数据准入门禁。
