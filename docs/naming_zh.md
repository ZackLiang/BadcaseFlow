# 命名说明

## 1. 项目名

项目名：`BadcaseFlow`

中文解释：

```text
让 badcase 流动起来。
```

这个项目的核心不是“训练按钮”，而是把 Agent 的失败样本持续转化为下一轮能力改进：

```text
badcase
  -> 归因
  -> 数据候选
  -> 评测候选
  -> 训练 recipe
  -> 训练后回评测
  -> 上线门禁
  -> 新 trace 回流
```

## 2. 为什么不用更宽泛的名字

### 不叫 Agent Training Platform

这个名字太像普通训练平台，容易让用户以为它只负责 SFT/RL 任务提交。

### 不叫 Agent Flywheel Platform

这个名字准确，但偏抽象。用户要读完解释才知道入口是 badcase 和 trace。

### 不叫 Self-Evolving Agent Platform

这个名字有想象力，但容易显得过大。BadcaseFlow 可以承载“受控自进化”的叙事，同时更具体。

## 3. 对外一句话

```text
BadcaseFlow 是一个 Agent 受控自进化平台，把失败 case、评测、数据生成、后训练和上线门禁串成可追踪闭环。
```

## 4. CLI 名

CLI 暂定：`bcf`

示例：

```bash
bcf ingest --workspace demo-agent --input examples/datasets/seed_tasks.jsonl --run runs/demo
bcf rollout --workspace demo-agent --run runs/demo --agent mock
bcf eval --workspace demo-agent --run runs/demo --suite examples/datasets/eval_tasks.jsonl
bcf analyze-cases --workspace demo-agent --run runs/demo
```

## 5. Tagline 备选

推荐：

```text
Turn agent failures into verified improvements.
```

中文：

```text
把 Agent 失败样本转化为可验证的能力改进。
```

