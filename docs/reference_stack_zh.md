# 开源选型与竞品笔记

## 1. 总结

目前没有看到一个成熟开源项目完整覆盖“数据生产、Agent 轨迹、评测、SFT/RL/OPD 训练、训练指标、case 自动分析、上线和线上反馈回流”的全闭环。

更现实的方案是自己做一个轻量控制面，把成熟开源组件接进来：

```text
trace/eval/case data plane: Langfuse / Phoenix / TensorZero 思路
training backend: LLaMA-Factory / verl / Axolotl / OpenRLHF
eval backend: custom runner + OpenCompass / LightEval / promptfoo
serving backend: vLLM / SGLang
metadata and workflow: 自研轻量平台层
```

这个项目的机会点不是重复造每个训练框架，而是把数据、评测、训练和失败 case 迭代接成一个对 Agent 友好的飞轮。

## 2. 接近但不完整的一体化项目

| 项目 | 能覆盖的部分 | 不足 | 对本项目的启发 |
|---|---|---|---|
| TensorZero | LLM gateway、observability、eval、optimization、A/B、生产反馈 | 后训练尤其 RL/OPD 不是其主要训练执行层 | 可以学习“生产反馈 -> 数据/优化”的闭环建模 |
| Langfuse | trace、prompt、dataset、experiment、eval、人审、API/export | 不负责真实训练 | 很适合做 trace/eval/case 数据层的参考 |
| Arize Phoenix | tracing、eval、datasets、experiments、prompt playground、debug | 不是训练平台；license/企业版边界需注意 | 样本级排障和 case 分析体验值得参考 |
| OpenPipe | 日志、fine-tuning、DPO、模型对比 | 开源版本演进状态需谨慎跟踪 | 证明“从日志到微调”的产品方向有需求 |

相关链接：

- TensorZero: https://github.com/tensorzero/tensorzero
- Langfuse: https://github.com/langfuse/langfuse
- Phoenix: https://github.com/Arize-ai/phoenix
- OpenPipe: https://github.com/OpenPipe/OpenPipe

## 3. 后训练框架

| 框架 | 适合做什么 | 平台接入方式 |
|---|---|---|
| LLaMA-Factory | SFT、RM、PPO、DPO、KTO、ORPO、SimPO；Web UI 和配置生态成熟 | 作为 SFT/DPO/轻量 PPO 的默认 adapter |
| verl | PPO、GRPO、DAPO、SPPO、OPD 等 LLM RL post-training；适合复杂 RL/OPD | 作为 RL/OPD/MOPD 的默认 adapter |
| Axolotl | YAML 配置化 SFT、偏好学习、GRPO、RM/PRM，生态活跃 | 作为可选训练 adapter |
| OpenRLHF | 高性能 RLHF，Ray/vLLM/DeepSpeed，多轮和自定义 reward | 作为高性能 RLHF adapter |
| TRL | Hugging Face 训练器库，适合轻量实验和教学 | 作为库级 adapter，而不是完整平台 |

相关链接：

- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- verl: https://github.com/verl-project/verl
- Axolotl: https://docs.axolotl.ai/
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
- TRL: https://huggingface.co/docs/trl/index

## 4. 评测与观测

| 工具 | 价值 |
|---|---|
| OpenCompass | 通用大模型 benchmark 和标准评测 |
| LightEval | Hugging Face 生态内的轻量评测与样本级结果 |
| promptfoo | prompt、RAG、agent 回归测试和 CI 门禁 |
| MLflow | 实验、参数、指标、artifact 管理 |
| TensorBoard | 训练曲线和标量指标 |
| OpenTelemetry | 统一 trace 语义和外部观测系统集成 |

平台第一版可以先自研一个简单 eval runner，把样本级结果落到 JSONL/SQLite。等协议稳定后再逐步接 OpenCompass、LightEval 和 promptfoo。

## 5. 数据生产与标注

| 工具 | 价值 |
|---|---|
| Data-Juicer | 大规模数据清洗、处理、分析 |
| DataFlow | 数据生成、清洗、评估、过滤，覆盖多类 LLM 数据准备 |
| distilabel | 合成数据、AI feedback、偏好数据构造 |
| Argilla | 人审标注、feedback dataset、偏好数据 |
| Label Studio | 通用标注与人工审核 |

第一版不必深接所有数据工具。建议先定义 `DataFactory` 插件接口：导入、生成、过滤、投影、导出。

## 6. 本项目应该避开的坑

- 不要把训练平台做成只会提交 shell 脚本的页面。训练前后的数据版本、评测门禁和 case 分析更重要。
- 不要只看平均分。Agent 训练必须保留样本级失败原因和工具轨迹。
- 不要把 badcase 直接塞回训练集。需要先做归因、去重、分桶和质量门禁。
- 不要把 reward 写死。reward 应是可版本化、可审计、可回放的 provider。
- 不要让模型自由决定工具权限。工具白名单和高风险升级必须在程序层实现。

