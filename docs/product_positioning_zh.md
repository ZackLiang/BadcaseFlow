# 产品定位：用户是谁，解决什么问题

## 1. 当前判断

这个项目不适合一开始就定位成通用 LLMOps 平台。LLMOps 范围太大，会把数据集、prompt、观测、训练、部署、权限、计费都卷进来，第一版很容易做散。

更好的切入点是：

```text
面向已经在做 Agent 应用或 Agent 后训练的团队，
提供一个把生产 trace、数据生产、评测、训练、case 分析和上线回流连起来的 Agent 受控自进化平台。
```

一句话表达：

```text
让团队把线上 Agent 的失败案例，稳定转化为可评测、可训练、可上线验证的下一版模型能力。
```

## 2. 候选用户画像

### 2.1 首选用户：Agent / 后训练工程团队

他们通常已经有：

- 一个业务 Agent 或 workflow agent；
- 一批线上 trace、人工反馈或离线评测集；
- SFT、DPO、GRPO、PPO、OPD 等训练诉求；
- vLLM/SGLang 或云模型服务；
- 对模型效果、上线门禁和回滚的工程要求。

他们现在的痛点：

- 数据、评测、训练、部署分散在不同脚本和平台里；
- badcase 只能人工翻日志，很难沉淀成下一轮训练数据；
- 训练完只看 loss，不知道真实任务有没有变好；
- 评测集、训练集、线上 trace 的版本关系不清楚；
- 每轮实验无法解释“为什么这个模型能上线”；
- RL/OPD 这类后训练链路门槛高，缺少可复用 recipe 和 preflight。

这个用户最值得先服务，因为他们痛点强、使用频率高，也愿意接受工程平台。

### 2.2 次选用户：Agent 应用开发团队

他们通常关心：

- 怎么收集用户反馈；
- 怎么看失败 case；
- 怎么让 prompt、工具和模型一起迭代；
- 怎么确认新版本不会把老能力训坏。

他们未必懂训练细节，所以 UI 和默认 workflow 对他们更重要。这个群体适合放在第二阶段服务。

### 2.3 次选用户：研究者和个人开发者

他们可能用平台来跑小模型实验、复现 SFT/DPO/OPD 流程、做 benchmark。

这类用户对开源传播有价值，但不是第一版产品的核心商业/工程用户。第一版可以让他们跑 synthetic demo，但不要为了他们把平台做成教学 notebook 集合。

### 2.4 暂不优先服务的人群

- 只想调用大模型 API、没有训练需求的 prompt 用户；
- 只需要通用日志观测的团队，这类需求 Langfuse/Phoenix 已经覆盖很多；
- 只想跑标准 benchmark 的团队，这类需求 OpenCompass/LightEval 更直接；
- 没有 Agent trace、没有评测集、也没有上线场景的纯探索用户。

## 3. 要解决的核心问题

### 问题一：Agent 失败 case 不能稳定转化为训练资产

很多团队有大量失败日志，但这些日志很难直接进入训练：

- 没有统一 trace schema；
- 工具调用、环境观察、模型输出混在一起；
- 没有标注失败原因；
- 没有区分“该补数据”还是“该改工具/评测/reward”；
- 低质量样本容易污染训练集。

平台要做的是把失败 case 变成结构化资产：

```text
badcase -> 错误归因 -> 样本分桶 -> 数据候选 -> 评测候选 -> 训练候选 -> 人工确认
```

### 问题二：训练和评测割裂

常见情况是训练脚本在一个地方，评测脚本在另一个地方，上线又是第三套流程。结果是：

- 训练 run 不知道用了哪版数据；
- 评测 report 不知道对应哪个 checkpoint；
- 上线模型不一定经过冻结评测集；
- 出问题后无法回溯到样本、reward 和 recipe。

平台要保证每个模型版本都有完整链路：

```text
dataset version -> train recipe -> train run -> checkpoint -> eval report -> promotion decision -> deployment
```

### 问题三：RL/OPD 后训练门槛高

SFT 相对容易，但 RL/OPD 会遇到更多工程问题：

- prompt/response/tokenizer 不兼容；
- teacher/student 资源池不清楚；
- reward function 不可回放；
- rollout 结果没有样本级分析；
- KL、reward、length、success rate 等指标分散在日志里；
- OOM、Ray resource、vLLM 版本问题排查成本高。

平台不必重写 verl/OpenRLHF，但要提供：

- recipe 模板；
- 数据 preflight；
- 资源计划；
- dry-run manifest；
- 指标收集；
- 训练失败归因；
- 训练后自动评测。

### 问题四：上线后缺少下一轮反馈闭环

训练好的 Agent 上线后，线上流量会产生新的分布：

- 新 query；
- 新工具失败；
- 新边界场景；
- 新业务规则；
- 新模型退化。

平台要把线上 trace 安全回流：

```text
线上采样 -> 数据治理 -> 评测 -> case board -> 下一轮数据/评测/训练
```

## 4. 首个落地场景建议

建议第一版选择“工具调用型任务 Agent”作为默认 demo，而不是金融、医疗、法律这类高敏领域，也不是纯聊天助手。

原因：

- 工具调用有明确 trace，适合做 Agent 训练飞轮；
- 成败更容易规则化评测；
- 可以用 calendar/search/calculator/rag 这类 synthetic 工具公开演示；
- 能自然覆盖 SFT、DPO、RL、OPD；
- 可以完全使用合成数据演示。

默认 demo 可以是：

```text
办公任务 Agent
  calendar.search_slots
  document.search
  calculator.evaluate
  ticket.create_draft
```

注意 `ticket.create_draft` 只创建草稿，不真的提交外部系统。这样可以演示有副作用工具的权限控制。

## 5. 产品承诺

平台应该承诺：

- 让每一条训练样本都能回溯到来源和评测；
- 让每一次训练都能回溯到数据、recipe、指标和产物；
- 让每一个上线模型都能回溯到评测报告和审批决策；
- 让每一批 badcase 都能生成下一轮可执行的迭代建议。

平台不应该承诺：

- 自动保证训练效果提升；
- 自动替用户设计 reward；
- 自动替用户审批高风险上线；
- 替代 LLaMA-Factory、verl、OpenRLHF 等训练框架；
- 替代 Langfuse/Phoenix 这类专门观测平台。

## 6. 一句话定位备选

### 版本 A：工程平台型

```text
BadcaseFlow 是一个开源 Agent 受控自进化平台，把 trace、评测、SFT/RL/OPD 训练、case 分析和上线回流串成可追踪闭环。
```

### 版本 B：问题导向型

```text
把线上 Agent 的失败案例，变成下一轮可评测、可训练、可上线验证的模型改进。
```

### 版本 C：开发者工具型

```text
一个给 Agent 团队使用的本地优先训练飞轮工具，从 synthetic demo 到真实后训练环境逐步迁移。
```

建议 README 首屏使用版本 A，项目介绍和演示视频使用版本 B。

## 7. 首版应该如何因此裁剪

如果首选用户是 Agent / 后训练工程团队，首版不必先做复杂权限、多租户和漂亮 UI，应该优先做：

1. 统一数据协议；
2. trace 到 eval 到 SFT export 的闭环；
3. badcase 归因和下一轮建议；
4. LLaMA-Factory/verl recipe dry-run；
5. 训练后自动评测的接口。

暂时不做：

- 多租户计费；
- 完整标注平台；
- 通用 observability 替代品；
- 在线 prompt 管理大而全；
- 复杂 Kubernetes 控制面。

## 8. 需要继续确认的问题

1. 我们第一批真实用户更像“后训练工程师”，还是“Agent 应用负责人”？
2. 第一版 demo 是办公工具 Agent，还是 RAG/合规类 Agent？
3. 用户最想先在 Web 里看到 case board，还是训练/评测 run 管理？
4. 训练执行层下一步优先支持 SSH、Ray、Kubernetes 还是 Slurm？
5. trace 回流优先接 Langfuse、Phoenix 还是 OpenTelemetry？

我的默认建议：

```text
先选“后训练工程师 + 工具调用型 Agent + 本地 CLI 闭环 + adapter 生态 + 可追踪 registry”。
```

这个组合最容易形成可信首版，也最容易从你已有的项目经验自然延展。
