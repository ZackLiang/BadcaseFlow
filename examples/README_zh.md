# 示例说明

这个目录提供一组可直接运行的 synthetic 示例，帮助用户理解数据协议、工具协议和训练 recipe。

## 目录

- `datasets/seed_tasks.jsonl`：合成种子任务，用来模拟线上 trace 或人工构造任务。
- `datasets/eval_tasks.jsonl`：合成评测任务，定义每条样本需要通过哪些检查。
- `tools/tool_manifest.example.json`：mock 工具清单，演示工具 schema、只读/有副作用边界。
- `recipes/sft_llamafactory.example.yaml`：LLaMA-Factory SFT recipe 示例。
- `recipes/opd_verl.example.yaml`：verl OPD recipe 示例。

## 为什么示例内容使用通用办公 Agent

第一版 demo 建议使用办公工具 Agent，而不是垂直行业场景。原因是：

- 工具调用轨迹清晰，适合展示 Agent 训练飞轮；
- 成败标准容易规则化；
- 示例可以完全合成；
- 可以覆盖 SFT、偏好数据、RL prompt、OPD 数据等多种训练输入。

后续用户可以通过 adapter 接入自己的工具、评测集、reward 和训练平台。
