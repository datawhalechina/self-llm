# Ling-3.0-tiny MLX QLoRA 微调

本节我们简要介绍如何基于 `mlx-lm` 对 Ling-3.0-tiny 进行 QLoRA 微调。LoRA 是一种高效微调方法，深入了解其原理可参见博客：[知乎|深入浅出Lora](https://zhuanlan.zhihu.com/p/650197598)。QLoRA 在此基础上把冻结的底座权重量化到 4-bit，训练峰值内存约 12GB——**16GB 内存的 Apple Silicon Mac 即可完成全流程**，无需云端 GPU。

本节的实战任务是把 Ling-3.0-tiny 微调成社交推理桌游「谁是卧底」的**玩家决策组件**：输入身份、词语与轮次历史，输出 ≤25 字发言或 `我投N号`/`我弃权` 投票。选这个任务是因为它同时考察能力可控的小任务：**格式合规**（输出协议）、**信息约束**（描述词但不泄词）、**决策**（从历史发言推断卧底）——而且数据可以 100% 程序化合成，无泄露、可复现。

这个教程会在同目录下给大家提供一个 [notebook](./02-Ling-3.0-tiny%20MLX%20QLoRA%20微调.ipynb) 文件，来让大家更好的学习。

## 环境准备

本文基础环境如下：

```
----------------
macOS（Apple Silicon，验证机型 MacBook Air M4 16GB）
python 3.11
mlx-lm 0.32.0
----------------
```

安装依赖：

```bash
pip install mlx-lm>=0.32.0
```

模型准备同上一节：下载 BF16 权重后**本地**量化为 4-bit（ModelScope 的 int4 预量化是 vLLM 专用，MLX 用不了）：

```shell
modelscope download --model inclusionAI/Ling-3.0-tiny --local_dir ~/models/Ling-3.0-tiny
python -m mlx_lm.convert --model ~/models/Ling-3.0-tiny \
    --q-bits 4 --mlx-path ~/models/Ling-3.0-tiny-4bit --trust-remote-code
```

## 指令集构建

LLM 的微调一般指指令微调过程。`mlx-lm` 的训练数据是 **chat 格式的 JSONL**，每行一个 `messages` 列表：

```json
{
    "messages": [
        {"role": "user", "content": "你在玩「谁是卧底」游戏。\n你的身份：平民\n你的词语：火锅\n……\n请只输出「我投N号」或「我弃权」："},
        {"role": "assistant", "content": "我投3号"}
    ]
}
```

数据合成的关键设计（完整合成器见本节配套代码）：

1. **程序化合成**：30 组相近词对（火锅/麻辣烫、可乐/雪碧……）× 属性池 × 模板，随机分配身份（1 卧底 5 平民），按模板生成每轮发言——卧底刻意模糊、平民用判别属性；再用规则打投票标签。固定随机种子，可复现、无泄露。
2. **任务平衡**：投票比发言难学，训练集里投票样本过采样 3 倍（约 432 条 speak + 780 条 vote），验证/测试集保持自然分布。
3. **留出测试**：留 6 组词对不出现在训练集，专门测泛化。
4. **训练/推理同分布**：提示词里「历史发言」的渲染格式必须与推理时**逐字节一致**；如果推理时会做窗口折叠/截断，训练数据必须覆盖折叠后的全部取值形态（这是很容易踩的坑，见文末常见问题）。

把生成的数据按 `mlx-lm` 的布局放好：

```
data/
├── train.jsonl   # 训练集
├── valid.jsonl   # 验证集
└── test.jsonl    # 测试集（评测用）
```

## QLoRA 训练

创建训练配置 `lora_config.yaml`：

```yaml
model: ~/models/Ling-3.0-tiny-4bit   # 指向 4-bit 量化目录即自动启用 QLoRA
train: true
data: ./data

fine_tune_type: lora
optimizer: adamw
num_layers: 16        # 24 层中的后 16 层挂 LoRA（MoE 专家权重全冻结）
batch_size: 1
grad_accumulation_steps: 4          # 等效 batch 4，16G 内存友好
iters: 1500
learning_rate: 1.0e-4
max_seq_length: 1024
mask_prompt: true                   # loss 只算 assistant 输出
grad_checkpoint: true               # 省内存

adapter_path: ./adapters            # LoRA 权重输出目录
save_every: 200
steps_per_report: 20
steps_per_eval: 100
val_batches: 16
seed: 42
trust_remote_code: true
```

启动训练：

```bash
python -m mlx_lm lora -c lora_config.yaml
```

关键配置说明：

- **LoRA 拓扑**：只挂后 16/24 层的注意力投影，MoE 专家权重全冻结——可训练参数 104M（约 1.3%），这是 16GB 内存能跑动训练的关键；
- **`mask_prompt`**：loss 只算 assistant 输出，不让模型浪费容量去背提示词；
- **1500 步约 2.5 小时**（M4 16GB 实测），每 200 步存 checkpoint，每 100 步跑验证集。

> **16GB 的训练边界（实测）**：连续训练约 625 步后会确定性触发
> `Metal Insufficient Memory`（同种子两次死在同一步，优化器状态分配峰值 + 碎片化所致）。
> 对策是**分段续训**：每段 ≤600 步，段间接力：
>
> ```bash
> python -m mlx_lm lora ... --iters 600 --adapter-path ./adapters-seg1
> python -m mlx_lm lora ... --iters 300 \
>     --resume-adapter-file ./adapters-seg1/0000600_adapters.safetensors \
>     --adapter-path ./adapters-seg2
> ```
>
> 注意检查点文件名是 **7 位**零填充（`0000600_adapters.safetensors`）；段间优化器态
> 重置、lr 重新预热，验证 loss 在段初有尖峰属预期。推理只需要 `adapters.safetensors`
> + `adapter_config.json` 两个文件（约 400MB），中间 checkpoints 不用带。

## 推理验证

微调完成后，用 `--adapter-path` 加载 LoRA 权重推理：

```bash
python -m mlx_lm.generate --model ~/models/Ling-3.0-tiny-4bit \
    --adapter-path ./adapters \
    --prompt "你在玩「谁是卧底」游戏。\n你的身份：平民\n你的词语：火锅\n……" \
    --max-tokens 64
```

或者起 OpenAI 兼容服务（客户端请求体里用 `adapters` 字段指定 LoRA 目录）：

```bash
python -m mlx_lm.server --model ~/models/Ling-3.0-tiny-4bit --port 8080
```

## 微调效果（实测）

以「谁是卧底」玩家决策任务为例，在 96 条测试集（含 6 组留出词对）上的前后对比
（固定种子采样 temp=0.7, top_p=0.9）：

| 指标 | 基线（4-bit 底座） | +QLoRA |
|---|---|---|
| 发言字数合规率（≤25 字） | 93.3% | **100%** |
| 发言自然度（非思考前缀） | 8.3% | **100%** |
| 发言格式合规率 | 3.3% | **93.3%** |
| 投票格式合法率（`我投N号`/`我弃权`） | 52.8% | **100%** |
| 投票正确率（对齐规则式标签） | 11.1% | 25.0% |

可以看到格式协议类指标（字数、可解析率）几乎满分，而「投票正确率」这类决策指标
提升有限——用规则式标签合成的数据有不可约的标签噪声，要提升判别力需要更强的
监督信号（如实战对局的真值标签、大模型教师蒸馏）。诚实记录天花板，也是教程的一部分。

## 常见问题

1. **微调后每个输出都带 `</thinking>` 前缀**：Ling 的 chat 模板带 thinking 块，
   `mask_prompt` 的偏移量算到 `<think>` 为止，把闭合标签留在了监督区。修法：
   推理时用 `apply_chat_template(..., thinking_option="off")` 渲染提示词，
   解析输出时剥掉 `<|role_end|>` 等模板残渣。**权重没坏，是渲染错位**。
2. **贪心解码模式坍缩**：temp=0 时投票 100% 输出「我弃权」、发言全用万能句
   （弃权是单一字符串，多个候选拉平了概率质量）。评测与推理统一用固定种子
   采样（temp=0.7, top_p=0.9）。
3. **训练分布覆盖不到推理形态**：如果推理时对输入做了折叠/截断/变换（比如长历史
   折叠成一行），训练数据必须覆盖该变换的全部取值——否则「训练/推理同格式」只是
   部分成立，实战一上来就是没见过的输入形态。
4. **量化 KV 别乱开**：`--kv-bits 8 --quantized-kv-start 2048` 在 >2048 token 的
   提示词上会崩（`AttributeError: 'list' object has no attribute 'swapaxes'`），
   是 mlx-lm 对该 MoE 架构的量化 KV 兼容问题（[ml-explore/mlx-lm#2057](https://github.com/ml-explore/mlx-lm/issues/2057)）。
   短提示词冒烟测试踩不到，真实长提示词一打就炸；16G 单实例显存够用，不开即可。

## 小结

本节走通了「程序化合成指令数据 → 16GB Mac 上 QLoRA 微调 → 评测对比」的完整流程。
同样的配方可以迁移到任意小任务组件的微调上：找一个**输出协议明确、数据可合成、
评测可自动化**的任务，QLoRA 让 1.3B 激活参数的模型在消费级笔记本上变成专属小工具。
