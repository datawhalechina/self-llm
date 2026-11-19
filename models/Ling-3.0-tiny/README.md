
<div align="center">
<h1>
  蚂蚁 Ling-3.0-tiny 大模型
</h1>
</div>

## 1. 模型简介

Ling-3.0-tiny（百灵）是蚂蚁集团开源的轻量级混合专家（MoE）语言模型，激活参数约 1.3B，采用 `bailing_hybrid` 架构（24 层）。它在极小的激活参数下保持了可用的指令跟随能力，非常适合**端侧部署与微调**——本章教程在 Apple Silicon Mac（验证机型：MacBook Air M4 16GB）上，使用苹果原生框架 [MLX](https://github.com/ml-explore/mlx) 完成量化、推理与 QLoRA 微调的全流程，无需任何云端 GPU。

模型地址（ModelScope）：https://www.modelscope.cn/models/inclusionAI/Ling-3.0-tiny

模型地址（HuggingFace）：https://huggingface.co/inclusionAI/Ling-3.0-tiny

## 2. 模型下载

| 模型 | 下载链接 |
|:---------------------:|:------------------------------------------------------------------------------------------------------------------:|
| Ling-3.0-tiny | [ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-3.0-tiny) \| [HuggingFace](https://huggingface.co/inclusionAI/Ling-3.0-tiny) |

## 3. 教程简介

本教程以 Ling-3.0-tiny 基础，介绍在 **Apple Silicon Mac** 上基于 **MLX** 框架的完整微调流程：

- 01-Ling-3.0-tiny MLX 量化与部署调用
- 02-Ling-3.0-tiny MLX QLoRA 微调（以「谁是卧底」玩家决策组件为例）

> 与本仓库其他章节的 transformers + peft（CUDA）路线不同，本章全部使用 `mlx-lm`，
> 训练与推理一体、量化后 16GB 内存即可跑通，是 Apple 芯片用户的推荐路线。
