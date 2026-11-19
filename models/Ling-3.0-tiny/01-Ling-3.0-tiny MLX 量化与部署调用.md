# Ling-3.0-tiny MLX 量化与部署调用

本节我们简要介绍如何基于苹果原生框架 MLX，在 Apple Silicon Mac 上对 Ling-3.0-tiny 模型进行 4-bit 量化与推理部署。MLX 是苹果为 Apple 芯片设计的机器学习框架，训练与推理共用一套栈，量化后的模型在 16GB 内存的 MacBook 上即可流畅运行。

这个教程会在同目录下给大家提供一个 [notebook](./01-Ling-3.0-tiny%20MLX%20量化与部署调用.ipynb) 文件，来让大家更好的学习。

## 环境准备

本文基础环境如下：

```
----------------
macOS（Apple Silicon，验证机型 MacBook Air M4 16GB）
python 3.11
mlx-lm 0.32.0
----------------
```

> 本文默认学习者已安装好 Python 环境。本章内容**只适用于 Apple Silicon Mac**；NVIDIA GPU 用户请参考本仓库其他章节的 transformers + peft 路线。

首先`pip`换源加速下载并安装依赖包

```bash
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install mlx-lm>=0.32.0
```

## 模型下载

使用 modelscope 命令行下载模型，参数model为模型名称，参数 local_dir 为模型的下载路径。
打开终端输入以下命令下载模型（BF16 权重约 15GB）：

```shell
modelscope download --model inclusionAI/Ling-3.0-tiny --local_dir ~/models/Ling-3.0-tiny
```

## 模型量化

MLX 训练/推理前需要把 BF16 权重本地量化为 4-bit（约 4.8GB）：

```shell
python -m mlx_lm.convert --model ~/models/Ling-3.0-tiny \
    --q-bits 4 --mlx-path ~/models/Ling-3.0-tiny-4bit --trust-remote-code
```

> **注意**：ModelScope 上的 int4 预量化版本是 vLLM 专用格式，MLX 无法使用，必须从 BF16 权重本地量化。

> **常见报错**：若量化后加载报
> `StrictDataclassFieldValidationError: Field 'kda_lower_bound' expected float, got int`，
> 是官方 `config.json` 里 `kda_lower_bound: -5`、`mtp_loss_scaling_factor: 0` 写成了
> 整数字面量，MLX 严格校验 float。把两处改成 `-5.0` / `0.0` 即可（量化前后两个目录都要改）。

## 命令行推理

量化完成后，即可用 `mlx_lm` 命令行进行对话测试：

```shell
python -m mlx_lm.generate --model ~/models/Ling-3.0-tiny-4bit \
    --prompt "介绍一下什么是机器学习" --max-tokens 256
```

## OpenAI 兼容服务部署

`mlx_lm.server` 可以启动一个 OpenAI 兼容的 HTTP 服务，供各种应用框架接入：

```shell
python -m mlx_lm.server --model ~/models/Ling-3.0-tiny-4bit --port 8080
```

服务启动后，用任意 OpenAI SDK / HTTP 客户端即可调用：

```shell
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "介绍一下什么是机器学习"}],
    "max_tokens": 256
  }'
```

> **16GB 内存建议**：服务参数加 `--prompt-cache-size 2` 开启提示词前缀缓存，
> 对「每次请求都带相同长系统提示词」的应用（如 Agent 框架）延迟收益明显；
> **暂时不要开** `--kv-bits 8 --quantized-kv-start 2048`（量化 KV 与该模型的 MLA
> 实现的兼容问题见 [ml-explore/mlx-lm#2057](https://github.com/ml-explore/mlx-lm/issues/2057)，
> 修复见 [#2058](https://github.com/ml-explore/mlx-lm/pull/2058)），单实例显存足够用。

## 小结

至此我们完成了 Ling-3.0-tiny 的量化与部署。下一节将介绍如何在 16GB 内存的 Mac 上，
用 QLoRA 把它微调成一个具体的任务组件——「谁是卧底」桌游的玩家决策模型。
