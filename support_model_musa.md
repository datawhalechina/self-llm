# MUSA 平台支持模型列表

> 本页面收录在**摩尔线程 MUSA** 平台上经过实测的大模型部署与微调教程。  
> 每篇教程都给到镜像、驱动版本、启动命令和踩坑说明，命令可直接复制执行。  
> 所有教程均在 MTT S5000 实机上跑通，非纸面推演。

## GPU 硬件环境支持

教程覆盖以下 MUSA 硬件平台：

- **MTT S5000**：训练与推理主力卡，80GB 显存，架构标识 `mp31`；单机 8 卡通过 MCCL 组网；
- **MTT S4000 / S3000**：训练与推理场景，单卡或小规模并行；
- 消费级 MTT S80 可跑通基础推理，但显存与算子覆盖有限，不在本列表的验证范围内。

> 硬件之外还需要匹配的软件版本：MUSA SDK 4.3.x / 5.x、驱动 3.3.x 及以上、torch_musa 与 torch 版本严格对齐。  
> 版本对不上时的典型表现是 `import torch_musa` 失败或算子报 `MUSA error: invalid device function`，不要在这上面纠缠，直接换成配套镜像。

## 目录

- [Qwen3](#qwen3)
- [Qwen3.5 / Qwen3.6](#qwen35--qwen36)
- [服务化性能与精度评测](#服务化性能与精度评测)
- [MUSA 环境配置通用指南](#musa-环境配置通用指南)
- [性能优化建议](#性能优化建议)
- [常见问题](#常见问题)

## 已支持模型列表

### Qwen3

[Qwen3](https://github.com/QwenLM/Qwen3)

- [x] [Qwen3-4B-transformers推理教程](./models_musa/qwen3/Qwen3-4B-transformers推理教程.md) @dongyang-mt

### Qwen3.5

[Qwen3.5](https://github.com/QwenLM/Qwen3.5)

- [x] [Qwen3.5-4B-LoRA及SwanLab可视化记录](./models_musa/Qwen3.5/Qwen3.5-4B-LoRA.ipynb) @dongyang-mt

> 架构要求：**Qwen3.5 系列的 qwen3_5 架构与 transformers 5.x 搭配才能正常推理**。  
> 老版本 transformers（4.x）加载 qwen2/qwen3 架构时可能输出乱码，这是环境级问题，换到 5.x 后消失。


## MUSA 环境配置通用指南

### 1. 系统要求

**操作系统：**

- Ubuntu 22.04 LTS x86_64（推荐，官方镜像即基于此）
- openEuler / 麒麟等信创发行版可用，但需自行处理驱动包与 glibc 版本，非官方验证路径

**硬件要求：**

- MTT S5000 单卡及以上（80GB 显存）
- 最低 64GB 内存（多卡推理或训练建议 256GB+）
- 存储：镜像约 10GB

### 2. 驱动固件安装与 Docker 环境准备

**MUSA 驱动：**

从摩尔线程官方渠道获取与 SDK 版本配套的驱动包（`.deb` 或 `.run`），安装后重启生效。

```bash
# 检查 GPU 状态：型号、显存、驱动版本、利用率
mthreads-gmi
```

预期能看到 `MTT S5000`、`80GB` 以及 `Driver Version 3.3.x`。设备数不对或看不到卡时，先查 `dmesg | grep -i mtgpu`，再看驱动包与内核版本是否匹配（换了内核要重新 DKMS 编译）。

**Docker 环境：**

```bash
# 基础镜像（二选一）
# 1) 训练 + 推理全套：内置 vLLM-MUSA、SGLang、DeepSpeed、TransformerEngine、verl
docker pull registry.mthreads.com/mcconline/training-suite:v2.1.7

# 2) 较新的 torch_musa 官方镜像（torch 2.11 + MUSA 5.2.0）
docker pull registry.mthreads.com/mcconline/torch-musa:2.11.0.post2-musa5.2.0-s5000-py312-ubuntu22.04-x86_64-devel
```

启动容器必须带 `--runtime=mthreads`，由运行时把驱动和设备节点注入容器：

```bash
docker run -it -d \
    --name musa-llm \
    --runtime=mthreads \
    -e MTHREADS_VISIBLE_DEVICES=all \
    --net=host \
    --shm-size=64g \
    --privileged \
    --security-opt label=disable \
    -v /data:/data \
    registry.mthreads.com/mcconline/training-suite:v2.1.7 \
    /bin/bash
```

```bash
# 容器内确认设备可用
mthreads-gmi
python3 -c "import torch, torch_musa; print(torch.__version__, torch.musa.device_count())"
```

> 容器里看不到卡，九成是漏了 `--runtime=mthreads` 或 `--privileged`。  
> MUSA 设备在容器内以 `/dev/dri` 下的 `card*` / `renderD*` 形式出现，没有这两个参数就拿不到设备权限。

### 3. 软件环境

**Python 环境（不走镜像、自行搭建时）：**

```bash
# torch 与 torch_musa 版本号必须一致，否则 import 就挂
conda create -n musa_llm python=3.10
conda activate musa_llm

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# torch_musa 从摩尔线程的 pypi 源取，版本号与 torch 保持一致
pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://dl.mthreads.com/repo/artifacts/index/pypi/
pip install torch_musa==2.7.1.post1 \
    --index-url https://dl.mthreads.com/repo/artifacts/index/pypi/
```

**核心依赖：**

- `torch` + `torch_musa`（版本严格对齐）
- `transformers`（Qwen3.6 系列需 5.x）
- `vllm`（MUSA 版，即 `vllm_musa` 插件）；`sglang`
- `modelscope` / `huggingface_hub`
- 训练侧：`peft`、`accelerate`、`deepspeed`、`transformer_engine`（MUSA 分支）

## 性能优化建议

### 1. CPU 与内存

这两条和CUDA 场景一致，收益稳定：

```bash
# CPU 高性能模式：相同时延约束下推理 TPS 约 +3%
cpupower -c all frequency-set -g performance

# 透明大页：多次实验的吞吐结果更稳
echo always > /sys/kernel/mm/transparent_hugepage/enabled
```

### 2. 框架加速

- 优先用 **vLLM-MUSA** 做服务化：PagedAttention + 连续批处理在 MUSA 上已完整移植；
- 长上下文或高并发场景可试 **SGLang-MUSA**，RadixAttention 对多轮共享前缀收益明显;
- 训练侧用 **TransformerEngine（MUSA 分支）+ flash-attention**，降低 attention 显存与算力开销;
- 静态 shape 的线上服务可上 `torch.musa.MUSAGraph` 捕获计算图，减少 kernel launch 开销（对应 CUDA Graph）。

关键启动参数（vLLM-MUSA）：

```bash
vllm serve Qwen/Qwen3-8B \
    --served-model-name qwen3-8b \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 128
```

`--gpu-memory-utilization` 决定 KV cache 能吃多少显存，拿不准时先 0.85 跑稳，再往上压。`--max-model-len` 按实际业务长度设，设大了会白占 KV cache。

### 3. 模型量化

- **AWQ / GPTQ（int4）**：量化好的权重用 vLLM-MUSA 直接加载（`--quantization awq` 或 `gptq`），显存占用约为 bf16 的 1/3，精度损失在多数对话场景可忽略；
- **torchao**：镜像已内置 0.16.0，可在训练后做 int8 / int4 权重量化，适合自研模型；
- 量化前先跑一遍精度基线（见评测章节），否则出了问题分不清是量化还是环境。

## 常见问题

### Q: 如何确认 MUSA 设备被正确识别？

```bash
# 1) 系统层
mthreads-gmi
```

```bash
# 2) 框架层
python3 -c "import torch, torch_musa; print(torch.musa.is_available(), torch.musa.device_count(), torch.musa.get_device_name(0))"
```

两条都正常才算环境就绪。`mthreads-gmi` 有卡但 `torch.musa.device_count()` 为 0，通常是驱动与 torch_musa 版本不配套。

### Q: 设了可见设备变量但没生效？

`torch_musa` 只认 **`MUSA_VISIBLE_DEVICES`**：

```bash
MUSA_VISIBLE_DEVICES=0,1 python infer.py   # device_count() -> 2
```

写成 `MTHREADS_VISIBLE_DEVICES` 或 `CUDA_VISIBLE_DEVICES` 都不会生效 —— 前者是 `--runtime=mthreads` 在**容器启动时**用的，后者在 MUSA 上被忽略。这两个变量层级不同，容易混。

### Q: 容器里 `mthreads-gmi` 报错 / 看不到设备？

按顺序查：

1. `docker run` 是否带 `--runtime=mthreads`；
2. 是否带 `--privileged`（官方镜像的启动方式，保证算子库能访问 `/dev/dri`）；
3. 宿主机 `mthreads-gmi` 是否正常。宿主机就看不到卡，问题在驱动，不在容器。

### Q: 推理输出乱码或胡言乱语？

先确认 transformers 版本与模型架构匹配。实测：老版本 transformers 加载 qwen2 / qwen3 架构会出现乱码，换到 transformers 5.x 后 qwen3_5 架构正常。这不是模型权重问题，别急着重下权重。

### Q: 训练时报梯度检查点 dtype 不匹配？

手动把权重转成 bf16 再配合 `bf16=True` 的 autocast，会撞上 gradient checkpointing 的 dtype 校验。处理办法：权重保持 fp32 / 用 `torch_dtype` 加载，或关掉 gradient checkpointing，二者选一。

### Q: 如何贡献新的 MUSA 模型教程？

欢迎提 PR。我们特别期待：

- 更多 MUSA 型号（S4000 / S3000 / 多机多卡）的部署教程；
- 训练、微调（LoRA / 全参 / RL）在 MUSA 上的跑通记录；
- 性能基准数据：吞吐、首 token 延迟、显存峰值，附上卡型与镜像版本。

提 PR 时请在文档开头写清 `mthreads-gmi` 输出、镜像 tag、torch_musa 版本 —— 没有这三样，别人复现不了。

> 💡 本列表持续更新。有特定模型的部署需求或想补的场景，走 Issue 或 PR 都行。

## 参考

- 镜像仓库：<https://mcconline.mthreads.com/repo/training-suite>
- [摩尔线程官网](https://www.mthreads.com/)
- [MUSA 开发者文档](https://docs.mthreads.com/)
- [self-llm 主仓库](https://github.com/datawhalechina/self-llm)
