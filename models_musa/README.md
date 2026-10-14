# models_musa —— 摩尔线程（MUSA）模型部署与微调教程

本目录是 [self-llm](https://github.com/datawhalechina/self-llm) 的**摩尔线程 MUSA** 适配分册，
收录在摩尔线程 GPU 上部署与微调大模型的完整教程。

所有教程基于**同一个基础镜像**，跟着做即可跑通。

## 基础环境

```
----------------
镜像      registry.mthreads.com/mcconline/training-suite:v2.1.7
操作系统  Ubuntu 22.04.5 LTS
GPU       MTT S5000 (80 GB) × 8
驱动      3.3.5-server（mthreads-gmi 2.3.2）
MUSA SDK  4.3.8（musa_toolkits / musa_runtime）
MCC       5.1.0
Python    3.10.12
torch     2.7.1.post1 + torch_musa 2.7.1.post1
----------------
```

`training-suite` 镜像已经内置了推理与训练所需的全部框架，**不需要自己编译安装**：

| 类别 | 已内置 |
| --- | --- |
| 推理 | vLLM 0.10.1.1（`vllm_musa`）、SGLang 0.5.6.post2、transformers 4.57.0+musa.p34 |
| 训练 | peft 0.20.0、accelerate 1.12.0、DeepSpeed 0.19.3、TransformerEngine、apex、flash-attention 2.6.3 |
| 强化学习 | verl 0.8.0、Ray 2.54.0、slime |
| 其他 | modelscope 1.39.1、torchao 0.16.0、flash_mla / flash_kda |

## 启动容器

```shell
docker run -it -d \
    --name musa-train \
    --runtime=mthreads \
    -e MTHREADS_VISIBLE_DEVICES=all \
    --net=host \
    --shm-size=64g \
    --privileged \
    --security-opt label=disable \
    -v /data:/data \
    registry.mthreads.com/mcconline/training-suite:v2.1.7 \
    /bin/bash

# 进入容器
docker exec -it musa-train bash
```

> - `--runtime=mthreads` 指定摩尔线程容器运行时，由它负责把 MUSA 驱动与设备注入容器（对应 CUDA 场景的 `--runtime=nvidia`）。
> - `-e MTHREADS_VISIBLE_DEVICES=all` 向容器内暴露全部 MUSA 设备；只想用部分卡时写成 `-e MTHREADS_VISIBLE_DEVICES=0,1`。
> - MUSA 设备在容器内通过 `/dev/dri` 暴露（`card0-8` / `renderD128-135`），`--privileged` 是官方镜像的启动方式，可保证算子库正常访问设备。
> - `--shm-size` 建议给到 64g 以上，多卡训练（MCCL、DataLoader 多进程）需要足够共享内存。
> - `--net=host` 便于 vLLM / SGLang 直接以 `localhost:端口` 对外提供 OpenAI 兼容接口。

## 环境自检

```shell
mthreads-gmi                                                    # 查看 GPU 型号、显存、驱动版本
python3 -c "import torch; print(torch.__version__, torch.musa.device_count())"
# 2.7.1.post1 8
```

## 模型权重

各篇教程与同级目录下的脚本统一使用**相对路径**引用权重，形如 `Qwen/Qwen3-4B`，
即执行目录下需要存在 `<组织>/<模型>` 这样的层级：

```
Qwen/
└── Qwen3-4B/
```

用 `modelscope` 下载时把 `cache_dir` 设为 `.`，就会自动落到这个布局：

```python
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-4B', cache_dir='.')
```

权重若已存放在别处，建个软链即可，不必搬动原始文件：

```shell
mkdir -p Qwen
ln -s /path/to/Qwen3-4B Qwen/Qwen3-4B
```

> 路径是相对**执行目录**解析的，请在 `Qwen/` 所在目录下运行脚本。

## 术语对照：CUDA → MUSA

MUSA 是全自研的 GPU 软件栈，但 PyTorch 侧的 API 与 CUDA **一一对位**：把代码里的
`cuda` 换成 `musa` 基本就能跑。下表是可以直接照抄的对照关系（均为本镜像内实测）。

### Python API

| 用途 | CUDA | MUSA |
| --- | --- | --- |
| 设备模块 | `torch.cuda` | `torch.musa` |
| 设备字符串 | `"cuda"` / `"cuda:0"` | `"musa"` / `"musa:0"` |
| 可用性检查 | `torch.cuda.is_available()` | `torch.musa.is_available()` |
| 设备数量 | `torch.cuda.device_count()` | `torch.musa.device_count()` |
| 当前设备 | `torch.cuda.current_device()` | `torch.musa.current_device()` |
| 设备属性 | `torch.cuda.get_device_properties(0)` | `torch.musa.get_device_properties(0)` |
| 数据搬运 | `x.to("cuda")` | `x.to("musa")` |
| 同步 | `torch.cuda.synchronize()` | `torch.musa.synchronize()` |
| 显存统计 | `torch.cuda.memory_allocated()` / `max_memory_allocated()` | `torch.musa.memory_allocated()` / `max_memory_allocated()` |
| 释放显存缓存 | `torch.cuda.empty_cache()` | `torch.musa.empty_cache()` |
| 显存报告 | `torch.cuda.memory_summary()` | `torch.musa.memory_summary()` |
| 随机数状态 | `torch.cuda.get_rng_state()` | `torch.musa.get_rng_state()` |
| 设置随机种子 | `torch.cuda.manual_seed_all(42)` | `torch.musa.manual_seed_all(42)` |
| 流 / 事件 | `torch.cuda.Stream` / `torch.cuda.Event` | `torch.musa.Stream` / `torch.musa.Event` |
| 图执行（Graph） | `torch.cuda.CUDAGraph` | `torch.musa.MUSAGraph` |
| 混合精度 | `torch.amp.autocast("cuda", dtype=torch.bfloat16)` | `torch.amp.autocast("musa", dtype=torch.bfloat16)` |
| 调用上游 CUDA 代码 | — | `torch.musa.overwrite_cuda_api()`（把 `cuda` 调用重定向到 MUSA） |

### 工具链与系统层

| 用途 | CUDA | MUSA |
| --- | --- | --- |
| GPU 监控 | `nvidia-smi` | `mthreads-gmi` |
| 编译器 | `nvcc`（`/usr/local/cuda`） | `mcc`，MUSA Compiler Collection，基于 Clang（`/usr/local/musa/bin/mcc`） |
| SDK 根目录 | `/usr/local/cuda`（`CUDA_HOME`） | `/usr/local/musa`（`MUSA_HOME`） |
| 可见设备变量 | `CUDA_VISIBLE_DEVICES=0,1` | `MUSA_VISIBLE_DEVICES=0,1` |
| 容器运行时 | `--runtime=nvidia` / `--gpus all` | `--runtime=mthreads` |
| 容器内暴露设备 | `-e NVIDIA_VISIBLE_DEVICES=all` | `-e MTHREADS_VISIBLE_DEVICES=all` |
| 容器设备节点 | `/dev/nvidia*` | `/dev/dri`（`card0-8` / `renderD128-135`） |
| 基础镜像 | `nvidia/cuda:xx.x-runtime` | `registry.mthreads.com/mcconline/training-suite` |
| PyTorch 包 | `torch`（CUDA 构建） | `torch` + `torch_musa` |
| 算力标识 | `sm_80` / `sm_90`，`get_device_capability()` → `(8, 0)` | 架构标识 `mp31`，`get_device_capability()` → `(3, 1)` |

### 加速库（一一对位，`/usr/local/musa/lib`）

| CUDA | MUSA | 本镜像版本 |
| --- | --- | --- |
| cuDNN（深度学习算子） | muDNN | 3.1.8.1 |
| cuBLAS / cuBLASLt（矩阵运算） | muBLAS / muBLASLt | 1.10.8 |
| cuFFT（傅里叶变换） | muFFT | 1.10.8 |
| cuRAND（随机数） | muRAND | 1.1.8 |
| cuSOLVER（线性代数） | muSolver | 1.4.8 |
| cuSPARSE（稀疏矩阵） | muSparse | 1.5.8 |
| NCCL（集合通信） | MCCL，Python 侧为 `torch.musa.mccl` | 2.11.4 |
| CUPTI（性能分析接口） | MUPTI | 1.0.8 |

> 实测确认：`MUSA_VISIBLE_DEVICES=0,1` 时 `torch.musa.device_count()` 返回 `2`（默认 `8`），
> 用法与 `CUDA_VISIBLE_DEVICES` 完全一致。

## 目录结构

```
models_musa/
├── README.md
└── qwen3/
    ├── 01-Qwen3-4B-transformers推理教程.md
    └── qwen3_transformers.py             # 最小推理示例（含思考模式与流式输出）
```

## 文档清单

| 编号 | 文档 | 内容 | 底层框架 |
| --- | --- | --- | --- |
| 01 | [Qwen3-4B transformers 推理教程](./qwen3/01-Qwen3-4B-transformers推理教程.md) | transformers 原生推理、思考模式、流式输出 | transformers 4.57.0 |

## 参考

- 镜像仓库：https://mcconline.mthreads.com/repo/training-suite
- [摩尔线程官网](https://www.mthreads.com/)
- [self-llm 主仓库](https://github.com/datawhalechina/self-llm)
