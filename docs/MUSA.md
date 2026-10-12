# 在 TrainingSuite v2.1.7 中运行 self-llm

本次适配使用 `registry.mthreads.com/mcctest/training-suite:v2.1.7`、`torchada` 和 `torch_musa`。目标机器是 sjq3（`10.121.38.3`，hostname `worker38003`）。命令在 tmux 中执行。

上游基线：`datawhalechina/self-llm` 的 `e0fe14a35123f5cab6d32cb9e716b571bf0994cf`。适配分支：`codex/local-26-musa`。

## 已提供的入口

| 文件 | 用途 |
| --- | --- |
| `scripts/musa/container.sh` | 使用指定镜像，在租到的卡上启动独立容器 |
| `scripts/musa/install.sh` | 保留镜像的 PyTorch/MUSA 栈；缺少 torchada 时从包索引安装 |
| `scripts/musa/run.py` | 在教程及其 Python 子进程启动前加载 torchada |
| `scripts/musa/smoke.py` | 离线小 Qwen2：训练、生成、LoRA 训练和保存重载 |
| `scripts/musa/qwen.py` | 本地 Qwen 模型的 Transformers 推理和 Trainer LoRA 示例 |

`run.py` 通过仅对本次进程生效的 `PYTHONPATH` 注入启动钩子。钩子先加载 torchada，再检查物理 MUSA 设备；导入失败或没有 MUSA 时直接退出。原教程及子进程的参数、工作目录和退出码保留。

torchada 默认可能保持 `torch.cuda.is_available()` 的原语义。启动钩子在导入前设置 `TORCHADA_FORCE_CUDA_AVAILABLE=1`，让原教程的 CUDA 可用性判断走 MUSA。转换后的张量实际设备类型是 `musa`，不能再用 `device.type == "cuda"` 判断是否在 GPU 上。

## 连接 sjq3 并准备容器

当前本机直连 sjq3 没有内网路由。已验证可通过 pc2 上的 SSH 会话登录：

```bash
# 本机已有 dongyang tmux 会话时，从新窗口执行；保留原窗口的任务。
tmux new-window -t dongyang -n LOCAL-26
ssh pc2
ssh mccxadmin@10.121.38.3
tmux attach -t dongyang
```

在 sjq3 的 tmux 窗口进入仓库目录后，先通过 GPU Worker 获取这台机器的 GPU 租约。以下变量必须来自实际申请结果，不能自行指定空闲卡代替租约：

```bash
export GPU_LEASE_ID='<有效租约 ID>'
export GPU_INDICES='<租约返回的物理卡号，以逗号分隔>'
export GPU_OWNER=dongyang
bash scripts/musa/container.sh bash -lc '
  bash scripts/musa/install.sh &&
  python scripts/musa/run.py scripts/musa/smoke.py
'
```

容器使用 `mthreads` runtime，传入 `MTHREADS_VISIBLE_DEVICES` 做物理设备隔离，退出自动删除。容器内设备从 0 重新编号；单卡 smoke 默认使用 `cuda:0`，由 torchada 转到 `musa:0`。租约需要按 TTL 续期，任务结束后释放。

`install.sh` 不升级 torch、torch_musa 或 transformers。只在 torchada 未安装时运行 `pip install --no-deps torchada`，然后检查导入和依赖一致性。GPU 不可用时不要运行 smoke；依赖核对可以用不暴露设备的 CPU 容器，但这种容器没有 runtime 注入的 MUSA 驱动库，不能用其导入结果判断 GPU 容器是否正常。

## 运行原教程

在原教程要求的工作目录内执行，保留它使用的模型和数据相对路径。以 Qwen LoRA 脚本为例：

```bash
cd /workspace/self-llm/models/Qwen
python ../../scripts/musa/run.py '04-Qwen-7B-Chat Lora 微调.py'
```

该入口解决导入顺序和 PyTorch CUDA API 映射。运行前仍需按原教程准备权重、数据与其 Python 依赖。不要直接照搬 CUDA 环境章节里的 torch、torchvision、NVIDIA 包安装命令。

用模块启动时，启动钩子也会传给 Python worker：

```bash
python scripts/musa/run.py --module torch.distributed.run \
  --standalone --nproc_per_node=1 path/to/tutorial.py
```

Notebook 中把下面单元放到所有 PyTorch、Transformers 导入之前；已导入过框架的 kernel 需要重启：

```python
import os
os.environ["TORCHADA_FORCE_CUDA_AVAILABLE"] = "1"
import torchada
import torch_musa
import torch
assert torch.musa.is_available()
```

## 本地 Qwen 推理和 LoRA 示例

`qwen.py` 使用已有本地权重，禁止自动下载；采用 eager attention，单卡运行。模型和数据放在仓库挂载目录下，命令在 sjq3 的 tmux 窗口执行：

```bash
bash scripts/musa/container.sh bash -lc '
  bash scripts/musa/install.sh &&
  python scripts/musa/run.py scripts/musa/qwen.py infer \
    --model-dir models_local/Qwen3-8B --prompt "请介绍一下你自己。"
'
```

LoRA 数据使用原教程的 `instruction`、`input`、`output` JSON 数组格式。先自行准备 `data/huanhuan.json`，仓库没有附带这个根目录数据文件。输出目录需为空；默认只训练一步，供验证 Trainer 链路：

```bash
bash scripts/musa/container.sh bash -lc '
  bash scripts/musa/install.sh &&
  python scripts/musa/run.py scripts/musa/qwen.py lora \
    --model-dir models_local/Qwen3-8B \
    --data-json data/huanhuan.json --output-dir output_musa/qwen3-lora \
    --max-steps 1 --max-length 256
'
```

该示例没有启用量化、自定义远程模型代码或 fused CUDA kernel。模型是否适用取决于镜像内 Transformers 的模型支持范围；真实 Qwen 权重的推理与训练目前待 sjq3 GPU 验证。

## 兼容范围

| 教程类型 | 处理方式 / 当前状态 |
| --- | --- |
| PyTorch CUDA API、纯 Transformers + PEFT | 已提供启动入口、离线 smoke 和本地 Qwen 示例；GPU 链路待验证 |
| bitsandbytes、4bit/QLoRA、GPTQ/AWQ | 依赖专门的 MUSA 量化实现，不能按 CUDA wheel 直接安装 |
| FlashAttention、FLA、causal-conv1d、Triton 自定义算子 | 需要匹配的 MUSA 算子或 eager 回退；逐项验证 |
| vLLM / SGLang | 必须使用匹配 MUSA 的引擎实现；普通 CUDA wheel 的教程不代表可用 |
| MLX / Metal、AMD ROCm、Ascend | 对应其他硬件后端，不纳入本次入口的支持结论 |

仓库包含多种环境和模型版本。启动入口可复用，GPU 算子、Trainer、量化和引擎兼容性需要按具体教程记录结果，不能从一个小模型 smoke 推断全仓已适配。

## 验证记录（2026-10-12）

- sjq3 的目标镜像已存在，image ID：`sha256:67eafc3ad1ce409cb61996b9f86d7e8893c225d0e802abe8589ef4c945219b5c`。
- 镜像包元数据：torch `2.7.1.post1`、torch_musa `2.7.1.post1`、transformers `4.57.0+musa.p34`、PEFT `0.20.0`、Accelerate `1.12.0`、datasets `3.0.0`。
- 镜像未安装 torchada；不暴露 GPU 的临时容器已成功安装 torchada `0.1.96`，没有安装或升级其依赖。
- 本机启动器的 4 项测试通过：参数/目录/退出码保留、模块和子进程加载钩子、导入失败阻止执行、未启用时不加载框架。
- sjq3 CPU 容器内同样 4 项启动器测试通过；这些测试使用假框架模块，只证明启动流程。
- sjq3 CPU 容器的 Python 编译检查、Shell 语法检查通过，最终 `CPU_CHECK_EXIT=0`；原始日志见 [CPU 检查日志](./validation/local26-20261012-cpu.log)。
- GPU Worker 和 gpu-dashboard 快照均未匹配到 sjq3 的 hostname 或 GPU UUID。GPU Worker 对 `10.121.38.3` 的单卡定向申请返回 `insufficient free GPUs`，没有取得租约。
- 尚未运行实际 MUSA smoke、BF16、真实模型推理、Trainer LoRA 或多卡训练，也未取得 torchada 在 GPU 容器中的导入结果。

本地启动器测试命令：

```bash
python -m unittest discover -s scripts/musa/tests -v
```

## 参考资料

- [torchada 官方仓库与迁移说明](https://github.com/MooreThreads/torchada)
- [TrainingSuite v2.1.7 官方发布说明](https://docs.mthreads.com/training-suite/version-2.1.7/release/)
- [钉钉：TrainingSuite v2.1.7 镜像发布](https://alidocs.dingtalk.com/i/nodes/np9zOoBVByznzOrYCeg9Az73W1DK0g6l)
- [钉钉：training-suite 2.1.7 测试报告](https://alidocs.dingtalk.com/i/nodes/3NwLYZXWyrlNlD0ncNDAdE9QVkyEqBQm)
- [钉钉：starvla 适配操作文档](https://alidocs.dingtalk.com/i/nodes/GZLxjv9VGGRYR5NqT6MKKXjxV6EDybno)
