# 在 TrainingSuite v2.1.7 中运行 self-llm

本次适配使用 `registry.mthreads.com/mcctest/training-suite:v2.1.7`、`torchada` 和 `torch_musa`。目标机器是 sjq3（`10.121.38.3`，hostname `worker38003`）。命令在 tmux 中执行。

上游基线：`datawhalechina/self-llm` 的 `e0fe14a35123f5cab6d32cb9e716b571bf0994cf`。适配分支：`codex/local-26-musa`。

## 已提供的入口

| 文件 | 用途 |
| --- | --- |
| `scripts/musa/container.sh` | 使用指定镜像和物理卡号启动独立容器 |
| `scripts/musa/install.sh` | 保留镜像的 PyTorch/MUSA 栈；缺少 torchada 时安装，并拒绝新增依赖冲突 |
| `scripts/musa/check_dependencies.py` | 保存镜像原有依赖冲突，比较安装后的变化 |
| `scripts/musa/run.py` | 在教程及其 Python 子进程启动前加载 torchada |
| `scripts/musa/smoke.py` | 离线小 Qwen2：训练、生成、LoRA 训练和保存重载 |
| `scripts/musa/qwen.py` | 本地 Qwen 模型的 Transformers 推理和 Trainer LoRA 示例 |

`run.py` 通过仅对本次进程生效的 `PYTHONPATH` 注入启动钩子。钩子先加载 torchada，再检查物理 MUSA 设备；导入失败或没有 MUSA 时直接退出。原教程及子进程的参数、工作目录和退出码保留。torchada 初始化时会临时关闭构建子进程的钩子，避免 Python 版 ninja 重复导入并卡在编译锁上；用户脚本启动前恢复钩子。

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

本任务按用户明确授权跳过 GPU Worker，通过 pc2 连接 sjq3，检查实际使用情况后选用空闲的物理 0 号卡。在 sjq3 的 tmux 窗口进入仓库目录后执行：

```bash
export GPU_ALLOW_UNLEASED=1
export GPU_INDICES=0
bash scripts/musa/container.sh bash -lc '
  bash scripts/musa/install.sh &&
  python scripts/musa/run.py scripts/musa/smoke.py
'
```

容器使用 `mthreads` runtime，传入 `MTHREADS_VISIBLE_DEVICES` 做物理设备隔离，退出自动删除。容器内设备从 0 重新编号；单卡 smoke 默认使用 `cuda:0`，由 torchada 转到 `musa:0`。

未设置 `GPU_ALLOW_UNLEASED=1` 时，启动器仍要求有效 `GPU_LEASE_ID`，并使用租约返回的 `GPU_INDICES`；租约模式须按 TTL 续期、结束后释放。直接使用模式不填写虚构的租约 ID，也不添加租约管理标签。

`install.sh` 不升级 torch、torch_musa 或 transformers。先保存镜像的 `pip check` 清单，只在 torchada 未安装时运行 `pip install --no-deps torchada`，再检查导入并比较依赖清单；出现新增冲突或包元数据解析错误时退出。镜像原有冲突会报告出来。目标镜像的 PEFT 0.20.0 会拒绝 torchao 0.9.0，普通 LoRA 也会因此失败；安装脚本在隔离容器中移除这个不兼容的可选包，再核对依赖清单。本入口用于非量化 LoRA，不支持 torchao 量化模型。

GPU 不可用时不要运行 smoke；依赖核对可以用不暴露设备的 CPU 容器，但这种容器没有 runtime 注入的 MUSA 驱动库，不能用其导入结果判断 GPU 容器是否正常。

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

`qwen.py` 使用已有本地权重，禁止自动下载；采用 eager attention，单卡运行。`MUSA_MODELS_ROOT` 指定宿主机模型目录，容器内以只读方式挂载到 `/models`。命令在 sjq3 的 tmux 窗口执行：

```bash
export MUSA_MODELS_ROOT=/data/models
bash scripts/musa/container.sh bash -lc '
  bash scripts/musa/install.sh &&
  python scripts/musa/run.py scripts/musa/qwen.py infer \
    --model-dir /models/Qwen3-0.6B --prompt "请介绍一下你自己。"
'
```

LoRA 数据使用原教程的 `instruction`、`input`、`output` JSON 数组格式。先自行准备 `data/huanhuan.json`，仓库没有附带这个根目录数据文件。输出目录需为空；默认只训练一步，供验证 Trainer 链路：

```bash
bash scripts/musa/container.sh bash -lc '
  bash scripts/musa/install.sh &&
  python scripts/musa/run.py scripts/musa/qwen.py lora \
    --model-dir /models/Qwen3-0.6B \
    --data-json data/huanhuan.json --output-dir output_musa/qwen3-lora \
    --max-steps 1 --max-length 256
'
```

该示例没有启用量化、自定义远程模型代码或 fused CUDA kernel。Qwen3-0.6B 和 Qwen2-0.5B-Instruct 的 BF16 推理及 Trainer LoRA 已在 sjq3 验证。Trainer 的随机采样器显式使用 CPU generator，保留随机打乱和种子；训练张量仍在 MUSA 上。其他模型取决于镜像内 Transformers 的支持范围。

## 兼容范围

| 教程类型 | 处理方式 / 当前状态 |
| --- | --- |
| PyTorch CUDA API、纯 Transformers + PEFT | 单卡 float32/BF16 smoke、Qwen2/Qwen3 推理及 Trainer LoRA 已通过；其他教程逐项验证 |
| bitsandbytes、4bit/QLoRA、GPTQ/AWQ | 依赖专门的 MUSA 量化实现，不能按 CUDA wheel 直接安装 |
| FlashAttention、FLA、causal-conv1d、Triton 自定义算子 | 需要匹配的 MUSA 算子或 eager 回退；逐项验证 |
| vLLM / SGLang | 必须使用匹配 MUSA 的引擎实现；普通 CUDA wheel 的教程不代表可用 |
| MLX / Metal、AMD ROCm、Ascend | 对应其他硬件后端，不纳入本次入口的支持结论 |

仓库包含多种环境和模型版本。启动入口可复用，GPU 算子、Trainer、量化和引擎兼容性需要按具体教程记录结果，不能从一个小模型 smoke 推断全仓已适配。

## 验证记录（2026-10-12）

- sjq3 的目标镜像已存在，image ID：`sha256:67eafc3ad1ce409cb61996b9f86d7e8893c225d0e802abe8589ef4c945219b5c`。
- 镜像包元数据：torch `2.7.1.post1`、torch_musa `2.7.1.post1`、transformers `4.57.0+musa.p34`、PEFT `0.20.0`、Accelerate `1.12.0`、datasets `3.0.0`。
- 镜像未安装 torchada；不暴露 GPU 的临时容器已成功安装 torchada `0.1.96`，没有安装或升级其依赖。
- 本机启动器的 5 项测试通过：参数/目录/退出码保留、模块和子进程加载钩子、导入失败阻止执行、未启用时不加载框架、构建子进程不递归导入。
- sjq3 CPU 容器内同样 4 项启动器测试通过；这些测试使用假框架模块，只证明启动流程。
- sjq3 CPU 容器的 Python 编译检查、Shell 语法检查通过，最终 `CPU_CHECK_EXIT=0`；原始日志见 [CPU 检查日志](./validation/local26-20261012-cpu.log)。
- 目标镜像在安装前已有 5 项依赖冲突，涉及 ml-dtypes、vLLM 与 vllm-musa；torchada 安装后清单不变。依赖比较允许保留这 5 项冲突；临时加入缺失依赖或损坏的包元数据时均返回非零退出码。sjq3 CPU 容器的最终 `DEPENDENCY_GATE_EXIT=0`，见 [依赖比较验证日志](./validation/local26-20261012-dependency-gate.txt)。这不代表 vLLM 或这些包已经兼容。
- 用户明确授权不使用 GPU Worker，直接通过 pc2 连接 sjq3，使用物理 0 号 S5000，未创建或伪造租约。
- GPU 容器中的 torchada / torch_musa 导入通过；float32 和 BF16 smoke 均 PASS，实际设备为 `musa:0`，训练、生成、LoRA 更新和 adapter 保存重载检查通过，`GPU_CHECK_EXIT=0`。原始日志见 [GPU smoke 日志](./validation/local26-20261012-gpu-smoke.log)。
- Qwen3-0.6B 和 Qwen2-0.5B-Instruct 的 BF16 推理均 PASS，实际设备 `musa:0`；Trainer LoRA 各训练 2 步，训练 loss 分别为 `2.7577483654022217` 和 `1.5840654969215393`，`MODEL_CHECK_EXIT=0`。输入为 2 条人工编写的示例记录，用于检查训练链路，不用于评价模型效果。见 [真实模型日志](./validation/local26-20261012-model.log)。
- 仓库 `container.sh` 的直接使用模式和只读本地模型挂载已在 sjq3 实测，Qwen3 推理 PASS，`LAUNCHER_CHECK_EXIT=0`。见 [容器入口日志](./validation/local26-20261012-launcher.log)。
- 本机启动钩子与容器参数检查共 11 项测试通过；见 [测试日志](./validation/local26-20261012-tests.log)。
- 两个真实模型的 adapter 已单独检查：Qwen3 保存 112 个张量，其中 56 个 LoRA B 张量非零；Qwen2 保存 96 个张量，其中 48 个 LoRA B 张量非零。所有张量均为有限值，配置与 tokenizer 文件可读，`ARTIFACT_CHECK_EXIT=0`。见 [adapter 检查日志](./validation/local26-20261012-artifact.log)。
- [Review Notebook](./validation/LOCAL-26-review.ipynb) 汇集实际执行命令和日志；Notebook 读取已有结果，不重复执行 GPU 任务。
- 通用启动层现为 CPU 索引采样保留 CPU Generator 和默认设备。实测发现 torchada 的 `get_default_device()` 返回 `musa:0`，而 `randperm` 的默认索引仍在 CPU；只改 Generator 类还不够。普通 Transformers `Trainer` 已直接完成 Qwen3-0.6B 的两步 BF16 LoRA，112 个 adapter 张量更新，loss 为 `6.338409185409546`。输入为人工 token 序列，结果只验证运行路径。见 [通用 Trainer 日志](./validation/local26-20261013-stock-trainer.log)。
- 本次未运行多卡训练、量化或 vLLM/SGLang。

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
