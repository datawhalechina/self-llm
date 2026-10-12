# LOCAL-26 验证记录（WIP）

这份报告和同目录 notebook 用于 review。notebook 只读取随包保存的原始日志，不启动容器、申请 GPU 或重跑命令。GPU 和真实模型命令已在 sjq3 的 `dongyang` tmux 独立窗口执行，结果以原始日志为准。

## 结果

| 项目 | 运行位置 | 原始日志 | 结果 |
| --- | --- | --- | --- |
| 依赖 gate | sjq3 `dongyang` tmux，CPU 容器 | `local26-20261012-dependency-gate.txt` | **PASS（有边界）**：真实镜像安装 torchada 后仍为基线 5 项冲突，`new_conflicts=[]`；新增冲突和 malformed metadata 两个负例均被拒绝，`DEPENDENCY_GATE_EXIT=0` |
| CPU runner | sjq3 `dongyang` tmux | `local26-20261012-cpu.log` | **PASS**：torchada `0.1.96`，runner 4 项测试通过 |
| GPU smoke | sjq3 `dongyang` tmux，MTT S5000 / `musa:0` | `local26-20261012-gpu-smoke.log` | **PASS**：float32、bf16 均完成前向、反向、生成、LoRA 训练和保存重载，`GPU_CHECK_EXIT=0` |
| 真实模型 | sjq3 `dongyang` tmux，MTT S5000 / `musa:0` | `local26-20261012-model.log` | **PASS**：Qwen3-0.6B、Qwen2-0.5B-Instruct 各完成 infer 和 LoRA 2 steps，`MODEL_CHECK_EXIT=0` |
| 启动器 | sjq3 `dongyang` tmux，显式 `GPU_ALLOW_UNLEASED=1` | `local26-20261012-launcher.log` | **PASS**：启动器 infer 成功，`LAUNCHER_CHECK_EXIT=0` |
| 容器/runner 单测 | macOS `dongyang` tmux | `local26-20261012-tests.log` | **PASS**：11 项通过，`TEST_CHECK_EXIT=0` |
| 静态检查 | macOS `dongyang` tmux | `local26-20261012-static.json` | **PASS**：compileall、bash -n、git diff --check 均为 0 |
| adapter 产物 | sjq3 `dongyang` tmux，CPU 加载权重核验 | `local26-20261012-artifact.log` | **PASS**：两个 adapter 的 safetensors 和非零 LoRA B 均通过，`ARTIFACT_CHECK_EXIT=0` |

## 关键证据

GPU smoke 使用 `python scripts/musa/run.py scripts/musa/smoke.py`，分别测试 `--dtype float32` 和 `--dtype bf16`。日志确认设备为 MTT S5000，物理类型为 MUSA；两种 dtype 的 `checks` 均为 true。

真实模型使用 `qwen.py`：

- `Qwen3-0.6B`：infer 通过；LoRA 2 steps，loss `2.7577483654022217`，输出 `/results/qwen3-lora`。
- `Qwen2-0.5B-Instruct`：infer 通过；LoRA 2 steps，loss `1.5840654969215393`，输出 `/results/qwen2-lora`。

产物核验结果：

- `qwen3-lora`：112 个 saved tensors，56 个非零 LoRA B tensors。
- `qwen2-lora`：96 个 saved tensors，48 个非零 LoRA B tensors。

依赖 gate 记录的基线冲突为 5 项：`ml-dtypes` 对 numpy、`vllm` 对 numpy/outlines-core/xgrammar，以及 `vllm-musa` 对 pyarrow。安装 torchada 后没有新增冲突；报告保留这些基线问题，不把环境描述成“依赖全部通过”。

## 归档命令

- `commands/local26-deps-gate.sh`
- `commands/local26-gpu.sh`
- `commands/local26-model.sh`
- `commands/local26-launcher-check.sh`
- `commands/local26-artifact.sh`
- `commands/local26-static.py`
- `commands/local26-verify-artifacts.py`

这些脚本已检查并移除了凭据字段；模型路径、普通参数和结果目录仍按实际运行记录保留。未读取或上传认证文件。

## 2026-10-13：普通 HF Trainer

通用启动层为 CPU 索引采样保留 CPU Generator 和默认设备。torchada 的全局默认设备 getter 返回 MUSA，而默认 randperm 索引仍在 CPU；只改 Generator 类仍会失败。

未改写的 Transformers Trainer 已在 sjq3 上完成 Qwen3-0.6B 两步 BF16 LoRA。设备为 `musa:0`，112 个 adapter 张量更新，loss 为 `6.338409185409546`，`STOCK_TRAINER_CHECK_EXIT=0`。输入为人工 token 序列，只验证运行路径。Notebook 保留了初次回归断言错误、生成器不匹配和最终通过三次日志。

新的 11 项测试、Python 编译和 diff 检查在 macOS tmux 执行并通过。见 [普通 Trainer 日志](./local26-20261013-stock-trainer.log) 和 [静态记录](./local26-20261013-static.json)。

## 剩余风险

镜像原有的 5 项依赖冲突仍需后续收敛；日志还出现 transformers 参数弃用、生成参数提示和 PEFT 缺少 config 的 warning。本次验证证明了当前镜像上的运行路径，但没有证明这些基线依赖在后续镜像升级后仍然兼容。

## 相关文件

- [钉钉 review 文档（含 Notebook 附件）](https://alidocs.dingtalk.com/i/nodes/dxXB52LJqwlKlO2nuZ7xj03jWqjMp697)
- [Gitee 适配分支](https://gitee.com/hiyangdong/self-llm/tree/codex/local-26-musa)
- [审阅 notebook](./LOCAL-26-review.ipynb)
- [依赖 gate 日志](./local26-20261012-dependency-gate.txt)
- [GPU smoke 日志](./local26-20261012-gpu-smoke.log)
- [真实模型日志](./local26-20261012-model.log)
- [启动器日志](./local26-20261012-launcher.log)
- [单测日志](./local26-20261012-tests.log)
- [静态检查结果](./local26-20261012-static.json)
- [adapter 产物日志](./local26-20261012-artifact.log)
