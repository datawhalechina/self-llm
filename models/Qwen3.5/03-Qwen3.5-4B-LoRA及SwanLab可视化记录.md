# Qwen3.5-4B-LoRA 及 SwanLab 可视化记录

> 本教程配套 notebook：[03-Qwen3.5-4B-LoRA.ipynb](./03-Qwen3.5-4B-LoRA.ipynb)

## Qwen3.5-4B 简介

`Qwen3.5-4B` 是通义千问团队推出的新一代基础模型，具备以下核心特点：

- **高效的混合架构**：将 **Gated Delta Network（门控增量网络，一种线性注意力）** 与传统全注意力（Full Attention）层交错堆叠——每 4 层中前 3 层为线性注意力、第 4 层为全注意力。线性注意力大幅降低了长序列的推理成本与显存占用。
- **统一的多模态底座**：模型内置视觉编码器（Vision Encoder），在文本能力与 Qwen3 持平的同时，兼顾视觉理解。
- **思维链（Thinking）能力**：默认开启思考模式，在最终回答前生成 `<think> ... </think>` 包裹的推理过程；也可通过 `enable_thinking=False` 关闭，直接给出回答。
- **长上下文**：支持最长 262144（256K）上下文。

由于其架构特殊性，加载与微调 Qwen3.5-4B 需要 `transformers>=4.57`。本教程使用官方推荐的纯 `transformers + peft` 方案完成 LoRA 微调，并使用 **SwanLab** 进行训练过程可视化。

## 环境配置

```bash
# 换清华镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 核心依赖（Qwen3.5 需要 transformers>=4.57）
pip install "transformers>=4.57"
pip install accelerate datasets peft swanlab modelscope
```

### 强烈建议：为线性注意力（Gated Delta Network）安装加速算子

Qwen3.5-4B 的 32 层中有 24 层是 GDN 线性注意力。**不装加速算子时这些层会回退到纯 PyTorch 实现，训练明显偏慢**（实测 3729 条数据跑 3 个 epoch 约 80 分钟）。装上 `flash-linear-attention`（fla）后可走 Triton/tilelang 加速路径，显著提速。

```bash
# 1) fla —— Triton 实现，直接 pip 安装即可，无需编译
pip install flash-linear-attention

# 2) causal-conv1d —— 线性注意力里的 conv1d 算子，需要 nvcc 编译（首次几分钟）
#    先确认系统有 CUDA toolkit（nvcc），AutoDL 上一般在 /usr/local/cuda-12.8
export CUDA_HOME=/usr/local/cuda-12.8          # 按你的实际 cuda 路径修改
export PATH=$CUDA_HOME/bin:$PATH
pip install causal-conv1d
```

> **运行训练前同样要设好 `CUDA_HOME`**（建议写进 `~/.bashrc` 永久生效）：
> ```bash
> echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
> echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
> source ~/.bashrc
> ```
> 否则 fla 底层的 tilelang 可能调用错误的 nvcc（报 `CUDA compiler and CUDA toolkit headers are incompatible`），导致训练卡住。
>
> 若你的环境里同时装了 vLLM / SGLang（会带入 CUDA 13 的 nvidia 包），fla 的 tilelang JIT 可能因 cu12/cu13 冲突出错——**建议 LoRA 微调单独用一个干净环境**（只装 torch/transformers/peft/fla 等），不要和 vLLM/SGLang 混装。
>
> 想进一步缩短训练时间，也可把 `num_train_epochs` 从 3 降到 1～2，或先用部分数据跑通。

> 考虑到部分同学配置环境可能会遇到一些问题，我们在 AutoDL 平台准备了 Qwen3 的环境镜像，点击下方链接并直接创建 Autodl 示例即可。
> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/Qwen3***

## 模型下载

使用 modelscope 中的 `snapshot_download` 函数下载模型，第一个参数为模型名称，参数 `cache_dir` 为模型的下载路径。

新建 `model_download.py` 文件并在其中输入以下内容，粘贴代码后记得保存文件。

```python
# model_download.py
from modelscope import snapshot_download

model_dir = snapshot_download('Qwen/Qwen3.5-4B', cache_dir='/root/autodl-tmp')
print(f"模型下载完成，保存路径为：{model_dir}")
```

然后在终端中输入 `python model_download.py` 执行下载，这里需要耐心等待一段时间直到模型下载完成。

> 注意：记得修改 `cache_dir` 为你的模型下载路径哦~

## 数据集构建

对大语言模型进行 `supervised-finetuning`（`sft`，有监督微调）的数据格式如下：

```json
{
  "instruction": "回答以下用户问题，仅输出答案。",
  "input": "1+1等于几?",
  "output": "2"
}
```

其中，`instruction` 是用户指令，告知模型其需要完成的任务；`input` 是用户输入，是完成用户指令所必须的输入内容；`output` 是模型应该给出的输出。

有监督微调的目标是让模型具备理解并遵循用户指令的能力。因此，在构建数据集时，我们应针对目标任务，针对性构建数据。比如，我们的目标是通过大量人物的对话数据微调得到一个能够 role-play 甄嬛对话风格的模型，因此在该场景下的数据示例如下：

```json
{
  "instruction": "你父亲是谁？",
  "input": "",
  "output": "家父是大理寺少卿甄远道。"
}
```

本教程使用的甄嬛对话示例微调数据集位于 [/dataset/huanhuan.json](../../dataset/huanhuan.json)（共 3729 条），数据格式为 `instruction / input / output` 的 Alpaca 格式。

## 数据准备

`LoRA`（`Low-Rank Adaptation`）训练的数据需要经过格式化、编码之后再输入给模型进行训练的，我们需要先将输入文本编码为 `input_ids`，将输出文本编码为 `labels`。这里我们直接使用 tokenizer 自带的 `apply_chat_template` 方法构造对话模板，避免手写特殊 token 出错。

### 认识 Qwen3.5 的 Chat Template

`Qwen3.5` 默认开启思考模式（`enable_thinking=True`），会在回答前生成 `<think>\n ... </think>\n\n` 推理过程；对于「角色扮演」这类任务，我们通常**关闭思考模式**（`enable_thinking=False`），让模型直接给出符合角色设定的回答。

```python
from transformers import AutoTokenizer

model_id = '/root/autodl-tmp/Qwen/Qwen3.5-4B'
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
    {"role": "user", "content": "你父亲是谁？"},
    {"role": "assistant", "content": "家父是大理寺少卿甄远道。"},
]

# 关闭思考模式
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
)
print(text)
```

输出如下：

```
<|im_start|>system
现在你要扮演皇帝身边的女人--甄嬛<|im_end|>
<|im_start|>user
你父亲是谁？<|im_end|>
<|im_start|>assistant
<think>

</think>

家父是大理寺少卿甄远道。<|im_end|>
```

可以看到，关闭思考模式后，模板会自动插入一个空的 `<think>\n\n</think>\n\n` 占位，模型据此直接输出最终回答。

### 构造处理函数

我们定义一个预处理函数 `process_func`：对每条样本，分别对「前缀（system + user，带 generation prompt）」和「完整对话」进行 tokenize，再通过**token 级别的切片**精确切出 `response` 部分——这样 `labels` 中只有回答部分参与 loss 计算，system 与 user 部分（以及 `<think></think>` 占位）被置为 `-100` 屏蔽。

```python
def process_func(example):
    MAX_LENGTH = 1024  # 最大序列长度
    SYS = "现在你要扮演皇帝身边的女人--甄嬛"

    messages = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": example["instruction"] + example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    # 前缀部分（system + user，带 generation prompt），不计算 loss
    prompt_ids = tokenizer.apply_chat_template(
        messages[:2], tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False,
    )
    # 完整对话（含 assistant 回答）
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        enable_thinking=False, return_dict=False,
    )
    # token 级别切片得到 response
    response_ids = full_ids[len(prompt_ids):]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > MAX_LENGTH:  # 超长截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
```

> 说明：`transformers` 新版本中 `apply_chat_template(tokenize=True)` 默认返回 `BatchEncoding`，传入 `return_dict=False` 可直接得到 token id 列表，便于做 token 级切片。

读入数据集并应用处理函数：

```python
import json
from datasets import Dataset

with open("/root/autodl-tmp/huanhuan.json", "r", encoding="utf-8") as f:
    data = json.load(f)

ds = Dataset.from_list(data)
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
tokenized_id
```

可以解码查看处理后的样本，确认格式正确：

```python
# 查看完整输入
print(tokenizer.decode(tokenized_id[0]["input_ids"]))
# 查看 labels（过滤掉 -100 后即为模型需要学习的回答）
print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[0]["labels"]))))
```

```
<|im_start|>system
现在你要扮演皇帝身边的女人--甄嬛<|im_end|>
<|im_start|>user
小姐，别的秀女都在求中选，唯有咱们小姐想被撂牌子，菩萨一定记得真真儿的——<|im_end|>
<|im_start|>assistant
<think>

</think>

嘘——都说许愿说破是不灵的。<|im_end|>
```

## 加载模型和 tokenizer

由于 Qwen3.5-4B 是多模态模型（`Qwen3_5ForConditionalGeneration`），当我们只需要文本能力时，使用 `AutoModelForCausalLM` 加载会自动得到文本语言模型 `Qwen3_5ForCausalLM`（不加载视觉塔，显存更省）。

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    '/root/autodl-tmp/Qwen/Qwen3.5-4B',
    dtype=torch.bfloat16,
    device_map="auto",
)

# 开启梯度检查点时需要调用该方法
model.enable_input_require_grads()
model.dtype   # torch.bfloat16
```

## LoRA Config

`LoraConfig` 中比较重要的参数如下：

- `task_type`：模型类型，绝大部分 `decoder-only` 的模型都是因果语言模型 `CAUSAL_LM`
- `target_modules`：需要训练的层名
- `r`：`LoRA` 的秩，决定低秩矩阵的维度，较小的 `r` 意味着更少的参数
- `lora_alpha`：缩放参数，与 `r` 一起决定 `LoRA` 更新的强度，实际缩放比例为 `lora_alpha/r`
- `lora_dropout`：应用于 `LoRA` 层的 `dropout rate`，用于防止过拟合

> **关于 Qwen3.5 的混合架构**：模型的 32 层中，每 4 层有 3 层是线性注意力（`linear_attn`），1 层是全注意力（`self_attn`）。
> - 全注意力层包含 `q_proj / k_proj / v_proj / o_proj`
> - 线性注意力层包含 `in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / out_proj`
> - 每一层都包含 MLP：`gate_proj / up_proj / down_proj`
>
> 下面我们采用与 Qwen3 一致的目标模块 `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`——它们覆盖了所有全注意力层和**每一层的 MLP**，已经能让每一层都参与到 LoRA 训练中。如果你想更充分地适配线性注意力层，也可以把 `in_proj_qkv`、`in_proj_z`、`out_proj` 加入 `target_modules`。

```python
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,   # 训练模式
    r=8,                    # Lora 秩
    lora_alpha=32,          # Lora alpha，缩放系数 = 32/8 = 4
    lora_dropout=0.1,       # Dropout 比例
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
```

输出（仅训练约 0.25% 的参数）：

```
trainable params: 10,616,832 || all params: 4,216,368,128 || trainable%: 0.2518
```

## Training Arguments

- `output_dir`：模型输出路径
- `per_device_train_batch_size`：每张卡上的 `batch_size`
- `gradient_accumulation_steps`：梯度累计步数
- `num_train_epochs`：训练轮数

```python
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

args = TrainingArguments(
    output_dir="./output/Qwen3_5_4B_LoRA",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)
```

## SwanLab 简介

![](./images/05-2.png)

[SwanLab](https://github.com/swanhubx/swanlab) 是一个开源的模型训练记录工具，面向 AI 研究者，提供了训练可视化、自动日志记录、超参数记录、实验对比、多人协同等功能。在 `SwanLab` 上，研究者能基于直观的可视化图表发现训练问题，对比多个实验找到研究灵感，并通过在线链接的分享与基于组织的多人协同训练，打破团队沟通的壁垒。

**为什么要记录训练**

相较于软件开发，模型训练更像一门实验科学。一个品质优秀的模型背后，往往是成千上万次实验。研究者需要不断尝试、记录、对比，积累经验，才能找到最佳的模型结构、超参数与数据配比。在这之中，如何高效进行记录与对比，对于研究效率的提升至关重要。

## 实例化 SwanLabCallback

建议先在 [SwanLab 官网](https://swanlab.cn/) 注册账号，然后在初始化阶段选择 `(2) Use an existing SwanLab account` 并使用 private API Key 登录。

```python
import swanlab
from swanlab.integration.transformers import SwanLabCallback

# 首次使用会提示登录，输入你在 SwanLab 官网获取的 API Key
swanlab_callback = SwanLabCallback(
    project="Qwen3.5-Lora",
    experiment_name="Qwen3.5-4B-LoRA",
)
```

## 使用 Trainer 训练

```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()
```

训练完成后，打开 SwanLab 即可查看训练过程中记录的参数与 loss 曲线：

![](./images/05-1.png)

## 加载 LoRA 权重推理

得到 checkpoint 之后，加载基础模型并挂载 LoRA 权重进行推理：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_id = '/root/autodl-tmp/Qwen/Qwen3.5-4B'                     # 基础模型路径
lora_path = './output/Qwen3_5_4B_LoRA/checkpoint-XXX'             # 训练得到的 LoRA 权重路径，按实际填写

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

# 挂载 LoRA 权重
model = PeftModel.from_pretrained(model, model_id=lora_path)
model.eval()

# 构造对话
messages = [
    {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
    {"role": "user", "content": "你是谁？"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,          # 关闭思考模式，直接输出角色回答
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

gen_kwargs = {"max_new_tokens": 128, "do_sample": True, "top_p": 0.8, "temperature": 0.7}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
outputs = outputs[:, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

输出示例：

```
我是甄嬛，家父是大理寺少卿甄远道。
```

可以看到，经过 LoRA 微调后，模型已经学会了甄嬛的说话风格与人物设定。
