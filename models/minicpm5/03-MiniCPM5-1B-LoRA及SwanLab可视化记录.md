# MiniCPM5-1B-LoRA 及 SwanLab 可视化记录

> 本教程配套 notebook：[03-MiniCPM5-1B-LoRA.ipynb](./03-MiniCPM5-1B-LoRA.ipynb)

## MiniCPM5-1B 简介

`MiniCPM5-1B` 是面壁智能（ModelBest）/ OpenBMB 发布的 1B 稠密 Transformer，采用**标准 `LlamaForCausalLM` 架构**（24 层，GQA，128K 上下文）。它内置 `<think>` chat template，支持「思考 / 非思考」双模式（`enable_thinking` 切换），并原生支持工具调用。1B 的体量非常适合在单卡上做 LoRA 微调实验。

本教程使用官方推荐的纯 `transformers + peft` 方案完成 LoRA 微调，并使用 **SwanLab** 记录训练过程。

## 环境配置

```bash
# 换清华镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 核心依赖（MiniCPM5 需要 transformers>=5.6）
pip install "transformers>=5.6"
pip install accelerate datasets peft swanlab modelscope
```

> 考虑到部分同学配置环境可能会遇到一些问题，我们在 AutoDL 平台准备了环境镜像，点击下方链接并直接创建 Autodl 示例即可。
> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/MiniCPM5***

## 模型下载

```python
# model_download.py
from modelscope import snapshot_download

model_dir = snapshot_download('OpenBMB/MiniCPM5-1B', cache_dir='/root/autodl-tmp')
print(f"模型下载完成，保存路径为：{model_dir}")
```

然后在终端中输入 `python model_download.py` 执行下载。

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

其中，`instruction` 是用户指令；`input` 是用户输入；`output` 是模型应该给出的输出。

我们的目标是通过大量人物对话数据微调得到一个能够 role-play 甄嬛对话风格的模型，数据示例如下：

```json
{
  "instruction": "你父亲是谁？",
  "input": "",
  "output": "家父是大理寺少卿甄远道。"
}
```

本教程使用的甄嬛对话示例微调数据集位于 [/dataset/huanhuan.json](../../dataset/huanhuan.json)（共 3729 条），数据格式为 `instruction / input / output` 的 Alpaca 格式。

## 数据准备

LoRA 训练的数据需要经过格式化、编码之后再输入给模型。这里我们直接使用 tokenizer 自带的 `apply_chat_template` 构造对话模板。

### 认识 MiniCPM5 的 Chat Template

`MiniCPM5-1B` 采用 `<|im_start|>role\n...<|im_end|>\n` 格式，并支持 `enable_thinking` 参数控制思考模式。对于「角色扮演」任务，我们关闭思考模式（`enable_thinking=False`）：

```python
from transformers import AutoTokenizer

model_id = '/root/autodl-tmp/OpenBMB/MiniCPM5-1B'
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
    {"role": "user", "content": "你父亲是谁？"},
    {"role": "assistant", "content": "家父是大理寺少卿甄远道。"},
]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
)
print(text)
```

输出如下：

```
<s><|im_start|>system
现在你要扮演皇帝身边的女人--甄嬛<|im_end|>
<|im_start|>user
你父亲是谁？<|im_end|>
<|im_start|>assistant
家父是大理寺少卿甄远道。<|im_end|>
```

### 构造处理函数

> **注意一个细节**：MiniCPM5 的模板在「带 generation prompt」时会追加 `<think>\n\n</think>\n\n`（非思考模式占位），但「完整对话渲染」时助手回合并**不**包含这个 think 块。因此这里**不能用 token 级切片**（`full[len(prompt):]`），而要分别对「前缀」和「回答」单独 tokenize 再拼接。

```python
def process_func(example):
    MAX_LENGTH = 1024
    SYS = "现在你要扮演皇帝身边的女人--甄嬛"

    messages = [{"role": "system", "content": SYS},
                {"role": "user", "content": example["instruction"] + example["input"]}]
    # 前缀（system + user，带 generation prompt，含非思考 think 占位），不计算 loss
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False,
    )
    # 回答部分：output + 结束符 <|im_end|>
    response_ids = tokenizer(example["output"], add_special_tokens=False).input_ids \
                   + [tokenizer.convert_tokens_to_ids("<|im_end|>")]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > MAX_LENGTH:  # 超长截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
```

> 说明：MiniCPM5 的 `<|im_end|>` token id 为 `130073`，`</s>`（eos）为 `1`。这里用 `<|im_end|>` 作为回答的结束符，与模板一致。

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

可以解码查看处理后的样本：

```python
print(tokenizer.decode(tokenized_id[0]["input_ids"]))
print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[0]["labels"]))))
```

```
<s><|im_start|>system
现在你要扮演皇帝身边的女人--甄嬛<|im_end|>
<|im_start|>user
小姐，别的秀女都在求中选，唯有咱们小姐想被撂牌子，菩萨一定记得真真儿的——<|im_end|>
<|im_start|>assistant
<think>

</think>

嘘——都说许愿说破是不灵的。<|im_end|>
```

`labels`（过滤掉 -100）：

```
嘘——都说许愿说破是不灵的。<|im_end|>
```

## 加载模型和 tokenizer

`MiniCPM5-1B` 是标准 `LlamaForCausalLM`，直接用 `AutoModelForCausalLM` 加载：

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    '/root/autodl-tmp/OpenBMB/MiniCPM5-1B',
    dtype=torch.bfloat16,
    device_map="auto",
)
model.enable_input_require_grads()   # 开启梯度检查点时需要
model.dtype   # torch.bfloat16
```

## LoRA Config

`MiniCPM5-1B` 是标准 Llama 架构，LoRA 目标模块与 Llama 一致：`q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj`。

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

输出（仅训练约 0.5% 的参数）：

```
trainable params: 5,603,328 || all params: 1,086,236,160 || trainable%: 0.5158
```

## Training Arguments

```python
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

args = TrainingArguments(
    output_dir="./output/MiniCPM5_1B_LoRA",
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

![](./images/swanlab.png)

[SwanLab](https://github.com/swanhubx/swanlab) 是一个开源的模型训练记录工具，提供训练可视化、自动日志记录、超参数记录、实验对比、多人协同等功能。

**为什么要记录训练**：模型训练更像一门实验科学，一个优秀模型背后往往是成千上万次实验。高效记录与对比对研究效率至关重要。

## 实例化 SwanLabCallback

建议先在 [SwanLab 官网](https://swanlab.cn/) 注册账号，初始化时选择 `(2) Use an existing SwanLab account` 并使用 private API Key 登录。

```python
import swanlab
from swanlab.integration.transformers import SwanLabCallback

swanlab_callback = SwanLabCallback(
    project="MiniCPM5-Lora",
    experiment_name="MiniCPM5-1B-LoRA",
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

训练完成后，打开 SwanLab 即可查看训练过程中记录的参数与 loss 曲线。

## 加载 LoRA 权重推理

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_id = '/root/autodl-tmp/OpenBMB/MiniCPM5-1B'
lora_path = './output/MiniCPM5_1B_LoRA/checkpoint-XXX'   # 按实际 checkpoint 填写

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, model_id=lora_path)
model.eval()

messages = [
    {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
    {"role": "user", "content": "你是谁？"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

gen_kwargs = {"max_new_tokens": 128, "do_sample": True, "top_p": 0.95, "temperature": 0.7}
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
