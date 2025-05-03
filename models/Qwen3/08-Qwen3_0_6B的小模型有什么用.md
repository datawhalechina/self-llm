# 大模型都这么厉害了，还需要微调吗？0.6B的小模型还有什么意义吗？

大家在日常使用Deepseek-R1或者是阿里新发布的Qwen3模型，他们的模型都是能力很强，所提供的API服也都可以满足大家的日常或者是公司开发所需。但大家也可以想一个简单的问题几个简单的问题，如下：

1. 公司的数据是够敏感，是否需要保密？
1. 日常使用大模型的任务是否很困难，对推理链是否刚需？
1. 任务调用的大模型API并发量是多少？每日资金消耗有多少？

对于问题1，如果公司数据敏感，那我建议不要调用供应商提供的大模型API。就算供应商保证不会拿你们数据做训练，但你们的数据还是泄漏了（会有不必要的风险），建议本地部署大模型。

对于问题2，如果使用大模型的场景问题很困难并且刚需推理链，那可以使用供应商的API，这样可以保证推理链的上下文不会爆显存。如果问题很简单，没有刚需推理链，那建议本地部署小模型即可。

对于问题3，如果任务很简单，且调用的大模型API并发量很高，那我建议微调一个特定任务的小模型，本地部署。这样可以满足高并发，并且可以减少资金消耗。（本地部署，默认硬件环境单卡4090）

看到这里，想必大家已经思考完了以上三个问题，心中有了答案。那我给出一个小小的案例。

## 微调模型的需求性

假如你的公司有一个从投诉的文本中抽取用户信息的任务。比如，你需要从以下文本中抽取用户姓名、住址、邮箱、投诉的问题等等。

> 这只是一个小小的案例，数据也是我用大模型批量制造的。真正的投诉数据不会这么“干净、整洁”。

INPUT：
```text
龙琳，宁夏回族自治区璐市城东林街g座 955491，邮箱 nafan@example.com。小区垃圾堆积成山，晚上噪音扰人清梦，停车难上加难，简直无法忍受！
```

OUTPUT：
```json
{
    "name": "龙琳",
    "address": "宁夏回族自治区璐市城东林街g座 955491",
    "email": "nafan@example.com",
    "question": "小区垃圾堆积成山，晚上噪音扰人清梦，停车难上加难，简直无法忍受！"
}
```


那你当然可以调用 Deepseek最强大的模型R1，也可以调用阿里最新发布最强大的模型 Qwen3-235B-A22B等等，这些模型的信息抽取效果也很非常的棒。

但有个问题，如果你有几百万条这样的数据要处理，全部调用最新的，最好的大模型可能需要消耗几万块钱。并且，如果这些投诉数据，比如电信投诉数据，电网投诉数据，这些数据是敏感的不可以直接放到外网的。

所以，综合数据敏感，和资金消耗。最好的选择就是微调一个小模型（如Qwen3-0.6B），既可以保证高并发，可以保证数据不泄漏，保证模型抽取的效果，还可以省钱！！！

那下面，用一个小案例带大家实操一下，微调Qwen3-0.6B小模型完成文本信息抽取任务。



## 配置环境 下载数据

> Colab 文件地址：https://colab.research.google.com/drive/18ByY11KVhIy6zWx1uKUjSzqeHTme-TtU?usp=drive_link

```python
!pip install datasets swanlab -q
```

```python
!wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1a0sf5C209CLW5824TJkUM4olMy0zZWpg' -O fake_sft.json
```

## 处理数据

```python
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
import torch
```

```python
# 将JSON文件转换为CSV文件
df = pd.read_json('fake_sft.json')
ds = Dataset.from_pandas(df)
ds[:3]
```

```python
model_id = "Qwen/Qwen3-0.6B"
```

```python
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer
```

对大语言模型进行 `supervised-finetuning`（`sft`，有监督微调）的数据格式如下：

```json
{
  "instruction": "回答以下用户问题，仅输出答案。",
  "input": "1+1等于几?",
  "output": "2"
}
```

其中，`instruction` 是用户指令，告知模型其需要完成的任务；`input` 是用户输入，是完成用户指令所必须的输入内容；`output` 是模型应该给出的输出。

有监督微调的目标是让模型具备理解并遵循用户指令的能力。因此，在构建数据集时，我们应针对我们的目标任务，针对性构建数据。比如，如果我们的目标是通过大量人物的对话数据微调得到一个能够 role-play 甄嬛对话风格的模型，因此在该场景下的数据示例如下：

```json
{
  "instruction": "你父亲是谁？",
  "input": "",
  "output": "家父是大理寺少卿甄远道。"
}
```

`Qwen3` 采用的 `Chat Template`格式如下：

由于 `Qwen3` 是混合推理模型，因此可以手动选择开启思考模式

不开启 `thinking mode`


```python
messages = [
    {"role": "system", "content": "You are a helpful AI"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm fine, think you. and you?"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
print(text)
```
```
<|im_start|>system
You are a helpful AI<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
<think>

</think>

I'm fine, think you. and you?<|im_end|>
<|im_start|>assistant
<think>

</think>
``` 


`LoRA`（`Low-Rank Adaptation`）训练的数据是需要经过格式化、编码之后再输入给模型进行训练的，我们需要先将输入文本编码为 `input_ids`，将输出文本编码为 `labels`，编码之后的结果是向量。我们首先定义一个预处理函数，这个函数用于对每一个样本，同时编码其输入、输出文本并返回一个编码后的字典：


```python
def process_func(example):
    MAX_LENGTH = 1024 # 设置最大序列长度为1024个token
    input_ids, attention_mask, labels = [], [], [] # 初始化返回值
    # 适配chat_template
    instruction = tokenizer(
        f"<s><|im_start|>system\n{example['system']}<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    # 将instructio部分和response部分的input_ids拼接，并在末尾添加eos token作为标记结束的token
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    # 注意力掩码，表示模型需要关注的位置
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    # 对于instruction，使用-100表示这些位置不计算loss（即模型不需要预测这部分）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 超出最大序列长度截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

```


```python
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
tokenized_id
```


```python
tokenizer.decode(tokenized_id[0]['input_ids'])
```

```python
tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))
```

## 加载模型

加载模型并配置LoraConfig

```python
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",torch_dtype=torch.bfloat16)
model
```


```
Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): Embedding(151936, 1024)
    (layers): ModuleList(
      (0-27): 28 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
          (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MLP(
          (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
        (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
      )
    )
    (norm): Qwen3RMSNorm((1024,), eps=1e-06)
    (rotary_emb): Qwen3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
)
```

```python
model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
```


## Lora Config

`LoraConfig`这个类中可以设置很多参数，比较重要的如下

- `task_type`：模型类型，现在绝大部分 `decoder_only` 的模型都是因果语言模型 `CAUSAL_LM`
- `target_modules`：需要训练的模型层的名字，主要就是 `attention`部分的层，不同的模型对应的层的名字不同
- `r`：`LoRA` 的秩，决定了低秩矩阵的维度，较小的 `r` 意味着更少的参数
- `lora_alpha`：缩放参数，与 `r` 一起决定了 `LoRA` 更新的强度。实际缩放比例为`lora_alpha/r`，在当前示例中是 `32 / 8 = 4` 倍
- `lora_dropout`：应用于 `LoRA` 层的 `dropout rate`，用于防止过拟合


```python
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
config
```


```python
model = get_peft_model(model, config)
config
```


```python
model.print_trainable_parameters()  # 模型参数训练量只有0.8395%
```

> trainable params: 5,046,272 || all params: 601,096,192 || trainable%: 0.8395


## Training Arguments

- `output_dir`：模型的输出路径
- `per_device_train_batch_size`：每张卡上的 `batch_size`
- `gradient_accumulation_steps`: 梯度累计
- `num_train_epochs`：顾名思义 `epoch`


```python
args = TrainingArguments(
    output_dir="Qwen3_instruct_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=1,
    num_train_epochs=3,
    save_steps=50,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)
```

## SwanLab 简介

[SwanLab](https://github.com/swanhubx/swanlab) 是一个开源的模型训练记录工具，面向 AI 研究者，提供了训练可视化、自动日志记录、超参数记录、实验对比、多人协同等功能。在 `SwanLab` 上，研究者能基于直观的可视化图表发现训练问题，对比多个实验找到研究灵感，并通过在线链接的分享与基于组织的多人协同训练，打破团队沟通的壁垒。

**为什么要记录训练**

相较于软件开发，模型训练更像一个实验科学。一个品质优秀的模型背后，往往是成千上万次实验。研究者需要不断尝试、记录、对比，积累经验，才能找到最佳的模型结构、超参数与数据配比。在这之中，如何高效进行记录与对比，对于研究效率的提升至关重要。

`(2) Use an existing SwanLab account` 并使用 private API Key 登录

```python
import swanlab
from swanlab.integration.transformers import SwanLabCallback

# 实例化SwanLabCallback
swanlab_callback = SwanLabCallback(
    project="Qwen3-Lora",  # 注意修改
    experiment_name="Qwen3-8B-LoRA-experiment"  # 注意修改
)
```


```python
import swanlab
from swanlab.integration.transformers import SwanLabCallback

# 实例化SwanLabCallback
swanlab_callback = SwanLabCallback(
    project="Qwen3-Lora",
    experiment_name="Qwen3-0.6B-extarct-lora-2"
)
```


```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback]
)
```


```python
trainer.train()
```


## 测试文本


```python
prompt = "龙琳   ，宁夏回族自治区璐市城东林街g座 955491，nafan@example.com。小区垃圾堆积成山，晚上噪音扰人清梦，停车难上加难，简直无法忍受！太插件了阿萨德看见啊啥的健康仨都会撒娇看到撒谎的、"

messages = [
    {"role": "system", "content": "将文本中的name、address、email、question提取出来，以json格式输出，字段为name、address、email、question，值为文本中提取出来的内容。"},
    {"role": "user", "content": prompt}
]

inputs = tokenizer.apply_chat_template(messages,
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True,
                                       enable_thinking=False).to('cuda')

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


```json
{
    "name": "龙琳",
    "address": "宁夏回族自治区璐市城东林街g座 955491",
    "email": "nafan@example.com",
    "question": "小区垃圾堆积成山，晚上噪音扰人清梦，停车难上加难，简直无法忍受！太插件了阿萨德看见啊啥的健康仨都会撒娇看到撒谎的、"
}
```
