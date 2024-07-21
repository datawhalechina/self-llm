# XVERSE-7B-Chat Lora 微调

## 概述

本节我们简要介绍如何基于 transformers、peft 等框架，对 XVERSE-7B-Chat 模型进行 Lora 微调。Lora 是一种高效微调方法，深入了解其原理可参见博客：[知乎|深入浅出Lora](https://zhuanlan.zhihu.com/p/650197598)。

这个教程会在同目录下给大家提供一个 [notebook](./05-XVERSE-7B-Chat%20Lora%20微调.ipynb) 文件，来让大家更好的学习。

## 环境配置

在完成基本环境配置和本地模型部署的情况下，你还需要安装一些第三方库，为了方便大家实践，我将环境打包放在 code 文件夹下了，可以使用以下命令：

```bash
cd code
pip install -r requirement.txt
```

在本节教程里，我们将微调数据集放置在根目录 [/dataset](https://github.com/datawhalechina/self-llm/blob/master/dataset/huanhuan.json)。

## 指令集构建

LLM 的微调一般指指令微调过程。所谓指令微调，是说我们使用的微调数据形如：

```json
{
  "instruction": "解释什么是人工智能。\n",
  "input": "",
  "output": "人工智能是一种利用计算机程序和算法创造出类似人类智能的技术，可以让计算机在解决问题、学习、推理和自然语言处理等方面表现出类似人类的能力。"
}
```

其中，`instruction` 是用户指令，告知模型其需要完成的任务；`input` 是用户输入，是完成用户指令所必须的输入内容；`output` 是模型应该给出的输出。而在 XVERSE 中数据的目标格式是这样的

```json
{
    "inputs": "Human:解释什么是人工智能。\n Assistant:", 
    "targets": "人工智能是一种利用计算机程序和算法创造出类似人类智能的技术，可以让计算机在解决问题、学习、推理和自然语言处理等方面表现出类似人类的能力。"}
```

## 数据格式化

`Lora` 训练的数据是需要经过格式化、编码之后再输入给模型进行训练的，如果是熟悉 `Pytorch` 模型训练流程的同学会知道，我们一般需要将输入文本编码为 input_ids，将输出文本编码为 `labels`，编码之后的结果都是多维的向量。我们首先定义一个预处理函数，这个函数用于对每一个样本，编码其输入、输出文本并返回一个编码后的字典：

```python
def process_func(example):
        MAX_LENGTH = 384
        input_ids = []
        labels = []

        instruction = tokenizer(text=f"Human:现在你要扮演皇帝身边的女人--甄嬛\n\n {example['instruction']}{example['input']}Assistant:", add_special_tokens=False)
        response = tokenizer(text=f"{example['output']}", add_special_tokens=False)
        input_ids = [tokenizer.bos_token_id] + instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
        labels = [tokenizer.bos_token_id] + [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        return {
            "input_ids": input_ids,
            "labels": labels
        }
```

经过格式化的数据，也就是送入模型的每一条数据，都是一个字典，包含了 `input_ids`、`labels` 两个键值对，其中 `input_ids` 是输入文本的编码，`labels` 是输出文本的编码。decode之后应该是这样的：

```json
'<|startoftext|>Human:现在你要扮演皇帝身边的女人--甄嬛\n\n 这个温太医啊，也是古怪，谁不知太医不得皇命不能为皇族以外的人请脉诊病，他倒好，十天半月便往咱们府里跑。Assistant:你们俩话太多了，我该和温太医要一剂药，好好治治你们。<|endoftext|>'
```

为什么会是这个形态呢？好问题！不同模型所对应的格式化输入都不一样，因为在 XVERSE 中它的template是这样的：`["Human: {{content}}\n\nAssistant: "]`，所有自然而然格式就是这样的，并且 XVERSE 的文本起始token和结束token也不一样。

## 加载tokenizer和模型

```python
import torch

model = AutoModelForCausalLM.from_pretrained('xverse/XVERSE-7B-Chat', trust_remote_code=True, torch_dtype=torch.half, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained('xverse/XVERSE-7B-Chat')
```

## 定义LoraConfig



`LoraConfig`这个类中可以设置很多参数，但主要的参数没多少，简单讲一讲，感兴趣的同学可以直接看源码。

- `task_type`：模型类型
- `target_modules`：需要训练的模型层的名字，主要就是`attention`部分的层，不同的模型对应的层的名字不同，可以传入数组，也可以字符串，也可以正则表达式。
- `r`：`lora`的秩，具体可以看`Lora`原理
- `lora_alpha`：`Lora alaph`，具体作用参见 `Lora` 原理

`Lora`的缩放是啥嘞？当然不是`r`（秩），这个缩放就是`lora_alpha/r`, 在这个`LoraConfig`中缩放就是4倍。

```python
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["c_attn", "c_proj", "w1", "w2"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
```

## 自定义 TrainingArguments 参数

`TrainingArguments`这个类的源码也介绍了每个参数的具体作用，当然大家可以来自行探索，这里就简单说几个常用的。

- `output_dir`：模型的输出路径
- `per_device_train_batch_size`：顾名思义 `batch_size`
- `gradient_accumulation_steps`: 梯度累加，如果你的显存比较小，那可以把 `batch_size` 设置小一点，梯度累加增大一些。
- `logging_steps`：多少步，输出一次`log`
- `num_train_epochs`：顾名思义 `epoch`
- `gradient_checkpointing`：梯度检查，这个一旦开启，模型就必须执行`model.enable_input_require_grads()`，这个原理大家可以自行探索，这里就不细说了。

```python
args = TrainingArguments(
    output_dir="./output/BlueLM",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=3,
    gradient_checkpointing=True,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True
)
```

## 使用 Trainer 训练

把 model 放进去，把上面设置的参数放进去，数据集放进去，OK！开始训练！

```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()
```

## 模型推理

使用最常用的方式进行推理:
> 注意将`return_token_type_ids`调为false

```python
model.eval()
text = "小姐，别的秀女都在求中选，唯有咱们小姐想被撂牌子，菩萨一定记得真真儿的——"
inputs = tokenizer(f"Human:{text} Assistant:", return_tensors="pt", return_token_type_ids=False, )
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

完整代码请看：[XVERSE-7B-Chat Lora 微调](./05-XVERSE-7B-Chat%20Lora%20微调.py)
