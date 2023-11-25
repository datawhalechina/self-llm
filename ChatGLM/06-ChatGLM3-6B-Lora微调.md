# 06-ChatGLM3-6B-chat Lora 微调

## 概述

本节我们简要介绍如何基于 transformers、peft 等框架，对 ChatGLM3-6B-chat 模型进行 Lora 微调。Lora 是一种高效微调方法，深入了解其原理可参见博客：[知乎|深入浅出Lora](https://zhuanlan.zhihu.com/p/650197598)。

本节所讲述的代码脚本在同级目录 [ChatGLM3-6B-chat Lora 微调](./06-ChatGLM3-6B-Lora微调.py) 下，运行该脚本来执行微调过程，但注意，本文代码未使用分布式框架，微调 ChatGLM3-6B-Chat 模型至少需要 21G 及以上的显存，且需要修改脚本文件中的模型路径和数据集路径。

这个教程会在同目录下给大家提供一个 [nodebook](./06-ChatGLM3-6B-Lora微调.ipynb) 文件，来让大家更好的学习。  
鸣谢大佬的详细讲解与教程: [transformers-code](https://github.com/zyds/transformers-code)

## 环境配置

在完成基本环境配置和本地模型部署的情况下，你还需要安装一些第三方库，可以使用以下命令：

```bash
pip install transformers==4.36.0.dev0
pip install peft==0.4.0.dev0
pip install datasets==2.10.1
pip install accelerate==0.20.3
```

在本节教程里，我们将微调数据集放置在根目录 [/dataset](../dataset/huanhuan.jsonl)。

## 指令集构建

LLM 的微调一般指指令微调过程。所谓指令微调，是说我们使用的微调数据形如：

```json
{
    "instrution":"回答以下用户问题，仅输出答案。",
    "input":"1+1等于几?",
    "output":"2"
}
```

其中，`instruction` 是用户指令，告知模型其需要完成的任务；`input` 是用户输入，是完成用户指令所必须的输入内容；`output` 是模型应该给出的输出。

即我们的核心训练目标是让模型具有理解并遵循用户指令的能力。因此，在指令集构建时，我们应针对我们的目标任务，针对性构建任务指令集。例如，在本节我们使用由笔者合作开源的 [Chat-甄嬛](https://github.com/KMnO4-zx/huanhuan-chat) 项目作为示例，我们的目标是构建一个能够模拟甄嬛对话风格的个性化 LLM，因此我们构造的指令形如：

```json
{
    "instruction": "现在你要扮演皇帝身边的女人--甄嬛",
    "input":"你是谁？",
    "output":"家父是大理寺少卿甄远道。"
}
```
我们所构造的全部指令数据集在根目录下。

## 数据格式化

`Lora` 训练的数据是需要经过格式化、编码之后再输入给模型进行训练的，如果是熟悉 `Pytorch` 模型训练流程的同学会知道，我们一般需要将输入文本编码为 input_ids，将输出文本编码为 `labels`，编码之后的结果都是多维的向量。我们首先定义一个预处理函数，这个函数用于对每一个样本，编码其输入、输出文本并返回一个编码后的字典：

```python
def process_func(example):
    MAX_LENGTH = 512
    input_ids, labels = [], []
    instruction = tokenizer.encode(text="\n".join(["<|system|>", "现在你要扮演皇帝身边的女人--甄嬛", "<|user|>", 
                                    example["instruction"] + example["input"] + "<|assistant|>"]).strip() + "\n",
                                    add_special_tokens=True, truncation=True, max_length=MAX_LENGTH)

    response = tokenizer.encode(text=example["output"], add_special_tokens=False, truncation=True,
    max_length=MAX_LENGTH)

    input_ids = instruction + response + [tokenizer.eos_token_id]
    labels = [tokenizer.pad_token_id] * len(instruction) + response + [tokenizer.eos_token_id]
    pad_len = MAX_LENGTH - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    labels += [tokenizer.pad_token_id] * pad_len
    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

    return {
        "input_ids": input_ids,
        "labels": labels
    }
```

经过格式化的数据，也就是送入模型的每一条数据，都是一个字典，包含了 `input_ids`、`labels` 两个键值对，其中 `input_ids` 是输入文本的编码，`labels` 是输出文本的编码。decode之后应该是这样的：

```text
[gMASK]sop <|system|>
现在你要扮演皇帝身边的女人--甄嬛
<|user|>
这个温太医啊，也是古怪，谁不知太医不得皇命不能为皇族以外的人请脉诊病，他倒好，十天半月便往咱们府里跑。<|assistant|>
 你们俩话太多了，我该和温太医要一剂药，好好治治你们。
```

为什么会是这个形态呢？好问题！不同模型所对应的格式化输入都不一样，所以需要我们深度模型的训练源码来查看，因为按照原本模型指令微调的形式进行Lora微调效果应该是最好的，所以我们依然遵循原本模型的输入格式。OK，这里我给大家放一下源码的链接，各位如果感兴趣可以自行探索一下：

[hugging face ChatGLM3仓库](https://github.com/THUDM/ChatGLM3/blob/main/finetune_chatmodel_demo/preprocess_utils.py)：其中的`InputOutputDataset`类。  
此外，还可以参考这个仓库对ChatGLM的数据处理[LLaMA-Factory](https://github.com/KMnO4-zx/LLaMA-Factory/blob/main/src/llmtuner/data/template.py)。


## 加载tokenizer和半精度模型

模型以半精度形式加载，如果你的显卡比较新的话，可以用`torch.bfolat`形式加载。对于自定义的模型一定要指定`trust_remote_code`参数为`True`。

```python
tokenizer = AutoTokenizer.from_pretrained('./model/chatglm3-6b', use_fast=False, trust_remote_code=True)

# 模型以半精度形式加载，如果你的显卡比较新的话，可以用torch.bfolat形式加载
model = AutoModelForCausalLM.from_pretrained('./model/chatglm3-6b', trust_remote_code=True, torch_dtype=torch.half, device_map="auto")
```

## 定义LoraConfig

`LoraConfig`这个类中可以设置很多参数，但主要的参数没多少，简单讲一讲，感兴趣的同学可以直接看源码。

- `task_type`：模型类型
- `target_modules`：需要训练的模型层的名字，主要就是`attention`部分的层，不同的模型对应的层的名字不同，可以传入数组，也可以字符串，也可以正则表达式。
- `r`：`lora`的秩，具体可以看`Lora`原理
- `lora_alpha`：`Lora alaph`，具体作用参见 `Lora` 原理 
- `modules_to_save`指定的是除了拆成lora的模块，其他的模块可以完整的指定训练。

`Lora`的缩放是啥嘞？当然不是`r`（秩），这个缩放就是`lora_alpha/r`, 在这个`LoraConfig`中缩放就是4倍。
这个缩放的本质并没有改变LoRa的参数量大小,本质在于将里面的参数数值做广播乘法,进行线性的缩放。

```python
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["query_key_value"],
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
# Data collator GLM源仓库从新封装了自己的data_collator,在这里进行沿用。

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=None,
    padding=False
)

args = TrainingArguments(
    output_dir="./output/ChatGLM",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=3,
    gradient_checkpointing=True,
    save_steps=100,
    learning_rate=1e-4,
)
```

### 使用 Trainer 训练

把 model 放进去，把上面设置的参数放进去，数据集放进去，OK！开始训练！

```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=data_collator,
)
trainer.train()
```

## 模型推理

可以用这种比较经典的方式推理。

```python
model.eval()
model = model.cuda()
ipt = tokenizer("<|system|>\n现在你要扮演皇帝身边的女人--甄嬛\n<|user|>\n {}\n{}".format("你是谁？", "").strip() + "<|assistant|>\n", return_tensors="pt").to(model.device)
tokenizer.decode(model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)
```

# 从新加载
通过PEFT所微调的模型，都可以使用下面的方法进行重新加载，并推理:
- 加载源model与tokenizer；
- 使用`PeftModel`合并源model与PEFT微调后的参数。

```python
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("./model/chatglm3-6b", trust_remote_code=True, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("./model/chatglm3-6b", use_fast=False, trust_remote_code=True)

p_model = PeftModel.from_pretrained(model, model_id="./output/ChatGLM/checkpoint-1000/")  # 将训练所得的LoRa权重加载起来

ipt = tokenizer("<|system|>\n现在你要扮演皇帝身边的女人--甄嬛\n<|user|>\n {}\n{}".format("你是谁？", "").strip() + "<|assistant|>\n", return_tensors="pt").to(model.device)
tokenizer.decode(p_model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)

```
