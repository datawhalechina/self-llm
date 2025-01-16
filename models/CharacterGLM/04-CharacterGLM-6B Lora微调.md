# 04-CharacterGLM-6B-Chat Lora微调

## 概述

本文简要介绍如何基于transformers、peft等框架，对CharacterGLM-6B-chat模型进行Lora微调。Lora原理可参考博客：[知乎|深入浅出Lora](https://zhuanlan.zhihu.com/p/650197598)
本文代码未使用分布式框架，微调 ChatGLM3-6B-Chat 模型至少需要 21G 及以上的显存，且需要修改脚本文件中的模型路径和数据集路径。

## 环境配置

在完成基本环境配置和本地模型部署的情况下，还需要安装一些第三方库，可以使用如下命令：

```python
pip install transformers==4.37.2
pip install peft==0.4.0.dev0
pip install datasets==2.10.1
pip install accelerate==0.21.0

```

在本节内容中，将微调数据集放置在根目录[/dataset](https://github.com/datawhalechina/self-llm/blob/master/dataset/huanhuan.json)。

## 指令集构建

LLM微调一般指指令微调过程。所谓指令微调，是说我们使用的微调数据形如：

```python
{
    "instruction":"回答用户以下问题，直接给出结果。"
    "input":"中国第一个诺贝尔奖得主是谁？"
    "output":"莫言"
}
```

其中instruction是用户指令，告知模型需要完成的任务；input是用户输入，是完成用户指令所必需的输入内容；output是模型应该给出的输出。

即我们的核心训练目标是让模型具有理解并遵循用户指令的能力。因此，在指令集构建时，我们应针对我们的目标任务，针对性构建任务指令集。在本文我们使用由笔者合作开源的[Chat-甄嬛项目](https://github.com/KMnO4-zx/huanhuan-chat)作为示例，我们的目标是构建一个能够模拟甄嬛对话风格的个性化LLM，因此我们构建的指令形如：

```python
{
    "instruction": "",
    "input":"你是谁？",
    "output":"家父是大理寺少卿甄远道。"
}
```

我们构造的全部指令数据集在根目录下。

## QA和Instruction的区别和联系

QA是指一问一答的形式，通常是用户提问，模型给出回答。而instruction则源自于Prompt Engineering，将问题拆分成两个部分：Instruction用于描述任务，Input用于描述待处理的对象。

问答(QA)格式的训练数据通常用于训练模型执行具体任务。例如，对于问题“请解释INFJ和ENTP两种MBTI性格之间的区别”

*问答(QA)格式：

```python
指令(instruction)：
输入(input)：INFJ和ENTP这两种MBTI性格之间的区别是什么？
```

*指令(Instruction)格式：

```python
指令(Instruction):请解释下面两种MBTI性格的区别
输入(input):INFJ和ENTP
```

## 数据格式化

Lora训练的数据是需要经过格式化、编码之后再输入给模型进行训练的，我们一般需要将输入文本编码为input_ids，将输出文本编码为labels,编码之后的结果都是多维向量。我们首先定义一个与处理函数，这个函数用于对每一个样本，编码其输入，输出文本并返回一个编码后的字典：

```python
def process_func(example):
    MAX_LENGTH = 512
    input_ids, labels = [], []
    prompt = tokenizer.encode("用户:\n"+"现在你要扮演皇帝身边的女人--甄嬛。", add_special_tokens=False)
    instruction_ = tokenizer.encode("\n".join([example["instruction"], example["input"]]).strip(), add_special_tokens=False,max_length=512)
    instruction = tokenizer.encode(prompt + instruction_)
    response = tokenizer.encode("CharacterGLM-6B:\n:" + example["output"], add_special_tokens=False)
    input_ids = instruction + response + [tokenizer.eos_token_id]
    labels = [tokenizer.pad_token_id] * len(instruction) + response + [tokenizer.eos_token_id]
    pad_len = MAX_LENGTH - len(input_ids)
    # print()
    input_ids += [tokenizer.pad_token_id] * pad_len
    labels += [tokenizer.pad_token_id] * pad_len
    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

    return {
        "input_ids": input_ids,
        "labels": labels
    }
```

经过格式化的数据，也就是送入模型的每一条数据，都是一个字典，包含了input_ids、labels两个键值对，其中input_ids是输入文本的编码，labels是输出文本的编码。

## 加载tokenizer和半精度模型

模型以版精度形式加载，如果显卡比较新，可以用torch.bfloat形式加载，对于自定义的模型一定要指定trust_remote_code参数为True

```python
tokenizer=AutoTokenizer.from_pretrained('/root/autodl-tmp/THUCoAI/CharacterGLM-6B',use_fast=False,trust_remote_code=True)

model=AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/THUCoAI/CharacterGLM-6B',trust_remote_code=True,torch_dtype=torch.half,device_map="auto")
```

## 定义LoraConfig

LoraConfig这个类中可以设置很多参数，部分参数展示如下：
task_type:模型类型
target——modules：需要训练的模型层的名字，主要就是attention部分的层，不同的模型对应的层的名字不同，可以传入数组，也可以字符串，也可以正则表达式。
r:lora的秩
lora_alpha:Lora alpha
modules_to_save:指定的是除了拆成lora的模块，其它的模块可以完整的指定训练

Lora的所方式lora_alpha/r，在这个LoraConfig中缩放就是4倍。这个缩放的本质并没有改变Lora的参数量大小，本质在于将里面的参数数值做广播乘法，进行线性的缩放。

```python
config=LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["query_key_value"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
```

## 自定义TraininArguments参数

TrainingArguments这个类的源码也介绍了每个参数的具体作用，常用的参数如下：
output_dir:模型的输出路径
per_device_train_batch_size:batch_size
gradient_accumulation_steps:梯度累加，如果显存比较小，可以把batch_size设置小一点，梯度累积增大一点
logging_steps:多少步，输出一次log
num_train_epochs:顾名思义epoch
gradient_chechpointing:梯度检查，这个一旦开启，模型就必须执行
model.enable_input_require_grads()

```python
data_collator=DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=None,
    padding=False
)
args=TrainingArguments(
    output_dir="./output/CharacterGLM",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=3,
    gradient_checkpointing=True,
    save_steps=100,
    learning_rate=1e-4,
)
```

## 使用Trainer训练

把model放进去，把上面设置的参数放进去，数据集放进去，开始训练

```python
trainer=Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=data_collator,
)
trainer.train()
```

## 模型推理

```python
model = model.cuda()
ipt = tokenizer("用户：{}\n{}".format("现在你要扮演皇帝身边的女人--甄嬛。你是谁？", "").strip() + "characterGLM-6B:\n", return_tensors="pt").to(model.device)
tokenizer.decode(model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)
```

## 从新加载

通过PEFT所微调的模型，都可以使用下面的方法进行重新加载，并推理：

加载源model与tokenizer；
使用PeftModel合并源model与PEFT微调后的参数

```python
from peft import Peftmodel
model=AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/THUCoAI/CharacterGLM-6B",trust_remote_code=True,low_cpu_mem_usage=True)
tokenizer=AutoTokenizer.from_pretrained("root/autodl-tmp/THUCoAI/CharacterGLM-6B",use_fast=False,trust_remote_code=True)
p_model=PeftModel.from_pretrained(model,model_id="./output/CharatcerGLM/checkpoint-1000/")
ipt = tokenizer("用户：{}\n{}".format("现在你要扮演皇帝身边的女人--甄嬛。你是谁？", "").strip() + "characterGLM-6B:\n", return_tensors="pt").to(model.device)
tokenizer.decode(p_model.generate(**ipt,max_length=128,do_sample=True)[0],skip_special_tokens=True)
```
