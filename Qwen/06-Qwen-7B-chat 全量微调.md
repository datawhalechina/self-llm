# Qwen-7B-chat 全量微调

## 修改代码

首先我们要准训练模型的代码，这里我们使用的 `modelscope` 上的 `Qwen-7B-chat` 模型，大家自行下载即可。

OK，模型下载完毕之后，我们就要准备代码文件。其实全量微调和 `Lora` 微调的代码基本一样，都采用了 `Trainer` 类来进行训练。只不过在全量微调的时候没有加载 `LoraConfig`，那我就直接给出代码，如果对代有什么问题，大家可以先自行探索Qwen lora的代码解释，有什么不懂的地方可以提`Issue`。

需要把代码中的模型地址修改一下，改成自己的模型地址。

```python
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, HfArgumentParser, Trainer
import os
import torch
from dataclasses import dataclass, field
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()


@dataclass
class FinetuneArguments:
    # 微调参数
    # field：dataclass 函数，用于指定变量初始化
    model_path: str = field(default="../../model/qwen/Qwen-7B-Chat/")

# 用于处理数据集的函数
def process_func(example):
    MAX_LENGTH = 128    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["<|im_start|>system", "现在你要扮演皇帝身边的女人--甄嬛.<|im_end|>" + "\n<|im_start|>user\n" + example["instruction"] + example["input"] + "<|im_end|>\n"]).strip(), add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer("<|im_start|>assistant\n" + example["output"] + "<|im_end|>\n", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  # Qwen的特殊构造就是这样的
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


if "__main__" == __name__:
    # 解析参数
    # Parse 命令行参数
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    # 处理数据集
    # 将JSON文件转换为CSV文件
    df = pd.read_json('./data/huanhuan.json')
    ds = Dataset.from_pandas(df)
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetune_args.model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eod_id
    # 将数据集变化为token形式
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    # 创建模型并以半精度形式加载
    model = AutoModelForCausalLM.from_pretrained(finetune_args.model_path, trust_remote_code=True, torch_dtype=torch.half, device_map={"": int(os.environ.get("LOCAL_RANK") or 0)})
    
    # 使用trainer训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )
    trainer.train() # 开始训练
    response, history = model.chat(tokenizer, "你是谁", history=[], system="现在你要扮演皇帝身边的女人--甄嬛.")
    print(response)
```

## DeepSpeed 环境配置

`DeepSpeed` 是微软开源的一个深度学习训练框架，可以用于分布式训练，同时还可以加速训练，减少显存占用。这里我们使用的是 `DeepSpeed` 的半精度训练，可以减少显存占用，加快训练速度。

首先我们需要安装 `DeepSpeed`，`DeepSpeed` 的安装很简单，但如果没有按照如下步骤安装，可能会出现一些问题。

首先创建一个崭新的，干净的conda环境，注意一定要使用当前目录下提供的`environment.yml`文件来创建环境，否则可能会出现一些问题。接着激活环境，安装`deepspeed`，使用`DS_BUILD_OPS=1`来安装`deepspeed`，这样会避免后续的很多报错。

```bash
conda env create -n deepspeed -f environment.yml --force
conda activate deepspeed 
DS_BUILD_OPS=1 pip install deepspeed
```

然后就是安装`transformers`等其他依赖，注意不需要再安装`torch`了，在创建环境的时候`torch`已经安装了。

```bash
pip install modelscope==1.9.5
pip install transformers==4.35.2
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.24.1
pip install datasets sentencepiece
pip install tiktoken
pip install transformers_stream_generator
```

注意：本环境是在`aws`服务器上安装并运行的，假如您在安装或者运行过程中遇到其他问题，欢迎提出`issue`，然后您解决之后，可以顺便提交`PR`，为项目添砖加瓦。

## 模型训练

首先创建`deepspeed`的`config.json`文件。我使用的是stage-2的配置。如果不懂也没关系，直接粘贴复制，创建为`ds_config.json`文件即可。

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "last_batch_iteration": -1,
            "total_num_steps": "auto",
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },
    "activation_checkpointing": {
        "partition_activations": false,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": false,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "min_lr": 5e-7,
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```
然后我们来创建运行所需的`bash`文，创建一个`train.sh`文件，内容如下：

```shell
num_gpus=4

deepspeed --num_gpus $num_gpus train.py \
    --deepspeed ./ds_config.json \
    --output_dir="./output/Qwen" \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --logging_steps=10 \
    --num_train_epochs=3 \
    --save_steps=100 \
    --learning_rate=1e-4 \
    --save_on_each_node=True \
```

接着在命令行输入：`bash train.sh`，开始训练。

## 注意： 
    
- 因为本脚本使用了`adam_cpu`来加载优化器参数，所以全量微调所需的显存会比较小，但仍然需要使用至少4张24G显存的卡来训练。
- 如果第一步创建`deepspeed`环境时候，没有使用`DS_BUILD_OPS=1`，那么可能会出现一些问题，比如`RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`，这个时候需要重新创建环境，然后再次运行。