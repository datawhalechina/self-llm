from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, HfArgumentParser, Trainer
import os
import torch
from peft import LoraConfig, TaskType, get_peft_model
from dataclasses import dataclass, field
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()

@dataclass
class FinetuneArguments:
    # 微调参数
    # field：dataclass 函数，用于指定变量初始化
    model_path: str = field(default="./OpenBMB/miniCPM-bf32")

# 用于处理数据集的函数
def process_func(example):
    MAX_LENGTH = 512    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<用户>{example['instruction']+example['input']}<AI>", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

 # loraConfig
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "v_proj"],  # 这个不同的模型需要设置不同的参数，需要看模型中的attention层
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)


if "__main__" == __name__:
    # 解析参数
    # Parse 命令行参数
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    # 处理数据集
    # 将JSON文件转换为CSV文件
    df = pd.read_json('./huanhuan.json')
    ds = Dataset.from_pandas(df)
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetune_args.model_path, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # 将数据集变化为token形式
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    # 创建模型并以半精度形式加载
    model = AutoModelForCausalLM.from_pretrained(finetune_args.model_path, trust_remote_code=True, torch_dtype=torch.half, device_map={"": int(os.environ.get("LOCAL_RANK") or 0)})
    # model = get_peft_model(model, config)
    # 使用trainer训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )
    trainer.train() # 开始训练
    trainer.save_model() # 保存模型