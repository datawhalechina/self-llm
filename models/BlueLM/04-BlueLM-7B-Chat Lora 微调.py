from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model




def process_func(example):
    MAX_LENGTH = 384
    input_ids = []
    labels = []

    instruction = tokenizer(
        text=f"[|Human|]:现在你要扮演皇帝身边的女人--甄嬛\n\n {example['instruction']}{example['input']}[|AI|]:",
        add_special_tokens=False)
    response = tokenizer(text=f"{example['output']}", add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    labels = [tokenizer.bos_token_id] + [-100] * len(instruction["input_ids"]) + response["input_ids"] + [
        tokenizer.eos_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "labels": labels
    }

# lora配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)

# 训练参数
args = TrainingArguments(
    output_dir="./output/BlueLM",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)

if __name__ == '__main__':
    # 将JSON文件转换为CSV文件
    df = pd.read_json('./huanhuan.json')
    ds = Dataset.from_pandas(df)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained('vivo-ai/BlueLM-7B-Chat', use_fast=False, trust_remote_code=True)

    # 将数据集变化为token形式
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    # 创建模型
    model = AutoModelForCausalLM.from_pretrained('vivo-ai/BlueLM-7B-Chat', trust_remote_code=True,
                                                 torch_dtype=torch.half, device_map="auto")

    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    # 模型合并
    model = get_peft_model(model, config)
    # 使用trainer训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()  # 开始训练
