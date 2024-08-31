from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

import torch
from peft import LoraConfig, TaskType, get_peft_model


# # 用于处理数据集的函数
def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["<|im_start|>system", "现在你要扮演皇帝身边的女人--甄嬛.<|im_end|>" + "\n<|im_start|>user\n" + example["instruction"] + example["input"] + "<|im_end|>\n"]).strip(), add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer("<|im_start|>assistant\n" + example["output"] + "<|im_end|>\n", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  # Yi-6B的特殊构造就是这样的
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# #  配置lora Config
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_attn", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)


# 配置训练参数
args = TrainingArguments(
    output_dir="./output/Yi-6B",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=3,
    gradient_checkpointing=True,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True
)


if "__main__" == __name__:
    # 将JSON文件转换为CSV文件
    df = pd.read_json('/root/dataset/huanhuan.json')
    ds = Dataset.from_pandas(df)
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained('01ai/Yi-6B-Chat', use_fast=False, trust_remote_code=True)
    # # 将数据集变化为token形式
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
    
    # 创建模型并以半精度形式加载，如果你的显卡比较新的话，可以用torch.bfolat形式加载
    model = AutoModelForCausalLM.from_pretrained('01ai/Yi-6B-Chat', trust_remote_code=True, torch_dtype=torch.half, device_map="auto")
    model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
    # 加载lora参数
    model = get_peft_model(model, config)
    
    # 使用trainer训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )
    trainer.train() # 开始训练
    
    # 模型推理
    model.eval()
    input = tokenizer("<|im_start|>system\n现在你要扮演皇帝身边的女人--甄嬛.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n".format("你是谁？", "").strip() + "\nassistant: ", return_tensors="pt").to(model.device)

    max_length = 512

    outputs = model.generate(
        **input,
        max_length=max_length,
        eos_token_id=7,
        do_sample=True,
        repetition_penalty=1.3,
        no_repeat_ngram_size=5,
        temperature=0.1,
        top_k=40,
        top_p=0.8,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))