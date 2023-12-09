import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import pandas as pd
from peft import TaskType, get_peft_model, LoraConfig


# 数据处理流程,参考GLM3仓库:https://github.com/THUDM/ChatGLM3/blob/main/finetune_chatmodel_demo/preprocess_utils
def process_func(example):
    MAX_LENGTH = 512
    input_ids, labels = [], []
    instruction = tokenizer.encode(text="\n".join(["<|system|>", "现在你要扮演皇帝身边的女人--甄嬛", "<|user|>", 
                                    example["instruction"] + example["input"] + "<|assistant|>"]).strip() + "\n",
                                    add_special_tokens=True, truncation=True, max_length=MAX_LENGTH)
    response = tokenizer.encode(text=example["output"], add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
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

args = TrainingArguments(
    output_dir="./output/ChatGLM",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=20,
    num_train_epochs=1
)


if "__main__" == __name__:
    # 将JSON文件转换为CSV文件,处理数据集
    df = pd.read_json('../dataset/huanhuan.jsonl')
    ds = Dataset.from_pandas(df)
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./model/chatglm3-6b", trust_remote_code=True)
    # 将数据集变化为token形式
    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

    # 创建模型
    model = AutoModelForCausalLM.from_pretrained("./model/chatglm3-6b",torch_dtype=torch.half, trust_remote_code=True, low_cpu_mem_usage=True)

    # 创建loRA参数
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules={"query_key_value"}, r=8, lora_alpha=32)

    # 模型合并
    model = get_peft_model(model, config)

    # 指定GLM的Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False
    )

    # 指定训练参数。
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    # 开始训练
    trainer.train()
