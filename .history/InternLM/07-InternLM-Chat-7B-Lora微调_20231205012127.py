from peft import LoraConfig, TaskType, get_peft_model
import torch
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

df = pd.read_json('./dataset/huanhuan.json')
ds = Dataset.from_pandas(df)
tokenizer = AutoTokenizer.from_pretrained(
    '/root/autodl-fs/Shanghai_AI_Laboratory/internlm-chat-7b', use_fast=False, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer

print(tokenizer.eos_token_id)

print(tokenizer.encode(ds[0]['instruction']))

text = "现在你要扮演皇帝身边的女人--甄嬛"
inputs = tokenizer(text, return_tensors="pt")
print(f"inputs:{inputs}")
# input_ids = inputs["input_ids"].to("cuda")
input_ids = inputs["input_ids"]

# %%
print(input_ids)

# %%


def process_func(example):
    MAX_LENGTH = 128
    input_ids, labels = [], []
    instruction = tokenizer.encode(text="\n".join(["<|system|>", "现在你要扮演皇帝身边的女人--甄嬛", "<|user|>" +
                                                   example["instruction"] + example["input"] + "<|assistant|>"]).strip() + "\n",
                                   add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
    response = tokenizer.encode(
        text=example["output"], add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
    input_ids = instruction + response + [tokenizer.eos_token_id]
    labels = [tokenizer.pad_token_id] * \
        len(instruction) + response + [tokenizer.eos_token_id]
    pad_len = MAX_LENGTH - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    labels += [tokenizer.pad_token_id] * pad_len
    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

    return {
        "input_ids": input_ids,
        "labels": labels
    }


# %%
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
tokenized_id

# %%
tokenizer.decode(tokenized_id[0]['input_ids'])

# %%
tokenizer.decode(tokenized_id[0]['input_ids'])

# %%
tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))

# %% [markdown]
# # 创建模型

# %%

model = AutoModelForCausalLM.from_pretrained(
    '/root/autodl-fs/Shanghai_AI_Laboratory/internlm-chat-7b', trust_remote_code=True, torch_dtype=torch.half, device_map="auto")
model

# %%
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# %%
model.dtype

# %%
for name, param in model.named_parameters():
    print(name)

# %% [markdown]
# # Lora 微调

# %% [markdown]
# 1. target_modules也可以传入正则项,比如以h.1结尾的query_key_value：".*\.1.*query_key_value"
# 2. modules_to_save指定的是除了拆成lora的模块，其他的模块可以完整的指定训练。

# %%

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)
config

# %%
model = get_peft_model(model, config)
config

# %%
model.print_trainable_parameters()

# %% [markdown]
# # 配置训练参数

# %%
args = TrainingArguments(
    output_dir="./output/InternLM",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    logging_steps=10,  # 输出步数
    num_train_epochs=1,
    gradient_checkpointing=True,
    save_steps=50,  # 保存步数
    learning_rate=1e-4,  # 学习率
    save_on_each_node=True
)

# %%
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# %%
trainer.train()

# %%
model.eval()
# ipt = tokenizer("<|im_start|>system\n现在你要扮演皇帝身边的女人--甄嬛.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n".format( "介绍一下自己是谁", "").strip() + "\nAssistant: ", return_tensors="pt").to(model.device)
# tokenizer.decode(model.generate(**ipt, max_length=512, do_sample=True, eos_token_id=tokenizer.eos_token_id, temperature=0.1)[0], skip_special_tokens=True)

# %%
response, history = model.chat(tokenizer, "介绍一下自己是谁", history=[])
response

# %%
model = model.cuda()
# ipt = tokenizer("<|system|>\n现在你要扮演皇帝身边的女人--甄嬛\n<|user|>\n {}\n{}".format("你是谁？", "").strip() + "<|assistant|>\n", return_tensors="pt").to(model.device)
# tokenizer.decode(model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)

# %%
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
assert len(response) != 0
response, history = model.chat(tokenizer, "请提供三个管理时间的建议。", history=history)
print(response)
assert len(response) != 0

# %%
ipt = tokenizer("<|Bot|>\n现在你要扮演皇帝身边的女人--甄嬛\n<|User|>\n {}\n{}".format("你是谁？",
                "").strip() + "<|assistant|>\n", return_tensors="pt").to(model.device)
tokenizer.decode(model.generate(**ipt, max_length=128,
                 do_sample=True)[0], skip_special_tokens=True)

# %%
response, history = model.chat(tokenizer, "你好", history=[])
response
