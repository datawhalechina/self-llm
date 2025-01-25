import json
from functools import partial
from typing import Dict
from torchvision import transforms
import torch
import transformers
from transformers import AutoModel, AutoTokenizer, TrainingArguments
from minicpm_datasets import SupervisedDataset, data_collator
from trainer import CPMTrainer
from peft import LoraConfig, get_peft_model, PeftModel
from modelscope import snapshot_download
import swanlab
from swanlab.integration.transformers import SwanLabCallback
import os
from PIL import Image

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_path,
    transform,
    data_collator=None,
    llm_type="qwen",
    slice_config=None,
    patch_size=14,
    query_nums=64,
    batch_vision=False,
    max_length=2048,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    print("Loading data...")

    train_json = json.load(open(data_path, "r"))
    train_dataset = SupervisedDataset(
        train_json,
        transform,
        tokenizer,
        slice_config=slice_config,
        llm_type=llm_type,
        patch_size=patch_size,
        query_nums=query_nums,
        batch_vision=batch_vision,
        max_length=max_length,
    )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator= partial(data_collator, max_length=max_length),
    )


model_id = "OpenBMB/MiniCPM-o-2_6"
data_path="./latex_ocr_train.json"
output_dir="./output/minicpm-o-2-6-latexocr"

llm_type: str = "qwen"
tune_vision: bool = True
tune_llm: bool = False
use_lora: bool = True

max_steps: int = 1000
model_max_length: int = 2048
max_slice_nums: int = 9

lora_rank: int = 64
lora_alpha: int = 16
lora_dropout: float = 0.1

# 设置Transformers训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    bf16=True,
    logging_strategy="steps",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    save_strategy="steps",
    save_steps=500,
    max_steps=max_steps,
    save_total_limit=10,
    learning_rate=1e-6,
    weight_decay=0.1,
    adam_beta2=0.95,
    warmup_ratio=0.01,
    lr_scheduler_type="cosine",
    logging_steps=10,
    gradient_checkpointing=True,
    label_names="labels",
    remove_unused_columns=False,
    gradient_checkpointing_kwargs={"use_reentrant":False},
    report_to="none",
)

# 下载模型
model_dir = snapshot_download(model_id, cache_dir="/root/autodl-tmp/", revision="master")

# 加载模型
model = AutoModel.from_pretrained(
    model_dir,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map=None,
    init_vision=True,
    init_audio=False,
    init_tts=False,
)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# 参数冻结
if not tune_vision:
    model.vpm.requires_grad_(False)
if not tune_llm:
    model.llm.requires_grad_(False)
    
# 配置Lora
if use_lora:
    # 如果同时微调llm和使用lora，则报错
    if use_lora and tune_llm:
        raise ValueError("The model cannot simultaneously adjust LLM parameters and apply LoRA.")
    
    print("Currently using LoRA for fine-tuning the MiniCPM-V model.")
    # 冻结llm参数
    for name, param in model.llm.named_parameters():
        param.requires_grad = False
    # 设置需要保存的模块
    modules_to_save = ['embed_tokens','resampler']
    if tune_vision:
        modules_to_save.append('vpm')
        
    # 设置lora配置
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules="llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
        lora_dropout=lora_dropout,
    )
    # 将模型转换为peft模型
    model = get_peft_model(model, lora_config)
    # 启用输入梯度
    model.enable_input_require_grads()

    model.config.slice_config.max_slice_nums = max_slice_nums
    slice_config = model.config.slice_config.to_dict()
    batch_vision = model.config.batch_vision_input

# 设置数据集预处理
transform_func = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5) , std=(0.5, 0.5, 0.5)
            ),
        ]
    )

# 数据集模块
data_module = make_supervised_data_module(
    tokenizer=tokenizer,
    data_path=data_path,
    transform=transform_func,
    data_collator=data_collator,
    slice_config=slice_config,
    llm_type=llm_type,
    patch_size=model.config.patch_size,
    query_nums=model.config.query_num,
    batch_vision=batch_vision,
    max_length=model_max_length,
)

# 集成SwanLab训练可视化工具
swanlab_callback = SwanLabCallback(
    project="minicpm-o-2-6-latexcor",
    experiment_name="minicpm-o-2-6",
    config={
        "github_repo": "self-llm",
        "model": "https://modelscope.cn/models/OpenBMB/MiniCPM-o-2_6",
        "dataset": "https://modelscope.cn/datasets/AI-ModelScope/LaTeX_OCR/summary",
        "model_id": model_id,
        "train_dataset_json_path": data_path,
        "output_dir": "output/output__lora",
        "token_max_length": model_max_length,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
    }
)

trainer = CPMTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    use_lora=use_lora,
    callbacks=[swanlab_callback],
    **data_module,
)

trainer.train()


# ========== 主观测试 ==========

# 释放trainer中的model显存
trainer.model.cpu()
del trainer.model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 获取测试模型，从output_dir中获取最新的checkpoint
load_model_path = f"{output_dir}/checkpoint-{max([int(d.split('-')[-1]) for d in os.listdir(output_dir) if d.startswith('checkpoint-')])}"
print(f"load_model_path: {load_model_path}")

origin_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
val_lora_model = PeftModel.from_pretrained(
    origin_model,
    load_model_path,
    device_map="auto",
    trust_remote_code=True
).eval().cuda()

# 读取测试数据
with open("./latex_ocr_val.json", "r") as f:
    test_dataset = json.load(f)

test_image_list = []
for item in test_dataset:
    image_file_path = item["image"]
    label = item["conversations"][1]["content"]
    
    image = Image.open(image_file_path).convert('RGB')

    question = "这张图对应的LaTex公式是什么？"
    msgs = [{'role': 'user', 'content': [image, question]}]

    answer = val_lora_model.chat(
        msgs=msgs,
        tokenizer=tokenizer
    )

    print(f"predict:{answer}")
    print(f"gt:{label}\n")

    test_image_list.append(swanlab.Image(image_file_path, caption=answer))

swanlab.log({"Prediction": test_image_list})

# 在Jupyter Notebook中运行时要停止SwanLab记录，需要调用swanlab.finish()
swanlab.finish()