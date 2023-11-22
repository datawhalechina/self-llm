#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2023/11/22 08:07:45
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   Atom-7B-Chat-Lora指令微调
'''

from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from dataclasses import dataclass, field
import datasets
import os
import transformers
from tqdm import tqdm
import json

# dataclass：Python 类修饰符，数据类，封装了__init__()、 __repr__()和__eq__()函数
@dataclass
class FinetuneArguments:
    # 微调参数
    # field：dataclass 函数，用于指定变量初始化
    base_model: str = field(default="Atom-7B")
    dataset_path: str = field(default="../../dataset/lora/huanhuan.json")
    model_path: str = field(default="../../dataset/model")
    lora_rank: int = field(default=8)
    max_seq_length: int = field(default=256)
    skip_overlength: bool = field(default=False)
    continue_training: bool = field(default=False)
    checkpoint: str = field(default=None)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["instruction"]
    target = example["output"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def read_jsonl(path, max_seq_length, model_path, skip_overlength=False):
    model_name = model_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        lst = json.load(f)
        print("加载jsonl数据集，数据总量为{}".format(len(lst)))
        for example in tqdm(lst):
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature

def data_collator(features: list, tokenizer) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 7B
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))

def main():

    # Parse 命令行参数
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    
    print("命令行参数")
    print("finetune_args:")
    print(finetune_args.__repr__())
    print("training_args:")
    print(training_args.__repr__())

    # 初始化底座模型
    
    tokenizer = AutoTokenizer.from_pretrained(finetune_args.model_path, trust_remote_code=True)

    # Atom 为底座
    if "Atom" in finetune_args.base_model:
        model = AutoModelForCausalLM.from_pretrained(
            finetune_args.model_path, trust_remote_code=True, device_map="auto")
        print("从{}加载模型成功".format(finetune_args.model_path))
    else:
        print("错误参数：底座模型必须是Atom")
        raise ValueError("错误参数：底座模型必须是Atom")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    # model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    # 设定 peft 参数
    if "Atom" in finetune_args.base_model:
        # 手动确定 LoRA 层
        target_modules = ['W_pack', 'down_proj', 'o_proj', 'gate_proj', 'up_proj']
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=finetune_args.lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules= target_modules
        )
    else:
        print("错误参数：底座模型必须是Atom")
        raise ValueError("错误参数：底座模型必须是Atom")

    # 是否从断点继续训练
    # 源点训练
    if not finetune_args.continue_training:
        model = get_peft_model(model, peft_config)
        print("加载 LoRA 参数成功")
    else:
        if finetune_args.check_point == None:
            print("断点训练需要给出 checkpoint 地址")
            raise ValueError("断点训练需要给出 checkpoint 地址")
        model = PeftModel.from_pretrained(model, finetune_args.check_point, is_trainable=True)
        print("从{}加载断点成功".format(finetune_args.check_point))

    # 加载数据集
    try:
        dataset = datasets.Dataset.from_generator(
                lambda: read_jsonl(finetune_args.dataset_path, finetune_args.max_seq_length, finetune_args.model_path, finetune_args.skip_overlength)
            ) 
    except Exception as e:
        print("从{}加载数据集失败".format(finetune_args.dataset_path))
        print("错误信息为：")
        print(e.__repr__())
        raise e   
    print("从{}加载数据集成功".format(finetune_args.dataset_path))

    # start train
    if "Atom" in finetune_args.base_model:
        trainer = ModifiedTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            data_collator=lambda x : data_collator(x, tokenizer),
        )

    print("成功加载 Trainer")
    trainer.train()
    print("训练完成，训练结果保存在{}".format(training_args.output_dir))
    # 保存模型
    model.save_pretrained(training_args.output_dir)
    print("模型参数保存在{}".format(training_args.output_dir))


if __name__ == "__main__":
    main()