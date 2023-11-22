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
    dataset_path: str = field(default="../../dataset/lora/huanhuan.json")
    model_path: str = field(default="../../dataset/model")
    lora_rank: int = field(default=8)
    max_seq_length: int = field(default=256)
    skip_overlength: bool = field(default=False)
    continue_training: bool = field(default=False)
    checkpoint: str = field(default=None)

def preprocess(tokenizer, config, example, max_seq_length):
    '''
    args:
    tokenizer：分词器，导入的 Atom 模型分词器
    config：模型配置，导入的 Atom 模型配置
    example: 待处理的样本
    max_seq_length：文本的最大长度
    returns：字典，包括 inputs_id 和 seq_len
    '''
    # 将 instruction 和 input 按照 Atom SFT 时的格式拼接起来
    prompt = "<s>Human: " + example["instruction"] + "请回答用户问题: " + example["input"] + "\n" + "</s><s>Assistant:"
    target = example["output"]
    # 使用分词器进行编码，设置 truncation 为 True，避免出现过长的样本
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    # 加入结束符 EOS
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    # 将 inputs_ids 和 seq_len 一起传回，后续会根据 seq_len 来切分 inputs 和 labels
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

# 读取源训练数据并处理
def read_jsonl(path, max_seq_length, model_path, skip_overlength=False):
    '''
    args:
    path：训练数据路径
    max_seq_length：文本的最大长度
    model_path：模型路径，此处主要是为了加载分词器和配置
    returns：使用 yield 返回格式化的特征
    '''
    # 加载模型的分词器和配置参数
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, device_map='auto')
    # 读取源文件
    with open(path, "r") as f:
        # jsonl 数据需要先 readlines 读取成字符转，再使用 json 加载
        lst = [json.loads(line) for line in f.readlines()]
        print("加载jsonl数据集，数据总量为{}".format(len(lst)))
        # 依次处理每一个样本
        for example in tqdm(lst):
            # 调用上文的预处理函数
            feature = preprocess(tokenizer, config, example, max_seq_length)
            # 如果设置了跳过过长的样本
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            # 截断过长的样本
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            # 通过 yield 返回迭代器
            yield feature

# 自定义采样函数
def data_collator(features: list, tokenizer) -> dict:
    '''
    args:
    features: 一个批量的数据
    tokenizer：分词器
    returns：格式化的特征
    '''
    # 统计 batch 内所有数据的长度，将它们补齐
    len_ids = [len(feature["input_ids"]) for feature in features]
    # 补齐至最大长度
    longest = max(len_ids)
    # 分别存放 input_ids 和 labels
    input_ids = []
    labels_list = []
    # 有的模型没有定义 PAD，那么我们就用 UNK 来代替
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    # 从最长的文本开始处理，可以优化内存使用
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        # labels 是将输入 PAD 之后保留输出的结果，用-100表示遮蔽，并且进行补齐，计算 loss 时会自动忽略 -100
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    # 在第0维进行拼接，也就是组成 batch_size*n*n 的矩阵
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


# 自定义 Trainer，继承自 transformers.trainer
class ModifiedTrainer(Trainer):
    # 重写损失计算函数，避免 LLaMA 类模型未定义 loss 的计算
    def compute_loss(self, model, inputs, return_outputs=False):
        # 7B
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss
    # 重写模型保存函数，从而保存模型的 Lora 参数
    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME
        # 如果输出路径不存在，创建一个
        os.makedirs(output_dir, exist_ok=True)
        # 保存了模型训练的各种超参数
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        # 选出了所有梯度没有被冻结的参数，也就是所有参与更新的 Lora 参数
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        # 保存所有 Lora 参数
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))

def main():

    # 解析命令行参数
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
    model = AutoModelForCausalLM.from_pretrained(
        finetune_args.model_path, trust_remote_code=True, device_map="auto")
    print("从{}加载模型成功".format(finetune_args.model_path))

    # 启用梯度检查点，允许模型在前向计算时丢弃一些中间激活值，并在反向传播中重新计算，从而优化内存使用
    model.gradient_checkpointing_enable()
    # 确保输入向量能够计算梯度
    model.enable_input_require_grads()
    # 在训练过程中关闭缓存，提高计算效率，推理时应该开启
    model.config.use_cache = (
        False  
    )

    # 设定 peft 参数
    # 手动确定 LoRA 层（注：理论上我们可以自动查找所有 Lora 层，但是在 LLaMA 类模型上出现 bug）
    target_modules = ['W_pack', 'down_proj', 'o_proj', 'gate_proj', 'up_proj']
    # 配置 Lora 参数
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, # 任务为语言模型建模
        inference_mode=False, # 训练模式
        r=finetune_args.lora_rank, # Lora 秩
        lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1,# Dropout 比例
        target_modules= target_modules # Lora 层
    )

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

    # 加载自定义 trainer
    trainer = ModifiedTrainer(
        model=model, # 待训练模型
        train_dataset=dataset, # 数据集
        args=training_args, # 训练参数
        data_collator=lambda x : data_collator(x, tokenizer), # 自定义采样函数
    )

    print("成功加载 Trainer")
    # 进行训练
    trainer.train()
    print("训练完成，训练结果保存在{}".format(training_args.output_dir))
    # 保存模型
    model.save_pretrained(training_args.output_dir)
    print("模型参数保存在{}".format(training_args.output_dir))


if __name__ == "__main__":
    main()