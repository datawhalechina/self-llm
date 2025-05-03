# Atom-7B-Chat 的 Lora 指令微调

## 概述

本节我们简要介绍如何基于 transformers、peft 等框架，对 Atom-7B-Chat 模型进行 Lora 微调。Lora 是一种高效微调方法，深入了解其原理可参见博客：[知乎|深入浅出Lora](https://zhuanlan.zhihu.com/p/650197598)。

本节所讲述的代码脚本在同级目录 [02-Atom-7B-Chat Lora](../Atom/02-Atom-7B-Chat%20Lora%20微调/train.py) 下，可以通过运行目录下 [train.sh](../Atom/02-Atom-7B-Chat%20Lora%20微调/train.sh) 脚本来执行微调过程，但注意，本文代码未使用分布式框架，微调 Atom-7B 模型至少需要 32G 及以上的显存。

## 环境配置

在完成基本环境配置和本地模型部署的情况下，你还需要安装一些第三方库，可以使用以下命令：

```bash
pip install transformers==4.36.0.dev0
pip install peft==0.4.0.dev0
pip install datasets==2.10.1
pip install accelerate==0.20.3
```

在本节教程里，我们将微调数据集放置在根目录 [/dataset](../../dataset/huanhuan.jsonl)，将基座模型参数放置在根目录 [/model](../../models)。

## 指令集构建

LLM 的微调一般指指令微调过程。所谓指令微调，是说我们使用的微调数据形如：

```json
{
    "instrution":"回答以下用户问题，仅输出答案。",
    "input":"1+1等于几?",
    "output":"2"
}
```

其中，`instruction` 是用户指令，告知模型其需要完成的任务；`input` 是用户输入，是完成用户指令所必须的输入内容；`output` 是模型应该给出的输出。

即我们的核心训练目标是让模型具有理解并遵循用户指令的能力。因此，在指令集构建时，我们应针对我们的目标任务，针对性构建任务指令集。例如，在本节我们使用由笔者合作开源的 [Chat-甄嬛](https://github.com/KMnO4-zx/huanhuan-chat) 项目作为示例，我们的目标是构建一个能够模拟甄嬛对话风格的个性化 LLM，因此我们构造的指令形如：

```json
{
    "instruction": "请参考下面内容中的甄嬛的说话风格和语气，回答我的问题。甄嬛的说话风格需要是口语化的，回复内容不要超过30个字，尽可能字数简短一些。\n对话风格案例内容：\n```用户：这首歌虽未直写男女相悦，可字字写着两心相悦后女子的欢喜神态，而且'双双金鹧鹄'也是并蒂成双之意。\n用户：既然如此，安常在怎么就没唱出花好之情？难不成是看见皇上跟本宫在一起，心有不悦才唱不好的吗？\n甄嬛：回禀华妃娘娘，安常在早上受了风寒，嗓子有些不适。\n\n用户：你不是要看院子里的白梅吗，怎么那么快就回来了？\n甄嬛：雪景看久了反倒眼晕，四郎本是好意在园子里种植白梅，可是一下雪反倒与雪景融为一色，倒看不出来了。",
    "input":"你是谁？",
    "output":"家父是大理寺少卿甄远道。"

}
```

我们所构造的全部指令数据集在根目录下。

## 数据格式化

Lora 训练的数据是需要经过格式化、编码之后再输入给模型进行训练的，如果是熟悉 Pytorch 模型训练流程的同学会知道，我们一般需要将输入文本编码为 `input_ids`，将输出文本编码为 `labels`，编码之后的结果都是多维的向量。我们首先定义一个预处理函数，这个函数用于对每一个样本，编码其输入、输出文本并返回一个编码后的字典：

```python
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
```

上述代码会对每一条样本进行格式化处理，我们再定义一个函数，这个函数基于上文函数，对源训练数据进行处理：

```python
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
```

完成上述函数后，我们使用 datasets 库提供的 from_generator 函数来根据上述函数生成我们数据的 Dataset 对象：

```python
# 通过 read_jsonl 函数返回的迭代器来生成 Dataset 对象，这个 Dataset 对象可以直接用在 transformers 框架中
dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl(
            finetune_args.dataset_path, finetune_args.max_seq_length, finetune_args.model_path, finetune_args.skip_overlength
            )
    ) 
```

## 采样函数

为了对每一个 batch 的数据进行动态补齐，避免造成资源浪费，我们没有在生成函数中进行补齐操作，因此我们需要定义一个自定义采样函数，这个函数代替了 torch 中默认的采样函数功能，并自定义地实现了补齐、labels 遮蔽等操作，后续其会以 lambda 函数的方式传入 trainer：

```python
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
```

## 自定义 Trainer

对于 Lora 微调，我们需要在基础 Trainer 的基础上继承一个自定义 Trainer，实现 Loss 计算（部分模型需要）和 Lora 参数的保存：

```python
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
```

## 参数解析

由于模型微调需要用到众多参数，最好以命令行方式传入，再通过 bash 脚本调用（例如 [train.sh](../Atom/02-Atom-7B-Chat-Lora/train.sh) 脚本）。微调参数众多，transformers 提供了众多参数的解析，我们需要额外定义一个参数解析类，解析 transformers 没有提供的、用于 Lora 微调的参数解析：

```python
# dataclass：Python 类修饰符，数据类，封装了__init__()、 __repr__()和__eq__()函数
@dataclass
class FinetuneArguments:
    # 微调参数
    # field：dataclass 函数，用于指定变量初始化
    # 训练集路径
    dataset_path: str = field(default="../../dataset/huanhuan.jsonl")
    # 基座模型参数路径
    model_path: str = field(default="../../dataset/model")
    # Lora 秩
    lora_rank: int = field(default=8)
    # 最大文本长度
    max_seq_length: int = field(default=256)
    # 是否跳过超长文本
    skip_overlength: bool = field(default=False)
    # 是否从断点继续训练
    continue_training: bool = field(default=False)
    # 断点路径，如果从断点继续训练需要传入
    checkpoint: str = field(default=None)
```

## 训练

完成上述定义和实现之后，我们可以正式开始我们的训练流程。首先我们需要解析脚本传入的训练参数，我们使用了 transformers 提供的 HfArgumentParser 函数，解析的参数包括 transformers 提供的 TrainingArguments 类（包括了一些常用训练参数）和我们自定义的 FinetuneArguments 类：

```python
# 解析命令行参数
finetune_args, training_args = HfArgumentParser(
    (FinetuneArguments, TrainingArguments)
).parse_args_into_dataclasses()
```

接下来加载底座模型并进行一定的配置：

```python
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
```

然后设定 Lora 参数：

```python
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
```

再基于 Lora 配置和底座模型，得到待训练的 Lora 模型（即冻结了非 Lora 层）。同时，需要判断是否是断点继续训练，如果是则要加载断点信息：

```python
# 是否从断点继续训练
# 源点训练
if not finetune_args.continue_training:
    # 对基座模型进行 Lora 融合
    model = get_peft_model(model, peft_config)
    print("加载 LoRA 参数成功")
else:
    if finetune_args.check_point == None:
        print("断点训练需要给出 checkpoint 地址")
        raise ValueError("断点训练需要给出 checkpoint 地址")
    # 断点继续训练则直接加载断点的 Lora 参数
    model = PeftModel.from_pretrained(model, finetune_args.check_point, is_trainable=True)
    print("从{}加载断点成功".format(finetune_args.check_point))
```

然后基于上述定义加载数据集，在这一部分，我们使用了 try except 来捕捉异常：

```python
# 加载数据集
try:
    # 调用上述定义函数生成迭代器
    dataset = datasets.Dataset.from_generator(
            lambda: read_jsonl(finetune_args.dataset_path, finetune_args.max_seq_length, finetune_args.model_path, finetune_args.skip_overlength)
        ) 
except Exception as e:
    print("从{}加载数据集失败".format(finetune_args.dataset_path))
    print("错误信息为：")
    print(e.__repr__())
    raise e   
print("从{}加载数据集成功".format(finetune_args.dataset_path))
```

最后，加载一个自定义的 trainer 并开始训练：

```python
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
```

通过上述代码，我们即可完成 Atom-7B-Chat 模型的 Lora 微调。我们将上述代码封装为 [train.py](../Atom/02-Atom-7B-Chat-Lora/train.py) 脚本，同时，提供一个启动训练的 [bash 脚本](../Atom/02-Atom-7B-Chat-Lora/train.sh)：

```bash
python train.py \
    --dataset_path ../../dataset/huanhuan.jsonl \ # 数据集路径
    --model_path /root/autodl-tmp/data/model/Atom \ # 基座模型路径
    --lora_rank 8 \ # lora 秩
    --per_device_train_batch_size 16 \ # batch_size
    --gradient_accumulation_steps 1 \ # 梯度累积轮次
    --max_steps 120000 \ # 训练最大步数，训练 epoch 数 = max_steps / (num_whole_data / batch_size)
    --save_steps 40000 \ # 每训练多少步保存一次参数
    --save_total_limit 3 \ # 最多保存多少个参数
    --learning_rate 1e-4 \ # 学习率
    --fp16 \ # 使用 float16 的精度
    --remove_unused_columns false \ # 数据集处理时是否去除没有使用的特征
    --logging_steps 10 \ # 每训练多少步输出一次
    --output_dir ../../output # 输出路径
```

直接在目录下运行该脚本（bash train.sh）即可开始训练。