# AMchat 高等数学大模型

## 📖 简介

AM (Advanced Mathematics) chat 是一个集成了数学知识和高等数学习题及其解答的大语言模型。该模型使用 Math 和高等数学习题及其解析融合的数据集，基于 InternLM2-Math-7B 模型，通过 xtuner 微调，专门设计用于解答高等数学问题。

你在这里可以学到 **收集数据**、**制作数据集**、**模型微调**和 **部署模型** 的全流程，帮助你更好地理解和掌握大模型应用开发的核心技术。

> *AMchat模型: Modelscope 地址：[*Link*](https://www.modelscope.cn/models/yondong/AMchat/summary)* ， *OpenXLab 地址：[*Link*](https://openxlab.org.cn/models/detail/youngdon/AMchat)*，HuggingFace 地址：[*Link*](https://huggingface.co/axyzdong/AMchat)\
> *AMchat 项目地址：*[*Link*](https://github.com/AXYZdong/AMchat)\
> *AMchat 应用地址：*[*Link*](https://openxlab.org.cn/apps/detail/youngdon/AMchat)\
> *AMchat 视频介绍：*[*Link*](https://www.bilibili.com/video/BV14v421i7So/) 


## 🛠️ 实现微调模型

接下来，我们将演示如何基于 Xtuner QLoRA 框架，快速实现一个用于高等数学的微调模型。 本教程将详细讲解针对 internLM2 进行数学微调的全流程，涵盖数据准备、微调执行以模型部署等环节。

本次微调示例选用 internlm2-math-7b 模型。你需要准备一台配备 24GB 显存的机器进行微调（例如 NVIDIA GeForce RTX 3090）。

微调模型的第一步是准备高质量的训练数据。对于数学大模型而言，你需要收集数学题目，数据来源可以是公开的数学数据集或自行收集的数据。

在准备完成数据来源以及获取到对应数据后，你需要使用该数据制作成结构化的数据集，以便于模型进行微调训练。


### Step 1: 数据准备

我们可以用调用api的方式让一些大模型自动生成数学题目，然后再将生成的数学题目进行答案的标注。

另外，我们也可以自己从一些考研数学题库中通过OCR的方式获取数学题目，然后再进行答案的标注。

对与每一条数据，需要整理成如下格式：

```json
{
    "conversation": [
        {
            "input": "设 a_{n}=\\frac{3}{2} \\int_{0}^{\\frac{n}{n+1}} x^{n-1} \\sqrt{1+x^{n}} \\mathrm{~d} x, 则 \\lim _{n \\rightarrow \\infty} n a_{n}=\\\\",
            "output": "\\left(1+\\mathrm{e}^{-1}\\right)^{\\frac{3}{2}}-1 \\\\ 解\\\\\\\\\\begin{aligned}\\\\a_{n} & =\\frac{3}{2} \\int_{0}^{\\frac{n}{n+1}} x^{n-1} \\sqrt{1+x^{n}} \\mathrm{~d} x=\\frac{3}{2} \\cdot \\frac{1}{n} \\int_{0}^{\\frac{n}{n+1}}\\left(1+x^{n}\\right)^{\\frac{1}{2}} \\mathrm{~d}\\left(1+x^{n}\\right) \\\\\\\\& =\\left.\\frac{3}{2} \\cdot \\frac{1}{n} \\cdot \\frac{2}{3}\\left(1+x^{n}\\right)^{\\frac{3}{2}}\\right|_{0} ^{\\frac{n}{n+1}}=\\frac{1}{n}\\left\\{\\left[1+\\left(\\frac{n}{n+1}\\right)^{n}\\right]^{\\frac{3}{2}}-1\\right\\} .\\\\\\end{aligned}\\\\\\\\\\\\由 \\lim _{n \\rightarrow \\infty}\\left(\\frac{n+1}{n}\\right)^{n}=\\mathrm{e}, 知 \\lim _{n \\rightarrow \\infty}\\left(\\frac{n}{n+1}\\right)^{n}=\\frac{1}{\\mathrm{e}}, 故\\\\\\\\\\lim _{n \\rightarrow \\infty} n a_{n}=\\lim _{n \\rightarrow \\infty}\\left\\{\\left[1+\\left(\\frac{n}{n+1}\\right)^{n}\\right]^{\\frac{3}{2}}-1\\right\\}=\\left(1+\\mathrm{e}^{-1}\\right)^{\\frac{3}{2}}-1 .\\\\"
        }
    ]
}
```

每个 "conversation" 字段包含一个对话，对话中包含一个输入和一个输出。输入是数学题目，输出是数学题目的答案。


### Step 2: 环境准备

1. clone 项目

```bash
git clone https://github.com/AXYZdong/AMchat.git
cd AMchat
```

2. 创建虚拟环境

```bash
conda env create -f environment.yml
conda activate AMchat
pip install xtuner
```

### Step 3:  模型微调

1. 基座模型下载

```bash
mkdir -p /root/math/model
```
`download.py`

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-math-7b', cache_dir='/root/math/model')
```

2. 准备配置文件

```bash
# 列出所有内置配置
xtuner list-cfg

mkdir -p /root/math/data
mkdir /root/math/config && cd /root/math/config

xtuner copy-cfg internlm2_chat_7b_qlora_oasst1_e3 .
```

3. 修改配置文件

> 仓库中 `config` 文件夹下已经提供了一个微调的配置文件，可以参考 `internlm_chat_7b_qlora_oasst1_e3_copy.py`。
> 可以直接使用，注意修改  `pretrained_model_name_or_path` 和 `data_path` 的路径。

配置文件代码如下：
```python
import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import oasst1_map_fn, template_map_fn_factory
from xtuner.engine import DatasetInfoHook, EvaluateChatHook
from xtuner.model import SupervisedFinetune
from xtuner.utils import PROMPT_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = '/root/math/model/Shanghai_AI_Laboratory/internlm2-math-7b'

# Data
data_path = '../dataset/AMchat_dataset.json'
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 2048
pack_to_max_length = True

# Scheduler & Optimizer
batch_size = 2  # per_device
accumulative_counts = 1
dataloader_num_workers = 0
max_epochs = 3
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = "You're a professor of mathematics."

evaluation_inputs = [
    '2x^2+3x+1=10，求x', '求积分 $\int_{0}^{1} x dx$'
]

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=SupervisedFinetune,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path)),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        T_max=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template)
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 100 iterations.
    logger=dict(type=LoggerHook, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per epoch.
    checkpoint=dict(type=CheckpointHook, interval=1),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)
```

也可以在命令行中直接修改配置文件。

```bash
cd /root/math/config
vim internlm_chat_7b_qlora_oasst1_e3_copy.py
```

```python
# 修改模型为本地路径
- pretrained_model_name_or_path = 'internlm/internlm-chat-7b'
+ pretrained_model_name_or_path = './internlm2-math-7b'

# 修改训练数据集为本地路径
- data_path = '../dataset/AMchat_dataset.json'
+ data_path = './data'
```

4. 开始微调

```bash
xtuner train /root/math/config/internlm2_chat_7b_qlora_oasst1_e3_copy.py
```

5. PTH 模型转换为 HuggingFace 模型

```bash
mkdir hf
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner convert pth_to_hf ./internlm2_chat_7b_qlora_oasst1_e3_copy.py \
                         ./work_dirs/internlm2_chat_7b_qlora_oasst1_e3_copy/epoch_3.pth \
                         ./hf
```

6. HuggingFace 模型合并到大语言模型
```bash
# 原始模型参数存放的位置
export NAME_OR_PATH_TO_LLM=/root/math/model/Shanghai_AI_Laboratory/internlm2-math-7b

# Hugging Face格式参数存放的位置
export NAME_OR_PATH_TO_ADAPTER=/root/math/config/hf

# 最终Merge后的参数存放的位置
mkdir /root/math/config/work_dirs/hf_merge
export SAVE_PATH=/root/math/config/work_dirs/hf_merge

# 执行参数Merge
xtuner convert merge \
    $NAME_OR_PATH_TO_LLM \
    $NAME_OR_PATH_TO_ADAPTER \
    $SAVE_PATH \
    --max-shard-size 2GB
```

7. Demo

```bash
streamlit run web_demo.py --server.address=0.0.0.0 --server.port 7860
```


### 致谢每一位贡献者 

核心贡献者：

- [张友东](https://github.com/AXYZdong) （Datawhale成员-东南大学）
- [宋志学](https://github.com/KMnO4-zx)（Datawhale成员-中国矿业大学(北京)）
- [肖鸿儒](https://github.com/Hongru0306)（Datawhale成员-同济大学）

贡献者目录：

https://github.com/AXYZdong/AMchat/graphs/contributors