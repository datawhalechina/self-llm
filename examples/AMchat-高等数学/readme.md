# AMchat Advanced Mathematics Large Model

## 📖 Introduction

AM (Advanced Mathematics) chat is a large language model integrated with mathematical knowledge and advanced mathematics exercises and their solutions. This model uses a dataset fused with Math and advanced mathematics exercises and their analyses. Based on the InternLM2-Math-7B model and fine-tuned via xtuner, it is specifically designed to solve advanced mathematics problems.

Here you can learn the full process of **data collection**, **dataset creation**, **model fine-tuning**, and **model deployment**, helping you better understand and master the core technologies of large model application development.

> *AMchat Model: Modelscope Address: [*Link*](https://www.modelscope.cn/models/yondong/AMchat/summary)*, *OpenXLab Address: [*Link*](https://openxlab.org.cn/models/detail/youngdon/AMchat)*, HuggingFace Address: [*Link*](https://huggingface.co/axyzdong/AMchat)\
> *AMchat Project Address:* [*Link*](https://github.com/AXYZdong/AMchat)\
> *AMchat Application Address:* [*Link*](https://openxlab.org.cn/apps/detail/youngdon/AMchat)\
> *AMchat Video Introduction:* [*Link*](https://www.bilibili.com/video/BV14v421i7So/) 


## 🛠️ Implementing Model Fine-tuning

Next, we will demonstrate how to quickly implement a fine-tuned model for advanced mathematics based on the Xtuner QLoRA framework. This tutorial will explain in detail the full process of mathematical fine-tuning for internLM2, covering data preparation, fine-tuning execution, and model deployment.

This fine-tuning example uses the internlm2-math-7b model. You need to prepare a machine equipped with 24GB of video memory for fine-tuning (e.g., NVIDIA GeForce RTX 3090).

The first step in fine-tuning a model is to prepare high-quality training data. For a mathematical large model, you need to collect math problems. The data source can be public math datasets or self-collected data.

After preparing the data source and obtaining the corresponding data, you need to make the data into a structured dataset to facilitate model fine-tuning training.


### Step 1: Data Preparation

We can use API calls to let some large models automatically generate math problems, and then annotate the answers to the generated math problems.

In addition, we can also obtain math problems from some postgraduate entrance examination math question banks through OCR, and then annotate the answers.

For each piece of data, it needs to be organized into the following format:

```json
{
    "conversation": [
        {
            "input": "Let a_{n}=\\frac{3}{2} \\int_{0}^{\\frac{n}{n+1}} x^{n-1} \\sqrt{1+x^{n}} \\mathrm{~d} x, then \\lim _{n \\rightarrow \\infty} n a_{n}=\\\\",
            "output": "\\left(1+\\mathrm{e}^{-1}\\right)^{\\frac{3}{2}}-1 \\\\ Solution\\\\\\\\\\begin{aligned}\\\\a_{n} & =\\frac{3}{2} \\int_{0}^{\\frac{n}{n+1}} x^{n-1} \\sqrt{1+x^{n}} \\mathrm{~d} x=\\frac{3}{2} \\cdot \\frac{1}{n} \\int_{0}^{\\frac{n}{n+1}}\\left(1+x^{n}\\right)^{\\frac{1}{2}} \\mathrm{~d}\\left(1+x^{n}\\right) \\\\\\\\& =\\left.\\frac{3}{2} \\cdot \\frac{1}{n} \\cdot \\frac{2}{3}\\left(1+x^{n}\\right)^{\\frac{3}{2}}\\right|_{0} ^{\\frac{n}{n+1}}=\\frac{1}{n}\\left\\{\\left[1+\\left(\\frac{n}{n+1}\\right)^{n}\\right]^{\\frac{3}{2}}-1\\right\\} .\\\\\\end{aligned}\\\\\\\\\\\\Since \\lim _{n \\rightarrow \\infty}\\left(\\frac{n+1}{n}\\right)^{n}=\\mathrm{e}, we know \\lim _{n \\rightarrow \\infty}\\left(\\frac{n}{n+1}\\right)^{n}=\\frac{1}{\\mathrm{e}}, so\\\\\\\\\\lim _{n \\rightarrow \\infty} n a_{n}=\\lim _{n \\rightarrow \\infty}\\left\\{\\left[1+\\left(\\frac{n}{n+1}\\right)^{n}\\right]^{\\frac{3}{2}}-1\\right\\}=\\left(1+\\mathrm{e}^{-1}\\right)^{\\frac{3}{2}}-1 .\\\\"
        }
    ]
}
```

Each "conversation" field contains a dialogue, which includes an input and an output. The input is the math problem, and the output is the answer to the math problem.

> Small-scale open source dataset: [AMchat_dataset](https://github.com/AXYZdong/AMchat/tree/main/dataset)

### Step 2: Environment Preparation

1. Clone the project

```bash
git clone https://github.com/AXYZdong/AMchat.git
cd AMchat
```

2. Create a virtual environment

```bash
conda env create -f environment.yml
conda activate AMchat
pip install xtuner
```

### Step 3: Model Fine-tuning

1. Download Base Model

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

2. Prepare Configuration File

```bash
# List all built-in configurations
xtuner list-cfg

mkdir -p /root/math/data
mkdir /root/math/config && cd /root/math/config

xtuner copy-cfg internlm2_chat_7b_qlora_oasst1_e3 .
```

3. Modify Configuration File

> A fine-tuning configuration file is already provided in the `config` folder of the repository, you can refer to `internlm_chat_7b_qlora_oasst1_e3_copy.py`.
> You can use it directly, pay attention to modifying the paths of `pretrained_model_name_or_path` and `data_path`.

The configuration file code is as follows:
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
    '2x^2+3x+1=10，find x', 'Calculate integral $\int_{0}^{1} x dx$'
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

You can also modify the configuration file directly in the command line.

```bash
cd /root/math/config
vim internlm_chat_7b_qlora_oasst1_e3_copy.py
```

```python
# Modify model to local path
- pretrained_model_name_or_path = 'internlm/internlm-chat-7b'
+ pretrained_model_name_or_path = './internlm2-math-7b'

# Modify training dataset to local path
- data_path = '../dataset/AMchat_dataset.json'
+ data_path = './data'
```

4. Start Fine-tuning

```bash
xtuner train /root/math/config/internlm2_chat_7b_qlora_oasst1_e3_copy.py
```

5. Convert PTH Model to HuggingFace Model

```bash
mkdir hf
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner convert pth_to_hf ./internlm2_chat_7b_qlora_oasst1_e3_copy.py \
                         ./work_dirs/internlm2_chat_7b_qlora_oasst1_e3_copy/epoch_3.pth \
                         ./hf
```

6. Merge HuggingFace Model into Large Language Model
```bash
# Location of original model parameters
export NAME_OR_PATH_TO_LLM=/root/math/model/Shanghai_AI_Laboratory/internlm2-math-7b

# Location of Hugging Face format parameters
export NAME_OR_PATH_TO_ADAPTER=/root/math/config/hf

# Location of final Merged parameters
mkdir /root/math/config/work_dirs/hf_merge
export SAVE_PATH=/root/math/config/work_dirs/hf_merge

# Execute Parameter Merge
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


### Acknowledgements to Every Contributor

Core Contributors:

- [Zhang Youdong](https://github.com/AXYZdong) (Datawhale Member - Southeast University)
- [Song Zhixue](https://github.com/KMnO4-zx) (Datawhale Member - China University of Mining and Technology (Beijing))
- [Xiao Hongru](https://github.com/Hongru0306) (Datawhale Member - Tongji University)

Contributors List:

https://github.com/AXYZdong/AMchat/graphs/contributors