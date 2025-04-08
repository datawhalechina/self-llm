# MiniCPM-2-V transformers 部署微调和应用

MiniCPM-V 2.0 是一个高效的多模态大型语言模型，具备 2.8B 参数。该模型在多个基准测试中表现出色，包括 OCRBench、TextVQA、MME 等，超越了许多参数量更大的模型。MiniCPM-V 2.0 具有以下特点：

1. **性能卓越**：在多项基准测试中达到最先进水平，尤其在场景文字理解上表现出色，与 Gemini Pro 相当。
2. **可靠行为**：通过多模态 RLHF 技术（多模态强化学习人类反馈），确保生成内容的可信度，匹配 GPT-4V 的防幻觉能力。
3. **高分辨率图像处理**：支持任意长宽比的高分辨率图像输入，提升细粒度视觉信息感知能力。
4. **高效部署**：能够在多数 GPU 卡和个人电脑上高效运行，甚至可在移动设备上运行。
5. **双语支持**：具备强大的中英文多模态能力，支持跨语言多模态应用。

模型可以在 NVIDIA GPU 或 Mac 的 MPS 上进行推理，并通过 vLLM 实现高效推理。详细的安装和使用指南请参考 [GitHub 仓库](https://github.com/OpenBMB/MiniCPM-V)。

MiniCPM-V 2.0 完全开源，免费供学术研究使用，并在填写问卷后免费用于商业用途。有关模型的更多信息和技术细节，请访问 [技术博客](https://openbmb.vercel.app/minicpm-v-2)。

可以通过如下的方式推理：


```python
from chat import MiniCPMVChat, img2base64
import torch
import json

torch.manual_seed(0)

chat_model = MiniCPMVChat('openbmb/MiniCPM-V-2')

im_64 = img2base64('./assets/airplane.jpeg')

# First round chat 
msgs = [{"role": "user", "content": "Tell me the model of this aircraft."}]

inputs = {"image": im_64, "question": json.dumps(msgs)}
answer = chat_model.chat(inputs)
print(answer)

# Second round chat 
# pass history context of multi-turn conversation
msgs.append({"role": "assistant", "content": answer})
msgs.append({"role": "user", "content": "Introduce something about Airbus A380."})

inputs = {"image": im_64, "question": json.dumps(msgs)}
answer = chat_model.chat(inputs)
print(answer)
```

### 数据准备

请将数据准备为如下的json格式，对于多模态图像，需要设置图像路径，支持多轮对话，但是每轮只能使用一张图片。


```python
  [
    {
      "id": "0",
      "image": 'path/to/image_0.jpg',
      "conversations": [
            {
              'role': 'user', 
              'content': '<image>\nHow many desserts are on the white plate?'
            }, 
            {
                'role': 'assistant', 
                'content': 'There are three desserts on the white plate.'
            },   
            {
                'role': 'user', 
                'content': 'What type of desserts are they?'
            },
            {
                'role': 'assistant', 
                'content': 'The desserts are cakes with bananas and pecans on top. They share similarities with donuts, but the presence of bananas and pecans differentiates them.'
            }, 
            {
                'role': 'user', 
                'content': 'What is the setting of the image?'}, 
            {
                'role': 'assistant', 
                'content': 'The image is set on a table top with a plate containing the three desserts.'
            },
        ]
    },
  ]
```

在训练 MiniCPM-V 2.0 模型时，可以使用 `finetune_lora.sh` 脚本。根据验证，最小的训练资源需求为3张RTX 3090显卡，同时需要使用 `cpu_low_memo` 的 `ds_config_zero2.json` 配置文件，因此需要依赖 DeepSpeed 框架。在文件开头需要进行如下设置：


### 设置步骤

1. **DeepSpeed 配置文件**：需要准备 `ds_config_zero2.json` 配置文件，设置为低内存使用模式（`cpu_low_memo`）。
2. **多GPU设置**：在脚本开头指定使用3张RTX 3090显卡进行训练。
3. **依赖安装**：确保环境中已经安装了 DeepSpeed，并正确配置路径和依赖。


```python
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2

# 设置 HF_HOME 环境变量 设置下载路径
export HF_HOME=/home/data/username/hf-models/
export HF_ENDPOINT=https://hf-mirror.com

GPUS_PER_NODE=3
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001


MODEL="openbmb/MiniCPM-V-2" # or openbmb/MiniCPM-V-2
DATA="./data/train_en_train.json" # json file
EVAL_DATA="./data/train_zh_train.json" # json file
LLM_TYPE="minicpm" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm
```


```python
!sh finetune_lora.sh
```


    /home/data/ckw/micromamba/envs/kewei-ai/lib/python3.12/site-packages/transformers/training_args.py:1474: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
      warnings.warn(
    [2024-06-08 09:05:58,136] [INFO] [comm.py:637:init_distributed] cdb=None
    [2024-06-08 09:05:58,136] [INFO] [comm.py:668:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
    /home/data/ckw/micromamba/envs/kewei-ai/lib/python3.12/site-packages/transformers/training_args.py:1474: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
      warnings.warn(
    [2024-06-08 09:05:58,153] [INFO] [comm.py:637:init_distributed] cdb=None
    /home/data/ckw/micromamba/envs/kewei-ai/lib/python3.12/site-packages/transformers/training_args.py:1474: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
      warnings.warn(
    [2024-06-08 09:05:58,205] [INFO] [comm.py:637:init_distributed] cdb=None
    The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
    The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
    The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
    Loading checkpoint shards: 100%|██████████████████| 2/2 [00:08<00:00,  4.06s/it]
    Loading checkpoint shards: 100%|██████████████████| 2/2 [00:08<00:00,  4.08s/it]
    Loading checkpoint shards: 100%|██████████████████| 2/2 [00:08<00:00,  4.17s/it]
    max_steps is given, it will override any value given in num_train_epochs
    Currently using LoRA for fine-tuning the MiniCPM-V model.
    max_steps is given, it will override any value given in num_train_epochs
    {'Total': 3458558752, 'Trainable': 733677856}
    llm_type=minicpm
    Loading data...
    max_steps is given, it will override any value given in num_train_epochs
      0%|                                                   | 0/998 [00:00<?, ?it/s]
      ...
    /home/data/ckw/micromamba/envs/kewei-ai/lib/python3.12/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
      warnings.warn(
    {'loss': 1.2035, 'grad_norm': 2.8456802368164062, 'learning_rate': 0.0, 'epoch': 0.01}
    {'loss': 1.2772, 'grad_norm': 4.5663909912109375, 'learning_rate': 3.010299956639811e-07, 'epoch': 0.01}
    {'loss': 1.3038, 'grad_norm': 4.5663909912109375, 'learning_rate': 3.010299956639811e-07, 'epoch': 0.02}
    {'loss': 1.4214, 'grad_norm': 4.5663909912109375, 'learning_rate': 3.010299956639811e-07, 'epoch': 0.02}
    {'loss': 1.279, 'grad_norm': 3.770563840866089, 'learning_rate': 4.771212547196623e-07, 'epoch': 0.03}
    ...
    {'loss': 1.0607, 'grad_norm': 3.499253988265991, 'learning_rate': 1e-06, 'epoch': 5.95}
    {'loss': 1.0804, 'grad_norm': 2.7949860095977783, 'learning_rate': 1e-06, 'epoch': 5.95}
    {'loss': 1.2137, 'grad_norm': 3.113947629928589, 'learning_rate': 1e-06, 'epoch': 5.96}
    {'loss': 0.9199, 'grad_norm': 3.8179216384887695, 'learning_rate': 1e-06, 'epoch': 5.96}
    {'loss': 1.0886, 'grad_norm': 2.0026695728302, 'learning_rate': 1e-06, 'epoch': 5.97}
    {'loss': 1.0101, 'grad_norm': 3.278071641921997, 'learning_rate': 1e-06, 'epoch': 5.98}
    {'train_runtime': 3244.7018, 'train_samples_per_second': 1.845, 'train_steps_per_second': 0.308, 'train_loss': 1.1752079226569327, 'epoch': 5.98}
    100%|█████████████████████████████████████████| 998/998 [54:04<00:00,  3.25s/it]
    /home/data/ckw/micromamba/envs/kewei-ai/lib/python3.12/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(


### 载入lora模型


```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Literal, Tuple
```


```python
@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None
```


```python
from peft import LoraConfig, get_peft_model, TaskType
def load_lora_config(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"
    )
    return get_peft_model(model, config)
```


```python
model = load_lora_config(model)
vpm_resampler_embedtokens_weight = torch.load(f"{path_to_adapter}/vpm_resampler_embedtokens.pt")
msg = model.load_state_dict(vpm_resampler_embedtokens_weight, strict=False)
```


```python
image = Image.open('屏幕截图 2024-05-13 104621.png').convert('RGB')
question = 'What is in the image? Please Speak English.'
msgs = [{'role': 'user', 'content': question}]

res, context, _ = model.chat(
    image=image,
    msgs=msgs,
    context=None,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7
)
print(res)
```
Output:

The image is a screenshot of the Windows File Explorer's '搜索' (search) menu. It includes options such as searching by file type, date modified or created, and other advanced search settings like excluding certain directories from searches using wildcards (*), including files with specific extensions ('* ...'), specifying subfolders to include in your query (`C: \My\Subfolder` for example), filtering based on size ranges within bytes/megabytes etc,'以及设置是否在文件夹或文件上显示修改时间。