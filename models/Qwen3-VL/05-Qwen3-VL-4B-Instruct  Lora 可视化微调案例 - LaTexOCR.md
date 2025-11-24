# Qwen/Qwen3-VL-4B-Instruct Lora Visual Fine-tuning Case - LaTexOCR

[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) is the strongest vision-language model in the Qwen series as of October 2025.

Qwen3-VL has significant improvements in text understanding and generation, visual perception and reasoning, extended context length, and enhanced spatial and video dynamic understanding. It features Dense and MoE architectures suitable for edge-to-cloud deployment, and has Instruct and reasoning-enhanced Thinking versions for flexible on-demand deployment.

For details, please visit [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct).

A notable enhancement is the OCR capability. The model card introduces that the model supports 32 languages (increased from 19); performs robustly under low light, blur, and tilt conditions; is more suitable for handling rare/ancient characters and jargon; and improves long document structure parsing.

In this article, we will briefly introduce how to perform Lora fine-tuning training on LaTeX_OCR using Qwen/Qwen3-VL-30B-A3B-Instruct and Qwen3-VL-4B-Instruct models based on frameworks such as transformers and peft, and use SwanLab to monitor the training process and evaluate the model effect.

> Note: The code used in this tutorial also supports 2.5 series models. For example, Qwen/Qwen2.5-VL-3B-Instruct can run normally on this script.

- **Training Code**: In the directory with the same name at the same level
- **Dataset**: [LaTeX_OCR](https://huggingface.co/datasets/linxy/LaTeX_OCR)
- **Model**: [Qwen3-VL-30B-A3B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Instruct/summary) & [Qwen3-VL-4B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct/summary)
- **Qwen/Qwen3-VL-30B-A3B-Instruct Video Memory Requirement**: 124+GB. If the video memory is insufficient, you can reduce the per_device_train_batch_size. The author uses two H20s for training, and the default batch size is 8. Based on this setting, it takes about 20 minutes. The batch size affects the time.
- **Qwen/Qwen2.5-VL-3B-Instruct Video Memory Requirement**: 20+GB. The author uses 1 H20 for training. You can also use a graphics card with 24 GB video memory, such as 3090, 4090, etc. When the batch size is 1, it takes 7 minutes, and when the batch size is 8, it takes 4 minutes.

**Table of Contents**

- [Qwen/Qwen3-VL-4B-Instruct Lora Visual Fine-tuning Case - LaTexOCR](#qwenqwen3-vl-4b-instruct-lora-visual-fine-tuning-case---latexocr)
  - [Environment Configuration](#environment-configuration)
  - [Prepare Dataset](#prepare-dataset)
  - [Model Download](#model-download)
  - [Integrate SwanLab](#integrate-swanlab)
  - [Introduction to Lora](#introduction-to-lora)
  - [Lora Configuration](#lora-configuration)
  - [Complete Fine-tuning Code](#complete-fine-tuning-code)
    - [Code](#code)
    - [Training Configuration](#training-configuration)
    - [Model Path Setting](#model-path-setting)
    - [Dataset Loading](#dataset-loading)
  - [Compare Model Output Before and After Fine-tuning](#compare-model-output-before-and-after-fine-tuning)
    - [Code](#code-1)
    - [Run Configuration](#run-configuration)
    - [Test Set Loading for Testing](#test-set-loading-for-testing)
  - [Model Fine-tuning Effect](#model-fine-tuning-effect)
    - [Qwen/Qwen3-VL-30B-A3B-Instruct](#qwenqwen3-vl-30b-a3b-instruct)
    - [Qwen/Qwen3-VL-4B-Instruct](#qwenqwen3-vl-4b-instruct)
    - [Fine-tuned Model Effect Display](#fine-tuned-model-effect-display)
      - [Qwen/Qwen3-VL-30B-A3B-Instruct](#qwenqwen3-vl-30b-a3b-instruct-1)
      - [Qwen/Qwen3-VL-4B-Instruct](#qwenqwen3-vl-4b-instruct-1)
      - [Summary](#summary)
  - [Supplementary Model Training Information](#supplementary-model-training-information)
  - [Common Error Solutions](#common-error-solutions)

## Environment Configuration

Ensure that your computer has at least one NVIDIA graphics card and the CUDA environment is installed. If you choose Qwen/Qwen3-VL-30B-A3B-Instruct for this training, it is relatively large and requires about 124GB of video memory. It is recommended to use two H20s to complete this experiment.

![Used Graphics Card](./images/05-1-1.png)

If computing resources are limited, it is recommended to use Qwen3-VL-4B-Instruct to complete this experiment. Only one 24 GB graphics card is needed to complete this experiment.
![Used Graphics Card](./images/05-1-2.png)

Install Python (version >= 3.12) and PyTorch capable of calling CUDA acceleration. The image uses Pytorch2.8.0 Python3.12 CUDA12.8.

![Qwen3 Model](./images/05-2.png)

Install third-party libraries related to Qwen3-VL fine-tuning using the following command:

```bash
python -m pip install --upgrade pip
```

Change the pypi source to accelerate library installation

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Some dependency libraries we mainly use this time are as follows:

```
notebook==7.4.7
numpy<2.0
datasets==4.2.0
peft==0.17.1
accelerate==1.10.1
mpmath==1.3.0
networkx==3.4.2
regex==2025.9.18
sympy==1.14.0
tokenizers==0.22.1
torch==2.8.0
torchvision>=0.23.0
transformers>=4.41.2
triton==3.4.0
qwen-vl-utils==0.0.14
matplotlib>=3.10.7
modelscope==1.30.0
python-dotenv>=1.1.1
swanlab
```

You can copy the above content and write it to the requirements.txt file, and then run the following command to install all dependency libraries:

```bash
pip install -r requirements.txt
```

## Prepare Dataset

The dataset used this time is [linxy/LaTeX_OCR](https://huggingface.co/datasets/linxy/LaTeX_OCR).
linxy/LaTeX_OCR is an open-source dataset containing five datasets.

1. small is a small dataset with 110 samples, used for testing.
2. full is a complete dataset of about 100k printed characters. In fact, the number of samples is slightly less than 100k because they used LaTeX's abstract syntax tree to remove many LaTeX that cannot be rendered.
3. synthetic_handwrite is a complete dataset of 100k handwritten characters, synthesized using handwritten fonts based on full formulas, which can be regarded as human handwriting on paper. The number of samples is actually slightly less than 100k, for the same reason as above.
4. human_handwrite is a smaller dataset of handwritten characters, more consistent with human handwriting on electronic screens. Mainly derived from CROHME. They have verified it with LaTeX's abstract syntax tree.
5. human_handwrite_print is a printed dataset from human_handwrite. The formula part is the same as human_handwrite, and the image part is rendered from the formula using LaTeX.

You can go to the source dataset page to view the subsets of the dataset. For example, the figure below shows the field names of each subset of the dataset.
Each dataset basically has only two fields, such as `text` and `image`.
![Dataset Subsets](images/05-3.png)

We can use the following code to load the dataset.

For the convenience of the experiment, you can select `small`, `full`, `synthetic_handwrite`, `human_handwrite`, or `human_handwrite_print` in `name`, and specify `train`, `validation`, `test`, etc. through `split`.

The following example shows how to load the training split and quickly check samples:

```python
from datasets import load_dataset

train_dataset = load_dataset("linxy/LaTeX_OCR", name="small", split="train")
print(train_dataset[2]["text"])
print(train_dataset[2])
print(len(train_dataset))
```

Output:

```text

\rho _ { L } ( q ) = \sum _ { m = 1 } ^ { L } \ P _ { L } ( m ) \ { \frac { 1 } { q ^ { m - 1 } } } .

{
'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=200x50 at 0x15A5D6CE210>,

'text': '\\rho _ { L } ( q ) = \\sum _ { m = 1 } ^ { L } \\ P _ { L } ( m ) \\ { \\frac { 1 } { q ^ { m - 1 } } } .'
}

50

```

If you need to obtain training, validation, and test splits at the same time, you can directly load the entire `DatasetDict`:

```python

from datasets import load_dataset

dataset = load_dataset("linxy/LaTeX_OCR", name="small")
print(dataset)
```

Output:

```text
DatasetDict({
    train: Dataset({
        features: ['image', 'text'],
        num_rows: 50
    })
    validation: Dataset({
        features: ['image', 'text'],
        num_rows: 30
    })
    test: Dataset({
        features: ['image', 'text'],
        num_rows: 30
    })
})
```

## Model Download

Before starting model training, we need to download the corresponding model.

To avoid model download failure due to network problems, we use modelscope to download the model.

The model addresses are at:

- [Qwen/Qwen3-VL-30B-A3B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Instruct/summary)：<https://modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Instruct/summary>
- [Qwen/Qwen3-VL-4B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct/summary)：<https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct/summary>

You can use the following command to download the model to the specified directory. The following takes downloading the model to the `./Qwen3-VL-30B-A3B-Instruct` directory as an example:

```bash
modelscope download --model Qwen/Qwen3-VL-30B-A3B-Instruct  --local_dir ./Qwen3-VL-30B-A3B-Instruct
```

Or use the following command to download the `Qwen/Qwen3-VL-4B-Instruct` model to the specified directory:

```bash
modelscope download --model Qwen/Qwen3-VL-4B-Instruct  --local_dir ./Qwen3-VL-4B-Instruct
```

> Note that Qwen/Qwen3-VL-30B-A3B-Instruct requires about 60GB of storage space, and Qwen/Qwen3-VL-4B-Instruct requires about 8GB of storage space. Before starting the download, if you need to fine-tune the 30 B model, ensure that the free disk space is more than 65 GB. If it is a 4 B model, the storage space size should be more than 10 GB.

If you need to use my code to run directly on AutoDL, you need to download the model to `/root/autodl-fs/Qwen3-VL-30B-A3B-Instruct`.

fs will occupy user space for a long time. If the user does not clean it up in time, fees will be deducted continuously, so I suggest you switch to auto-tmp. Note that after switching to auto-tmp, you need to modify the code for loading the model.

## Integrate SwanLab

SwanLab has been integrated with Transformers. The usage is to add a SwanLabCallback instance to the callbacks parameter of Trainer, which can automatically record hyperparameters and training indicators. The simplified code is as follows:

```python
from swanlab.integration.transformers import SwanLabCallback
from transformers import Trainer

swanlab_callback = SwanLabCallback()

trainer = Trainer(
    ...
    callbacks=[swanlab_callback],
)
```

To use SwanLab for the first time, you need to register an account on the [official website](https://swanlab.cn/space/~), then copy your API Key on the user settings page, and paste it when prompted to log in at the start of training. No need to log in again subsequently.

Note: The use of SwanLab is free for personal use.

The page after login is shown in the figure below.

![SwanLab](images/05-4.png)

Click on one of them, you can see the specific experimental training details.

![Experiment](images/05-5.png)

Clicking on one of them will display the specific loss changes and other indicators. Of course, SwanLab has other indicators that can be monitored. You can check the documentation on the official website.

SwanLab Address: <https://swanlab.cn/>

In my code, I set the api_key to load from environment variables, so you need to create a file named .env and add SWAN_LAB=Your API Key.

```text
SWAN_LAB=Your API Key
```

The api_key can be obtained at the position shown in the figure below.
![Get API Key](images/05-6.png)

## Introduction to Lora

The full name of Lora is Low-Rank Adaptation.
Traditional model fine-tuning methods, i.e., full parameter fine-tuning, require updating all parameters in the model.

The core idea of Lora is that the weight change matrix $\Delta W$ can be approximately decomposed into the product of two smaller matrices, and then only the two smaller matrices are updated.

It does not add extra computational latency during inference. This is because its bypass structure can be merged back into the original weight matrix before inference.

That is to say, we can merge the adapter weights into the backbone network through simple matrix addition $W' = W_0 + BA$, thereby obtaining a new weight matrix.

> "LoRA: Low-Rank Adaptation of Large Language Models" paper address: <https://arxiv.org/abs/2106.09685>

## Lora Configuration

```python
lora_config_dict = {
        "lora_rank": 128,
        "lora_alpha": 16,
        "lora_dropout": 0,
    }

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=target_modules,
        inference_mode=False,
        r=lora_config_dict["lora_rank"],
        lora_alpha=lora_config_dict["lora_alpha"],
        lora_dropout=lora_config_dict["lora_dropout"],
        bias="none",
    )
```

The above is the code for creating Lora configuration. If you need to adjust, you can adjust lora_config_dict and target_modules, mainly setting them.

target_modules: Which modules in the model the LoRA adapter should act on. Here it is set to ["q_proj", "k_proj", "v_proj", "o_proj"].

These are the core linear projection layers in the Transformer model's self-attention mechanism, responsible for generating queries, keys, values, and outputs.

r=128: This is the rank of LoRA.

lora_alpha=16: This is the scaling factor alpha of LoRA, which is α in the formula.

lora_dropout=0: This parameter sets the dropout rate of the LoRA layer.
The complete forward propagation formula in the paper is as follows.

$$h=W_{0}x+\Delta Wx=W_{0}x+BAx$$

α is a constant. The benefit of doing this is that when changing the size of rank r, the need to readjust hyperparameters can be reduced.

The forward propagation formula with α is as follows.

$$h = W_{0}x + \frac{α}{r}BAx$$

## Complete Fine-tuning Code

### Code

<details><summary>Click to expand/collapse the complete fine-tuning code</summary>

```python
import os

import torch
from typing import Any, Dict, List

from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    TrainingArguments,
    Trainer,
    AutoProcessor,
    AutoTokenizer,
    AutoConfig,
)
import importlib
import matplotlib.pyplot as plt
from swanlab.integration.transformers import SwanLabCallback
from dotenv import load_dotenv


class Qwen3VLDataCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_id_tensors = [
            torch.as_tensor(sample["input_ids"], dtype=torch.long) for sample in features
        ]
        attention_tensors = [
            torch.as_tensor(sample["attention_mask"], dtype=torch.long) for sample in features
        ]
        label_tensors = [
            torch.as_tensor(sample["labels"], dtype=torch.long) for sample in features
        ]

        max_length = max(t.size(0) for t in input_id_tensors)
        pad_id = (
            self.tokenizer.pad_token_id
            if getattr(self.tokenizer, "pad_token_id", None) is not None
            else self.tokenizer.eos_token_id
        )
        if pad_id is None:
            raise ValueError("Both pad_token_id and eos_token_id are None, cannot perform padding.")

        input_ids = torch.full((len(features), max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_length), dtype=torch.long)
        labels = torch.full((len(features), max_length), -100, dtype=torch.long)

        for idx, (ids, attn, lbl) in enumerate(zip(input_id_tensors, attention_tensors, label_tensors)):
            length = ids.size(0)
            input_ids[idx, :length] = ids
            attention_mask[idx, :length] = attn
            labels[idx, :length] = lbl

        pixel_tensors = []
        for sample in features:
            pv = sample["pixel_values"]
            if not isinstance(pv, torch.Tensor):
                pv = torch.tensor(pv, dtype=torch.float32)
            pixel_tensors.append(pv)
        pixel_values = torch.cat(pixel_tensors, dim=0)

        image_grid_thw = torch.stack(
            [torch.as_tensor(sample["image_grid_thw"], dtype=torch.long).view(-1) for sample in features], dim=0
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


PROMPT_TEXT = "Transcribe the LaTeX of this image."


def process_func(example, tokenizer, processor):
    MAX_LENGTH = 8192
    image = example["image"]
    output_content = example["text"]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        do_resize=True,  
    )

    instruction_input_ids = inputs["input_ids"][0]

    instruction_attention_mask = inputs["attention_mask"][0]

    instruction_pixel_values = inputs["pixel_values"]

    instruction_image_grid_thw = inputs["image_grid_thw"][0]

    response = tokenizer(f"{output_content}", add_special_tokens=False)
    response_input_ids = response["input_ids"]
    response_attention_mask = response.get(
        "attention_mask", [1] * len(response_input_ids)
    )

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        if not response_input_ids or response_input_ids[-1] != eos_token_id:
            response_input_ids = response_input_ids + [eos_token_id]
            response_attention_mask = response_attention_mask + [1]
    else:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Need to define eos_token_id or pad_token_id to end the response sequence.")
        response_input_ids = response_input_ids + [pad_token_id]
        response_attention_mask = response_attention_mask + [1]

    input_ids = instruction_input_ids + response_input_ids
    attention_mask = instruction_attention_mask + response_attention_mask
    labels = (
        [-100] * len(instruction_input_ids) + response_input_ids
    )
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": instruction_pixel_values,
        "image_grid_thw": instruction_image_grid_thw,
    }


def main():
    load_dotenv()
    os.environ["SWANLAB_API_KEY"] = os.getenv("SWAN_LAB")

    data_fraction = 0.002

    ds = load_dataset("linxy/LaTeX_OCR", "synthetic_handwrite")

    ds = ds.shuffle(seed=222)

    train_data = ds["train"].select(range(int(len(ds["train"]) * data_fraction)))
    print(f"Training data size: {len(train_data)}")
    test_data = ds["test"].select(range(int(len(ds["test"]) * data_fraction)))
    print(f"Test data size: {len(test_data)}")

    # model_id = "/root/autodl-fs/Qwen3-VL-30B-A3B-Instruct"
    # model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    # output_dir = "/root/autodl-fs/output/Qwen3-VL-30B"
    
    model_id = "/root/autodl-tmp/Qwen3-VL-4B-Instruct"
    output_dir = "/root/autodl-tmp/Qwen3-VL-4B"
    

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=os.environ.get("HF_HOME", "./"), use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=os.environ.get("HF_HOME", "./"), use_fast=False)

    config = AutoConfig.from_pretrained(model_id, cache_dir=os.environ.get("HF_HOME", "./"), trust_remote_code=True)
    arch = (config.architectures or [None])[0]
    module_name = f"transformers.models.{config.model_type}.modeling_{config.model_type}"
    module = importlib.import_module(module_name)
    model_cls = getattr(module, arch)
    model = model_cls.from_pretrained(
        model_id,
        cache_dir=os.environ.get("HF_HOME", "./"),
        device_map="auto",
        trust_remote_code=True,
    )

    model.to(dtype=torch.bfloat16)

    model.config.use_cache = False

    map_kwargs = {"tokenizer": tokenizer, "processor": processor}
    train_dataset = train_data.map(
        process_func,
        remove_columns=train_data.column_names,
        fn_kwargs=map_kwargs,
    )

    lora_config_dict = {
        "lora_rank": 128,
        "lora_alpha": 16,
        "lora_dropout": 0,
    }

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=target_modules,
        inference_mode=False,
        r=lora_config_dict["lora_rank"],
        lora_alpha=lora_config_dict["lora_alpha"],
        lora_dropout=lora_config_dict["lora_dropout"],
        bias="none",
    )

    peft_model = get_peft_model(model, config)

    peft_model.enable_input_require_grads()

    swanlab_callback = SwanLabCallback(
        project="Qwen3-VL-finetune",
        experiment_name="qwen3-vl-latex-ocr",
        config={
            "model": model_id,
            "dataset": "linxy/LaTeX_OCR",
            "prompt": PROMPT_TEXT,
            "train_data_number": len(train_data),
            "lora_rank": lora_config_dict["lora_rank"],
            "lora_alpha": lora_config_dict["lora_alpha"],
            "lora_dropout": lora_config_dict["lora_dropout"],
        },
    )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8, # Batch size per GPU
        gradient_accumulation_steps=1, # Gradient accumulation steps
        logging_steps=10, 
        logging_first_step=5, 
        num_train_epochs=8, # Number of training epochs
        save_steps=50, # Save model every X steps
        save_total_limit=3, # Maximum number of models to save
        learning_rate=1e-4, # Learning rate
        gradient_checkpointing=True, # Gradient checkpointing
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        report_to="none",
    )

    eval_dataset = test_data.map(
        process_func,
        remove_columns=test_data.column_names,
        fn_kwargs=map_kwargs,
    )

    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Qwen3VLDataCollator(tokenizer=tokenizer),
        callbacks=[swanlab_callback],
    )

    trainer.train()

    logs = trainer.state.log_history
    steps = [log['step'] for log in logs if 'loss' in log]
    losses = [log['loss'] for log in logs if 'loss' in log]
    plt.plot(steps, losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss (Qwen3-VL-30B)')

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "training_loss.png"))

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
```

</details>

### Training Configuration

The training configuration is as follows:

```python
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8, # Batch size per GPU
    gradient_accumulation_steps=1, # Gradient accumulation steps
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=8, # Number of training epochs
    save_steps=50, # Save model every X steps
    save_total_limit=3, # Maximum number of models to save
    learning_rate=1e-4, # Learning rate
    gradient_checkpointing=True, # Gradient checkpointing
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
)
```

### Model Path Setting

The model path setting part is:

```python
# model_id = "/root/autodl-fs/Qwen3-VL-30B-A3B-Instruct"
# model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
# output_dir = "/root/autodl-fs/output/Qwen3-VL-30B"

model_id = "/root/autodl-tmp/Qwen3-VL-4B-Instruct"
output_dir = "/root/autodl-tmp/Qwen3-VL-4B"
```

You can modify based on my original code and replace it with the model you want to fine-tune.

### Dataset Loading

```python
data_fraction = 0.002

ds = load_dataset("linxy/LaTeX_OCR", "synthetic_handwrite")

ds = ds.shuffle(seed=222)

train_data = ds["train"].select(range(int(len(ds["train"]) * data_fraction)))
print(f"Training data size: {len(train_data)}")
test_data = ds["test"].select(range(int(len(ds["test"]) * data_fraction)))
print(f"Test data size: {len(test_data)}")
```

The dataset loading in the model training part is mainly proportionally sampled through the data_fraction parameter. Because loading all at once requires a long time for fine-tuning, you can use this parameter to proportionally sample the data, which also allows for quick training and timely parameter optimization through training effects.

## Compare Model Output Before and After Fine-tuning

### Code

We can use the following code to compare the model output before and after fine-tuning.
<details><summary>Click to view code</summary>

```python

import os
import sys
from typing import List, Tuple

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
import importlib

from qwen_vl_utils import process_vision_info


PROMPT_TEXT = "Transcribe the LaTeX of this image."
# Use local base model and LoRA directory
# BASE_MODEL_ID = "/root/autodl-fs/Qwen3-VL-30B-A3B-Instruct"
# PEFT_DIR = "/root/autodl-fs/output/Qwen3-VL-30B"
BASE_MODEL_ID = "/root/autodl-tmp/Qwen3-VL-4B-Instruct"
PEFT_DIR = "/root/autodl-tmp/Qwen3-VL-4B"
# Whether to merge LoRA in memory (not saving to disk)
MERGE_LORA_IN_MEMORY = True
NUM_TEST_SAMPLES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32


def load_backbone(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=os.environ.get("HF_HOME", "./"), use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=os.environ.get("HF_HOME", "./"), use_fast=False, trust_remote_code=True)

    config = AutoConfig.from_pretrained(model_id, cache_dir=os.environ.get("HF_HOME", "./"), trust_remote_code=True)
    arch = (config.architectures or [None])[0]
    module_name = f"transformers.models.{config.model_type}.modeling_{config.model_type}"
    module = importlib.import_module(module_name)
    model_cls = getattr(module, arch)

    model = model_cls.from_pretrained(
        model_id,
        cache_dir=os.environ.get("HF_HOME", "./"),
        device_map="auto" if DEVICE.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.to(dtype=DTYPE)
    
    return model, tokenizer, processor


def load_lora_model(peft_dir: str, base_model_id: str = BASE_MODEL_ID):
    if not os.path.isdir(peft_dir):
        raise FileNotFoundError(f"Fine-tuned model directory not found: {peft_dir}")

    # Base
    base_model, _base_tok, _base_proc = load_backbone(base_model_id)

    # Load LoRA first
    peft_model = PeftModel.from_pretrained(base_model, peft_dir)
    model = peft_model
    if MERGE_LORA_IN_MEMORY:
        try:
            model = peft_model.merge_and_unload()
            print("LoRA memory merge successful.")
        except Exception:
            print("Warning: LoRA memory merge failed, continuing with unmerged model.")
            # If merge fails, revert to unmerged model
            model = peft_model
    model.to(dtype=DTYPE)
    model.eval()


    # tokenizer/processor prioritize reading from LoRA directory to ensure chat_template and vocabulary consistency
    tokenizer = AutoTokenizer.from_pretrained(peft_dir, cache_dir=os.environ.get("HF_HOME", "./"), use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(peft_dir, cache_dir=os.environ.get("HF_HOME", "./"), use_fast=False, trust_remote_code=True)
    return model, tokenizer, processor


def build_inputs(processor, image, prompt_text: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, do_resize=True)
    return inputs


def ensure_block_dollars(text: str) -> str:
    if text is None:
        return "$$$$"
    s = str(text).strip()
    if s.startswith("$$") and s.endswith("$$"):
        return s
    if s.startswith("$") and s.endswith("$") and not s.startswith("$$") and not s.endswith("$$"):
        inner = s[1:-1].strip()
        return f"$${inner}$$"
    if s.count("$$") >= 2:
        return s
    return f"$${s}$$"


@torch.inference_mode()
def generate_answer(model, tokenizer, processor, image, max_new_tokens: int = 512) -> str:
    inputs = build_inputs(processor, image, PROMPT_TEXT)

    input_ids = torch.as_tensor(inputs["input_ids"], device=DEVICE)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = torch.as_tensor(attention_mask, device=DEVICE)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

    pixel_values = inputs.get("pixel_values")
    pixel_values = torch.as_tensor(pixel_values, device=DEVICE)

    image_grid_thw = inputs.get("image_grid_thw")
    image_grid_thw = torch.as_tensor(image_grid_thw, device=DEVICE)

    gen_kwargs = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
    }
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask
    if image_grid_thw is not None:
        gen_kwargs["image_grid_thw"] = image_grid_thw

    outputs = model.generate(**gen_kwargs)
    gen_seq = outputs[0].tolist()
    prompt_len = input_ids.shape[1]
    gen_ids = gen_seq[prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def main():
    print("Loading dataset linxy/LaTeX_OCR (synthetic_handwrite)...")
    ds = load_dataset("linxy/LaTeX_OCR", "synthetic_handwrite")
    ds = ds.shuffle(seed=222)
    # test_split = ds["train"].select(range(NUM_TEST_SAMPLES))
    test_split = ds["test"].select(range(NUM_TEST_SAMPLES))

    print("Loading base model...")
    base_model, base_tokenizer, base_processor = load_backbone(BASE_MODEL_ID)
    try:
        if hasattr(base_model, "gradient_checkpointing"):
            base_model.gradient_checkpointing_disable()
        if hasattr(base_model, "config"):
            base_model.config.use_cache = True
        if hasattr(base_model, "generation_config") and base_model.generation_config is not None:
            base_model.generation_config.use_cache = True
    except Exception:
        pass
    base_model.eval()

    print(f"Loading LoRA fine-tuned model from: {PEFT_DIR}")
    try:
        lora_model, lora_tokenizer, lora_processor = load_lora_model(PEFT_DIR, BASE_MODEL_ID)
        try:
            if hasattr(lora_model, "gradient_checkpointing"):
                lora_model.gradient_checkpointing_disable()
            if hasattr(lora_model, "config"):
                lora_model.config.use_cache = True
        except Exception:
            pass
    except Exception as e:
        print(f"Failed to load fine-tuned model: {e}")
        print("Comparing with base model only.")
        lora_model = None
        lora_tokenizer = base_tokenizer
        lora_processor = base_processor

    print(f"\n===== Inference Comparison on {NUM_TEST_SAMPLES} samples =====\n")
    for idx, sample in enumerate(test_split):
        image = sample["image"]
        gt = sample.get("text", "")
        print(f"[Sample {idx}]------------------------------")
        print(f"GT: {ensure_block_dollars(gt)}")

        base_pred = ensure_block_dollars(generate_answer(base_model, base_tokenizer, base_processor, image))
        print(f"Base: {base_pred}")

        if lora_model is not None:
            lora_pred = ensure_block_dollars(generate_answer(lora_model, lora_tokenizer, lora_processor, image))
            print(f"LoRA: {lora_pred}")
        else:
            print("LoRA: <not loaded>")

        print()


if __name__ == "__main__":
    main()


```

</details>

### Run Configuration

The model path setting and other settings are in the beginning of the file, as follows.

```python
PROMPT_TEXT = "Transcribe the LaTeX of this image." # Prompt used.

# Use local base model and LoRA directory
# BASE_MODEL_ID = "/root/autodl-fs/Qwen3-VL-30B-A3B-Instruct"
# PEFT_DIR = "/root/autodl-fs/output/Qwen3-VL-30B"
BASE_MODEL_ID = "/root/autodl-tmp/Qwen3-VL-4B-Instruct"
PEFT_DIR = "/root/autodl-tmp/Qwen3-VL-4B"
# Whether to merge LoRA in memory (not saving to disk)
MERGE_LORA_IN_MEMORY = True
NUM_TEST_SAMPLES = 5 # Number of test samples used
```

### Test Set Loading for Testing

```python
ds = load_dataset("linxy/LaTeX_OCR", "synthetic_handwrite")
ds = ds.shuffle(seed=222)
# test_split = ds["train"].select(range(NUM_TEST_SAMPLES))
test_split = ds["test"].select(range(NUM_TEST_SAMPLES))
```

This is the code used to load the dataset for testing, where NUM_TEST_SAMPLES is used to control the number of samples.

## Model Fine-tuning Effect

### Qwen/Qwen3-VL-30B-A3B-Instruct

The figure below is the fine-tuning chart of the `Qwen/Qwen3-VL-30B-A3B-Instruct` model, using a batch size of 8.

![Model Fine-tuning Chart](images/05-7-1.png)

From the effect of the chart, the loss is basically in a stable decreasing state, proving that our training effect is fitting the dataset.

### Qwen/Qwen3-VL-4B-Instruct

The figure below is the fine-tuning chart of the `Qwen/Qwen3-VL-4B-Instruct` model, with a batch size of 1.

![Model Fine-tuning Chart](images/05-7-2.png)

The figure below is the fine-tuning chart of the `qwen/Qwen3-VL-4B-Instruct` model, with a batch size of 8.

![Model Fine-tuning Chart](images/05-7-3.png)

### Fine-tuned Model Effect Display

#### Qwen/Qwen3-VL-30B-A3B-Instruct

Comparison of model effects before and after fine-tuning 1, GT is the ground truth, Base is the base model, and LoRA is the fine-tuned model.

![Comparison of Model Effects Before and After Fine-tuning](images/05-8.png)

Comparison of model effects before and after fine-tuning 2.

![Comparison of Model Effects Before and After Fine-tuning](images/05-9.png)
Comparison of model effects before and after fine-tuning 3.

![Comparison of Model Effects Before and After Fine-tuning](images/05-10.png)

#### Qwen/Qwen3-VL-4B-Instruct

Comparison of model effects before and after fine-tuning 1. Here is the effect trained using a batch size of 1. It can be seen that the extraction effect is relatively poor.
![Comparison of Model Effects Before and After Fine-tuning](images/05-10-1.png)

Comparison of model effects before and after fine-tuning 2. Here is the effect trained using a batch size of 8. It can be seen that the effect is much better than before.

![Comparison of Model Effects Before and After Fine-tuning](images/05-10-2.png)

#### Summary

The above shows the comparison of model effects before and after fine-tuning.

Although it seems that there is an improvement in the before and after comparison in some examples of Qwen/Qwen3-VL-30B-A3B-Instruct, I also found other problems with the model after fine-tuning.

For example, occasionally some examples are not as good as the model before fine-tuning. I think it is caused by the model being a bit overfitted. Because the fine-tuning chart shows that our training epochs are a bit too many.

In this model fine-tuning, I not only fine-tuned once, but multiple times. I also tried various parameters and different sub-datasets for training in this training, so I also found some useful observations during the training process.

Originally, I wanted to use the handwritten formula recognition dataset for training.

However, during the first training using the handwritten formula dataset, the model fitting did not seem to be good, because there may be many ways to write a character in the handwritten formula dataset. If I train with only a small amount of dataset, the model fine-tuning effect is not good, so I switched back to non-handwritten formulas.

Later, I used the small sub-training set for fine-tuning. At first, I only set one epoch of fine-tuning, but the effect was not good. The content output by the model before and after fine-tuning was almost exactly the same, and the same for two epochs.

Then I slowly adjusted the training epochs. When the epoch reached 9, it was obvious that the loss was no longer going down all the time, but instead rose in some parts. I thought I would set the training epoch to 8 first.

Another point is the setting of batch size. This parameter has a great impact on the training results. From Qwen/Qwen3-VL-4B-Instruct, it can be seen that when the batch size is set to 1, the model training effect is worse. I estimate it is overfitted. When the batch size is set to 8, the effect is relatively better.

After that, I thought it was caused by the batch size factor, so I switched the dataset back to the handwritten dataset and then fine-tuned it. The result was as I expected, and the model effect improved significantly after fine-tuning.

In summary, even the performance of Qwen/Qwen3-VL-30B-A3B-Instruct before fine-tuning is significantly improved on the test set after fine-tuning. In the first five test cases before fine-tuning, only one recognition result was correct, which is 20% accuracy. The fine-tuned model has about 60% accuracy on the test set.

If full fine-tuning is performed on the dataset, I think the model effect can reach a better accuracy. Interested partners can try it.

Interested readers can try other parameter settings, such as rank, lora_alpha, learning rate, batch_size, etc., and then compare the differences before and after adjustment.

## Supplementary Model Training Information

![GPU Usage](images/05-11.png)

![GPU Usage](images/05-12.png)

![Environment Information](images/05-13.png)

![System Hardware](images/05-14.png)

![Card](images/05-15.png)

## Common Error Solutions

![numpy error](images/05-16.png)
If you encounter the error shown in the figure above, that is:

```bash
pyarrow.lib.ArrowTypeError: Did not pass numpy.dtype object
```

In this case, I think it is caused by the numpy version.
You can use the following command to fix the version:

```bash
pip install --upgrade numpy
```

Run this command, and then re-run the code, it should be able to fix this error.
