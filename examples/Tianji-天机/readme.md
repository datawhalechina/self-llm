# Introduction to Tianji

Want to learn large language model applications from scratch? Want to copy and paste code directly? Tianji meets all your needs; even, you can learn something unexpected.

Chinese culture emphasizes human sophistication, involving complex social rules, etiquette, and interpersonal relationships. The Social team provides coping methods, dialogue cases, and scenario simulations based on various Chinese-style cultural scenarios through in-depth understanding of Chinese context, cultural characteristics, and rich corpus.

Currently, the application scenarios supported by Tianji include: how to give **blessings, toasts, treat guests, give gifts, interpersonal communication, resolve embarrassment, and deal with conflicts**, etc., and more LLM EQ improvement plans suggested by community partners are in production~

In addition to treating this project as a fun social large model, you can also treat it as a complete full-stack large language model application introductory learning repository! You can learn **prompt engineering, agent creation, model fine-tuning, RAG data cleaning and use, and code specifications**... and other large language model application knowledge you need. After learning this project, you can quickly transform it into your own new large language model project. We look forward to you becoming the next master of large model application development!

> *Tianji Open Source Link:* [*Link*](https://github.com/SocialAI-tianji/Tianji)
> *Tianji Official Website:* [*Link*](https://socialai-tianji.github.io/socialai-web/)
> *Tianji Knowledge Base Version Application Address:* [*Link*](http://120.76.130.14:6006/knowledges/)
> *Tianji Prompt Version Application Address:* [*Link*](http://120.76.130.14:6006/prompt/)
> *Tianji Video Introduction:* [*Link*](https://www.bilibili.com/video/BV1cvbyefEfp)

Due to the large amount of content and time constraints, this tutorial will only take you to briefly see how to quickly run Tianji's "Blessing" fine-tuning model. More interesting content is waiting for you to explore in the Tianji project and documentation!

## Implementing the Blessing Fine-tuning Model

Next, let's quickly implement a blessing fine-tuning model of our own. Based on Xtuner Qlora, we will explain in detail how to fine-tune the Tianji blessing module for internLM2, covering the entire process including data manufacturing, inference fine-tuning, etc.

In this fine-tuning demonstration, we use the internlm2-chat-7b model. You need to prepare a machine with 24G VRAM for fine-tuning (3090 is sufficient).

The first step in fine-tuning a model is to prepare high-quality training data. For a blessing model, you need to collect data on various blessings. The data source can be public blessing datasets, social media, e-books, or any text containing rich blessings.

After preparing the data source and obtaining the corresponding data, you need to use the data text for data manufacturing (such as the few shot demonstrated below, but this is just a minimal example. For real data manufacturing, you need to use a data "knowledge" chunk to generate corresponding QA pairs. This is the data we expect to get in the end.

Therefore, theoretically, the best data is to use this existing knowledge and use a smarter large model to obtain high-precision reply QA pair data based on this knowledge. Some people also achieve format extraction by extracting novel text dialogues through large models, but in short, what you need is a smart large model with unlimited firepower to help you clean text data.

When you successfully get through fine-tuning, you will find that **the truly complex work is in cleaning data, processing, generating data, and classifying data**. These are the **biggest difficult problems** that affect the final effect.

Here it is recommended that you use a local llm for data cleaning (unless you are wealthy), otherwise the api key will easily be used up in minutes. You can deploy local llama3-chinese or qwen for data manufacturing work.

Next, let's look at how to perform data manufacturing:

### Data Processing

#### Data Manufacturing

Before cleaning data, please ensure that you have installed the corresponding SDK such as zhipuai and openai SDK. Install and run directly.

```python
from zhipuai import ZhipuAI
import time
import json
import random
import datetime

# zhipuai
# Fill in your own APIKey here
# zhipu_api_key = ""
# client = ZhipuAI(api_key=zhipu_api_key)
# def get_data_zhipu(content):
#     response = client.chat.completions.create(
#         model="glm-4",  # Fill in the model name to be called
#         messages=[
#             {"role": "system", "content": "You are now a blessing master who is proficient in speech, loves others, respects elders, and is rich in literary talent. Please edit a text to express the blessing of the corresponding scene"},
#             {"role": "user",
#              "content": content,
#              "temperature": 1} # Diversified output
#         ],
#     )
#     res = response.choices[0].message.content
#     return res

# deepseek
from openai import OpenAI
deepseek_key = ""  # Fill in the deepseek key here
client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com/v1")
def get_data_ds(content):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are now a blessing master who is proficient in speech, loves others, respects elders, and is rich in literary talent. Please edit a text to express the blessing of the corresponding scene"},
            {"role": "user",
            "content": content,
            "temperature": 1} # Diversified output
        ]
    )
    res = response.choices[0].message.content
    return res

# Use large models to supplement different objects. Currently 28 types
name_list = ['Teacher Zhao', 'Uncle (Mother\'s brother)', 'Uncle (Father\'s elder brother)', 'President Li', 'Neighbor Aunt Zhao', 'Mother', 'Sister', 'Younger Sister', 'Brother', 'Younger Brother', 'Grandfather', 'Grandmother', 'Maternal Grandfather',
        'Maternal Grandmother', 'Aunt (Father\'s elder brother\'s wife)', 'Uncle', 'Aunt', 'Cousin (Male)', 'Cousin (Female)', 'Cousin (Male)', 'Cousin (Female)', 'Mentor', 'Classmate', 'Colleague', 'Leader',
        'Neighbor', 'Boss', 'Doctor', ]

# Use large models to supplement corresponding scenes. Currently 18 types
scenes = ['Birthday', 'Spring Festival', 'Lantern Festival', 'Dragon Boat Festival', 'Qixi Festival', 'Mid-Autumn Festival',
            'Double Ninth Festival', 'New Year\'s Eve', 'Laba Festival','Successful Negotiation','Housewarming', 'Anniversary' ,'Happy Wedding' ,'Family Harmony', 'Good Results in Competition' ,'Get Rich','Job Promotion ','Recovery', ]

# Use large models to supplement different styles, add more fewshot to create better data
styles = {
    "Little Red Book":
    {
        "style_temple":"Little Red Book style, add 1-2 emoji expressions to each strip to increase interest.\n### Note, you must refer to the artistic style of the following sentences for blessing writing (Note! Only look at the sentence making style), and the end of the blessing should bring a modal particle. Reference sentences are: {} ###",
        "if_example":True,
        "examples":
        [
    'Reciting your name silently, wishing you a bright future, brilliant as the galaxy. May what you spend be called auspicious time, and what you get be called wish fulfilled!',
    'Hope that at the end of the year, I send you winter peace, peace and joy, and everything goes well.',
    'Hope you don\'t have to run to the sea to see the spring flowers bloom; don\'t have to wander, and can meet the companion of a lifetime!',
    'Wish us well in spring, summer, autumn and winter, wish you talk freely, wish you romance, wish you meet yourself in the wind, and only joy remains thereafter.',
    'Hope you can love clearly, hate directly, like sincerely, stand in the sun openly, praise yourself loudly without shame, and learn to love yourself!',
    'Glory ahead, warmth behind, everything in the past is a prologue.',
    'May the people you miss be safe and happy. May the things you think of go well!',
        ]
    },
    "Normal":
    {
        "style_temple":"Normal style, just be polite",
        "if_example":False,
        "examples":[]
    },
    "Serious":
    {
        "style_temple":"Business serious style, required for use in workplace or elder blessings, appear polite, capable, sentences can be longer",
        "if_example":False,
        "examples":[]
    }
}

random_finalprompt_sentence = [
    '', # Default case
    'The answer does not need to include the object title and scene information, nor does it need to include "May you" "Wish you" (Object title and wish you need to appear for your own elders),',
    'The answer does not need to include the object title and scene information,',
    'The answer does not need to include "May you" "Wish you"',
]
final_prompt = """
The word count of the blessing is less than {}. \n
Please write a blessing copy that fits the object's identity and scene atmosphere according to the object title and scene. The required style is: {} \n, note that there should be no title mixed in, the object title is: {}, the blessing scene is: {}. \n
{} Use different tones (respectful, humorous, close) according to different objects, please return the blessing text directly, do not say any other words:
"""

if __name__ == "__main__":
    ##### Configure here #####
    roop_count = 2
    now_count = 0
    stylename = "Little Red Book" # Little Red Book, Normal, Serious
    output_number_limit = 50 # Limit answer output length, 100 for serious, less than 20 for normal
    ##### Configure here #####

    for roop in range(roop_count):
        conversations = []
        for name in name_list:
            for scene in scenes:
                try:
                    if styles[stylename]['if_example']:
                        style_prompt = styles[stylename]['style_temple'].format(random.choice(styles[stylename]['examples']))
                    else:
                        style_prompt = styles[stylename]['style_temple']
                    input_prompt = final_prompt.format(output_number_limit, style_prompt, name, scene,random.choice(random_finalprompt_sentence))

                    response = get_data_ds(input_prompt)
                    now_count += 1

                    if '\n' in str(response):
                        response = str(response).split('\n')[0]

                    print(name,scene,'response:',response)
                    print("Current generated count:", now_count)
                    if stylename == 'Normal':
                        # Default without style specification
                        _input_prompt = f"Wish {name} {scene}"
                    else:
                        _input_prompt = f"Wish {name} {scene}, {stylename} style"
                    print("input:",_input_prompt)

                    conversation = {
                        "conversation": [
                            {
                                "system": "You are now a blessing master, help me send corresponding blessings for different people, things, and festivals",
                                "src_input":input_prompt,
                                "style_name":stylename,
                                "input": _input_prompt,
                                "output": str(response).replace('\"','')
                            }
                        ]
                    }

                    # Add dialogue to list
                    conversations.append(conversation)
                except Exception as e:
                    print(e)
                    continue

        now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        file_path = f"./wishes_{stylename}_{now_time}.json"
        with open(file_path, "w", encoding='utf8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=4)

```

**Note**, why do we need to replace input with the format of `f"Wish {name} {scene}"` here - because the input here needs to simulate human input as much as possible, and cannot be the input when manufacturing data. In addition, we have set three styles here: Normal, Little Red Book, Serious; expect expected output when style name trigger is added.

This is just the simplest example. How to change the grammatical style after generation according to the length of the blessing, and how to be closer to the style of real people; these all require high-quality data + good data manufacturing methods to obtain.

For example, if we want to control the length and grammatical style of the blessing here, first we need to add new blessing length control conditions (such as Little Red Book and Normal style here) to the previous manufacturing conditions (such as the previous object and scene are a condition), and at this time the sentences referenced by few shot should also be different, so as to ensure that our data manufacturing llm can return results of expected length. If it is to control the grammatical style, we need to crawl a large number of literary books, Little Red Book and other copies written by real humans for cleaning, and then use these as few shot to get strict returns. We need to use strict prompt words to let the model write similar sentences or separately fine-tune a model version belonging to this literary category for manufacturing corresponding data (sometimes few shot instruction following is not so useful).

💡The part with `random_xxxxxxx_sentence` in the code indicates that this is a randomness injection list. We can maintain some sentences to improve randomness (such as modification of additional conditions) so that the results returned by the large model are more distinctive.

If run successfully, you will see output similar to the following. After waiting for a while, you will get the local json file `wishes_0501_5000.json`:

```
Classmate Family Harmony response: "Fireworks every year, warm and fuzzy, 🏡❤️ Home is where the heart is."
Current generated count: 914
Classmate Good Results in Competition response: "Brilliant as the galaxy, bright future🌟, may you get what you wish!"
Current generated count: 915
Classmate Get Rich response: "Proud of success, money rolls to you🎉💰"
Current generated count: 916
Classmate Job Promotion response: "Light of promotion, illuminating the galaxy, future is bright as rosy clouds.🌟🌈"
Current generated count: 917
Classmate Recovery Blessing response: "Bid farewell to pain, be strong like a flower. 🌱✨May your future be bright and your body and mind be brilliant."
Current generated count: 918
```

💡Note, this is just a rough traversal of all roles and scenes, but **not all roles are suitable for all scenes** (many are inappropriate). To improve this, a heatmap should be made for mapping. If it is not suitable to produce this data, skip it directly; or do a match after getting the data. If it meets the inappropriate role + scene at the same time, remove the data QA pair.

#### Data Merge

Because our previous data is saved once after running a round (to prevent wasted efforts), you may have multiple jsons that need to be combined. Here is a script to merge all jsons in a folder and clean the json format into a format consistent with the training script:

```bash
import os
import json

def extract_and_merge_conversations(folder_path, output_file):
    all_conversations = []

    # Traverse the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)

            # Open and read JSON file
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Extract required fields
                for item in data:
                    for conversation in item['conversation']:
                        extracted = {
                            'system': conversation['system'],
                            'input': conversation['input'],
                            'output': conversation['output']
                        }
                        # Wrap each dialogue in a 'conversation' key and add to the list as an independent object
                        all_conversations.append({'conversation': [extracted]})

    # Write the merged dialogue data to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(all_conversations, file, ensure_ascii=False, indent=4)

# Usage example
folder_path = 'tianji_wishes_datasets'  # Folder path to scan
output_file = 'tianji-wishes-chinese-v0.1.json'     # Output file name and path
extract_and_merge_conversations(folder_path, output_file)
```

After merging, it is the fine-tuning dataset we need.

#### Secondary Cleaning

After getting the initial data, there may still be some strange things, such as wrong sentence length return, not an answer but a very short sentence `The current blessing is as follows`, adding modal particles easily leads to `! La~` `. Oh!` Such strange phenomena of punctuation appearing in front, so we need to use a cleaning script to filter the data to a certain extent. Since it is relatively long (and has not been elegantly iterated), the cleaning script is placed at [https://github.com/SocialAI-tianji/Tianji](https://github.com/SocialAI-tianji/Tianji), you can check it yourself.

Of course, if you have time and energy, the most important thing is to clean it manually. Of course, **the most important thing is to record relatively good information and make good output when manufacturing data, which can greatly save the time of secondary cleaning.**

#### Direct Download

For the convenience of everyone's use, the download address of the manufactured data is provided here. You can get it from huggingface yourself:

[https://huggingface.co/datasets/sanbu/tianji-wishes-chinese/blob/main/tianji-wishes-chinese-v0.1.json](https://huggingface.co/datasets/sanbu/tianji-wishes-chinese/blob/main/tianji-wishes-chinese-v0.1.json)

Mirror site download:

[https://hf-mirror.com/datasets/sanbu/tianji-wishes-chinese](https://hf-mirror.com/datasets/sanbu/tianji-wishes-chinese)

### Environment Preparation

Next, we prepare the fine-tuning environment. Due to time constraints (the standard process is the same), we will omit the detailed operations here. For detailed operations, please refer to the official xtuner tutorial [https://github.com/InternLM/Tutorial/tree/main/xtuner](https://github.com/InternLM/Tutorial/tree/main/xtuner), or the Xtuner Qlora part in the self-llm project: [https://github.com/datawhalechina/self-llm/blob/master/InternLM2/04-InternLM2-7B-chat%20Xtuner%20Qlora%20%E5%BE%AE%E8%B0%83.md](https://github.com/datawhalechina/self-llm/blob/master/InternLM2/04-InternLM2-7B-chat%20Xtuner%20Qlora%20%E5%BE%AE%E8%B0%83.md)

⚠ The following process is based on **python 3.10**, please pay attention to the version.

First create a virtual environment, then install the following dependencies:

```python
python -m pip install --upgrade pip
pip install modelscope==1.9.5
pip install transformers==4.36.2
pip install streamlit==1.39.0
pip install sentencepiece==0.1.99
pip install accelerate==0.24.1
pip install transformers_stream_generator==0.0.4
pip install einops ujson
pip install protobuf
```

Install xtuner:

```
git clone -b v0.1.18 https://github.com/InternLM/xtuner
cd xtuner && pip install -e '.[all]'

# Verify success
xtuner version
```

Model Download:

Execute the following python file in a suitable location:

```python

from modelscope import snapshot_download

model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b', cache_dir='./model_temp', revision='master')
```

After completing the above process, you can officially enter the fine-tuning stage.

### Formal Fine-tuning

In order to fine-tune our own dataset configuration, we need to modify the config of xtuner. First, check what configurations are available:

```python
xtuner list-cfg | grep internlm2

internlm2_7b_full_finetune_custom_dataset_e1
internlm2_7b_full_finetune_custom_dataset_e1_sequence_parallel_4
internlm2_7b_qlora_alpaca_e3
internlm2_7b_qlora_arxiv_gentitle_e3
internlm2_7b_qlora_code_alpaca_e3
internlm2_7b_qlora_colorist_e5
internlm2_7b_qlora_json_e3
internlm2_7b_qlora_lawyer_e3
internlm2_7b_qlora_msagent_react_e3_gpu8
internlm2_7b_qlora_oasst1_512_e3
internlm2_7b_qlora_oasst1_e3
internlm2_7b_qlora_sql_e3
```

```python
# Create a folder for fine-tuning work
mkdir /home/finetune
# Copy configuration file
cd /home/finetune && xtuner copy-cfg internlm2_chat_7b_qlora_oasst1_e3 ./
```

Next, we need to modify the configuration file. Simply put, you need to make these modifications:

```python
# Modify model to local path
- pretrained_model_name_or_path = 'internlm2/internlm2-chat-7b'
+ pretrained_model_name_or_path = '/home/model_temp/Shanghai_AI_Laboratory/internlm2-chat-7b'

# Modify training dataset to local path
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = '/home/merged_data.json'

# Modify Evaluate
-
evaluation_freq = 500
SYSTEM = ''
evaluation_inputs = [
    '请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai'
]
+
evaluation_freq = 50
SYSTEM = 'You are now a blessing master, help me send corresponding blessings for different people, things, and festivals'
evaluation_inputs = [
    'Happy birthday to sister', 'Wish sister a smooth negotiation','Happy Lantern Festival to everyone'
]

# Modify dataset loading
- dataset=dict(type=load_dataset, path=data_path),
+ dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path)),
```

The following is the modified result, you can copy it directly (just modify the model path and training set path, as well as the input of Evaluate, you can turn it into your own configuration file to start training.)
Or, you can also get all fine-tuning configurations using xtuner in the main repository of tianji `https://github.com/SocialAI-tianji/Tianji/tree/main/tianji/finetune/xtuner`.

```python
# Copyright (c) OpenMMLab. All rights reserved.
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
from xtuner.engine.hooks import (DatasetInfoHook, EvaluateChatHook,
                                 VarlenAttnArgsToMessageHubHook)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.utils import PROMPT_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = '/home/model_temp/Shanghai_AI_Laboratory/internlm2-chat-7b'
use_varlen_attn = False

# Data
data_path = '/home/tianji-wishes-test_0502.json'
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 2048
pack_to_max_length = True

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 16
dataloader_num_workers = 0
max_epochs = 3
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 50
save_total_limit = 10  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 50
SYSTEM = 'You are now a blessing master, help me send corresponding blessings for different people, things, and festivals'
evaluation_inputs = [
    'Happy birthday to sister','Happy birthday to sister, serious style','Happy birthday to sister, Little Red Book style', 'Wish sister a smooth negotiation, Little Red Book style','Happy Lantern Festival to everyone','Happy Spring Festival to the leader, serious style'
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
    use_varlen_attn=use_varlen_attn,
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
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn))

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
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

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

if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
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

# set log processor
log_processor = dict(by_epoch=False)

```

Next, start training directly with the new configuration (if your VRAM is insufficient, you can switch to `--deepspeed deepspeed_zero3`)

```bash
xtuner train ./internlm2_chat_7b_qlora_oasst1_e3_copy.py  --deepspeed deepspeed_zero2
```

After training, all weight files are placed in `work_dirs` under the training directory. The directory structure is roughly as follows:

```python
drwxr-xr-x 3 root root       4096 May  2 12:23 20240502_122337/
-rw-r--r-- 1 root root       6413 May  2 12:24 internlm2_chat_7b_qlora_oasst1_e3_copy.py
-rw-r--r-- 1 root root 1886589762 May  2 12:43 iter_500.pth
-rw-r--r-- 1 root root 1886601474 May  2 12:50 iter_657.pth
-rw-r--r-- 1 root root         76 May  2 12:50 last_checkpoint
```

It is worth noting that usually only one round of fine-tuning is needed here, because LLMs usually have a photographic memory (there are many related studies) and are prone to overfitting.

- If you want to see more changes brought by hyperparameters to the results, please refer to:

LoRA and QLoRA Fine-tuning Large Language Models: Insights after Hundreds of Experiments - OneFlow Article - Zhihu
[https://zhuanlan.zhihu.com/p/664912829](https://zhuanlan.zhihu.com/p/664912829)

- If you want to use full fine-tuning, for internlm2-7B please prepare at least 2xA100 80G graphics cards and use the following command to enable fine-tuning. (It is recommended that you use "ten thousand" level data before performing full fine-tuning. Currently, Tianji related data is not enough to support good full fine-tuning). It is recommended to mix in more normal dialogue data to ensure the effect of full fine-tuning is normal. The value of NPROC_PER_NODE indicates how many graphics cards are used for fine-tuning. At this time, the VRAM usage of each of the two cards is close to 79G.

```bash
NPROC_PER_NODE=2 xtuner train ./全量微调配置.py  --deepspeed deepspeed_zero3
```

### Effect Verification

First, we need to convert the format to HF, and then merge it with the original model. A unified script is provided here:

```bash
HF_OUTPUT_DIR="./hf" # Output address after lora is converted to hf format
MERGE_OUTPUT_DIR="./merge" # Output address after merging with the original model
SCRIPT_PATH="./internlm2_chat_7b_qlora_oasst1_e3_copy.py" # Training configuration file
SRC_MODEL_PATH="/home/model_temp/Shanghai_AI_Laboratory/internlm2-chat-7b" # Original model address
WEIGHTS_PATH="/home/finetune/work_dirs/internlm2_chat_7b_qlora_oasst1_e3_copy/iter_150.pth" # lora weight address

rm -rf $HF_OUTPUT_DIR
rm -rf $MERGE_OUTPUT_DIR
mkdir -p $HF_OUTPUT_DIR
mkdir -p $MERGE_OUTPUT_DIR

xtuner convert pth_to_hf "${SCRIPT_PATH}" "${WEIGHTS_PATH}" "${HF_OUTPUT_DIR}"
xtuner convert merge \
    "${SRC_MODEL_PATH}" \
    "${HF_OUTPUT_DIR}" \
    "${MERGE_OUTPUT_DIR}" \
    --max-shard-size "2GB"
```

If this step reports an error, please check if WEIGHTS_PATH is correct.

Of course, you can also not merge (you can upload weights after merging), but directly load lora after conversion. The corresponding script is as follows:

```bash
HF_OUTPUT_DIR="./hf" # Output address after lora is converted to hf format
SCRIPT_PATH="./internlm2_chat_7b_qlora_oasst1_e3_copy.py" # Training configuration file
SRC_MODEL_PATH="/home/model_temp/Shanghai_AI_Laboratory/internlm2-chat-7b"
WEIGHTS_PATH="/home/finetune/work_dirs/internlm2_chat_7b_qlora_oasst1_e3_copy/iter_150.pth"

rm -rf $HF_OUTPUT_DIR
rm -rf $MERGE_OUTPUT_DIR
mkdir -p $HF_OUTPUT_DIR

xtuner convert pth_to_hf "${SCRIPT_PATH}" "${WEIGHTS_PATH}" "${HF_OUTPUT_DIR}"

xtuner chat "${SRC_MODEL_PATH}" --adapter "${HF_OUTPUT_DIR}" --prompt-template internlm2_chat --system "You are now a blessing master, help me send corresponding blessings for different people, things, and festivals" --temperature 0.7
```

Start conversation:

```python
# If you want more diversity add --temperature 1
xtuner chat ./merge --prompt-template internlm2_chat --system "You are now a blessing master, help me send corresponding blessings for different people, things, and festivals" --temperature 0.7
```

At this time, you will see the following display. You only need to input the previous prompt and trigger style appropriately to verify. What is inconvenient here is that in order to remove the influence of context, we need to repeatedly input RESET to reset.

```bash
double enter to end input (EXIT: exit chat, RESET: reset history) >>> Wish younger brother a job promotion, Little Red Book style

Brother, your talent is like flowers in spring, blooming with brilliant brilliance. Work is like a fish in water, the road to promotion is smooth, may you continue to shine and become a star in the workplace! Come on! 🌟🌈<|im_end|>

double enter to end input (EXIT: exit chat, RESET: reset history) >>> RESET

Log: History responses have been removed!

double enter to end input (EXIT: exit chat, RESET: reset history) >>> I want to send Mid-Autumn Festival blessings to my brother, Little Red Book style

🌕🎉 Brother, the moon is round and people are reunited during the Mid-Autumn Festival. May your career be as round as the moon, your family happy and reunited, full of joy and happiness! 🌟<|im_end|>

double enter to end input (EXIT: exit chat, RESET: reset history) >>> RESET

Log: History responses have been removed!

double enter to end input (EXIT: exit chat, RESET: reset history) >>> I want to send birthday wishes to Teacher Zhao, serious style

Respected Teacher Zhao, on this auspicious occasion, I wish you longevity, health, and a prosperous career. May you be like the flowers of spring in the years to come, blooming with even more brilliant brilliance, setting a lofty example for the students. May your life be as beautiful as a poem, and every day be filled with wisdom and joy. May your work be as pleasant as a movement, and every note carries the love and persistence for the cause of education. Here, I sincerely wish you a happy birthday and all the best! <|im_end|>
```

### Web Demo

For the convenience of testing, we can also use the streamlit demo. The usage method is as follows:

`pip install streamlit==1.39.0`

```
# Find a place to clone the repository
git clone https://github.com/InternLM/InternLM.git
```

Change the model path in lines 29 and 33 of `/root/code/InternLM/web_demo.py` to the path where the parameters are stored after Merge `/home/finetune/merge`

Run the `web_demo.py` file in the `/root/personal_assistant/code/InternLM` directory according to the following command.

```python
streamlit run /root/personal_assistant/code/InternLM/web_demo.py --server.address 127.0.0.1 --server.port 6006
```

Through the above steps, you can successfully fine-tune the initial version of the Tianji blessing module and successfully run inference.

However, since there is still room for improvement in the quality of manufacturing data, answer randomness injection, and input randomness enhancement, there are still many ways to achieve it. Sending blessings can become better, more intelligent, and more universal, thereby truly getting closer to the usage scenarios in Chinese, making the llm dialogue more "human".

## Thanks to Every Contributor

Due to the large number of contributors, we cannot list them all. We sincerely thank every partner who has contributed to the Tianji project. It is they who make this project better. We also look forward to your joining. Believe me, you can also become light!

https://github.com/SocialAI-tianji/Tianji/blob/main/docs/contributor.md
