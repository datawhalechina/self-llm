# 天机简介

想从零学会大语言模型应用？想直接复制粘贴代码？天机满足你的一切需求；甚至，你还能学到一点意想不到的东西。
中国文化注重人情世故，涉及到复杂的社交规则、礼仪和人际关系。Social 团队通过深入理解中文语境、文化特点和丰富的语料，提供了基于各类中国式文化的场景下的应对方法、对话案例与情景模拟。
目前天机已经支持的应用场景包括：如何送**祝福、敬酒、请客、送礼、人际交流、化解尴尬以及应对矛盾**等等，还有更多社区小伙伴们建议的 LLM 提高情商方案正在制作中~
除了把这个项目当成一个好玩的社交大模型外，你还可以把它当作完整的全栈大语言模型应用入门学习仓库! 你可以在里面学到**提示词工程 、智能体制作、 模型微调、RAG 数据清洗与使用，以及代码规范**...等等你所需要的大语言模型应用知识，你可以在通过学习该项目后,  快速改造出属于自己的新的大语言模型项目，我们期待你成为下一个大模型应用开发高手！

> *Tianji 开源链接：*[*Link*](https://github.com/SocialAI-tianji/Tianji)
> *Tianji 官方网站：*[*Link*](https://socialai-tianji.github.io/socialai-web/)
> *Tianji 知识库版应用地址：*[*Link*](http://120.76.130.14:6006/knowledges/)
> *Tianji prompt版应用地址：*[*Link*](http://120.76.130.14:6006/prompt/)
> *Tianji 视频介绍：*[*Link*](https://www.bilibili.com/video/BV1cvbyefEfp)

由于内容众多，时间关系，本篇教程只会带你简单看看如何快速运行天机的“送祝福”微调模型，更多有趣的内容，等你来天机项目和文档亲自探索！

## 实现送祝福微调模型

接下来，让我们快速实现一个属于自己的送祝福微调模型。我们将基于 Xtuner Qlora，详细讲解如何对 internLM2 进行天机送祝福模块的微调，全流程包括数据制造、推理微调等所有环节。

在本次微调示范中，我们选用的是 internlm2-chat-7b 模型，你需要准备一台24G显存的机器用于微调（3090即可）。

微调一个模型的第一步是准备高质量的训练数据。对于一个送祝福模型，你需要收集各种祝福语的数据，数据来源可以是公开的祝福语数据集、社交媒体、电子书籍或者任何包含丰富祝福语的文本。

在准备完成数据来源以及获取到对应数据后，你需要使用该数据文本进行数据制造（比如下面演示的few shot，但这只是最小的例子，真正意义的数据制造你需要用一个数据“知识”切块去生成对应的QA对，这是才是我们最后期望得到的数据。

所以，理论上最好的数据是利用这些现有知识，通过更聪明的大模型基于这些知识得到高精度的回复QA对数据，也有的人是通过大模型抽取小说文本对话的方式来实现格式抽取，但总之你需要的是一个最好无限火力的聪明大模型来帮助你进行文本数据清洗。

当你成功打通微调后，你会发现**真正复杂的工作都是在清洗数据、处理、生成数据、归类数据**上，这些才是影响最后效果的**最大难点问题**。

这里推荐你使用本地的llm去进行数据清洗（除非你财大气粗），否则api key很容易分分钟用完，你可以通过部署本地 llama3-chinese 或者 qwen 进行数据制造工作。

接下来我们来看看如何进行数据制造：

### 数据处理

#### 数据制造

在清洗数据前，请确保你已经安装对应SDK如zhipuai以及openai SDK，安装后直接运行即可。

```python
from zhipuai import ZhipuAI
import time
import json
import random
import datetime

# zhipuai
# 此处填写您自己的APIKey
# zhipu_api_key = ""
# client = ZhipuAI(api_key=zhipu_api_key)
# def get_data_zhipu(content):
#     response = client.chat.completions.create(
#         model="glm-4",  # 填写需要调用的模型名称
#         messages=[
#             {"role": "system", "content": "你现在是一个精通言语表达、热爱他人、尊重长辈、富有文采的送祝福大师，请你编辑一条文本，表示对应场景的祝福语"},
#             {"role": "user",
#              "content": content,
#              "temperature": 1} # 多样化输出
#         ],
#     )
#     res = response.choices[0].message.content
#     return res

# deepseek
from openai import OpenAI
deepseek_key = ""  #此处填写deepseek的key
client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com/v1")
def get_data_ds(content):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你现在是一个精通言语表达、热爱他人、尊重长辈、富有文采的送祝福大师，请你编辑一条文本，表示对应场景的祝福语"},
            {"role": "user",
             "content": content,
             "temperature": 1} # 多样化输出
        ]
    )
    res = response.choices[0].message.content
    return res

# 可利用大模型补充不同对象  当前28种
name_list = ['赵老师', '大舅', '大伯', '李总', '邻居赵大妈', '母亲', '姐姐', '妹妹', '哥哥', '弟弟', '爷爷', '奶奶', '外公',
        '外婆', '伯母', '叔叔', '阿姨', '堂兄', '堂妹', '表哥', '表妹', '导师', '同学', '同事', '领导',
        '邻居', '老板', '医生', ]

# 可利用大模型补充对应场景 当前18种
scenes = ['生日', '春节', '元宵节', '端午节', '七夕节', '中秋节',
            '重阳节', '除夕', '腊八节','谈判顺利','乔迁新居', '周年纪念' ,'新婚快乐' ,'家庭和睦', '比赛取得好成绩' ,'发财','工作升职 ','康复', ]

# 可利用大模型补充不同风格，加入更多 fewshot 造出更好的数据
styles = {
    "小红书":
    {
        "style_temple":"小红书风格，每条加入1-2个emoji表情包来增加趣味性。\n### 注意，你要参考下列句子的艺术风格进行祝福语撰写（注意！只看造句风格），祝福语结尾都带上语气助词词，参考句子为：{} ###",
        "if_example":True,
        "examples":
        [
    '默念你的名,祝你前途云蒸霞蔚，灿若星河。愿你度过的叫吉时，得到的叫如愿！',
    '希望你岁末将至，敬颂冬绥，平安喜乐，万事胜意。',
    '希望你不用奔赴大海，也能看到春暖花开；不用颠沛流离，也能遇到一生所伴！',
    '祝我们好在春夏秋冬,祝你阔谈，祝你烂漫，祝你和自己相约在风里，此后只剩欢愉。',
    '希望你可以明确地爱，直接的厌恶，真诚的喜欢，站在太阳下的坦荡，大声无愧地称赞自己，学会爱自己！',
    '前方荣光万丈，身后温暖一方，凡是过往，皆为序章。',
    '愿所念之人 平安喜乐。愿所想之事 顺心如意！',
        ]
    },
    "正常":
    {
        "style_temple":"正常风格，有礼貌即可",
        "if_example":False,
        "examples":[]
    },
    "严肃":
    {
        "style_temple":"商业严肃风格，要求用在职场或长辈祝福上，显得有礼貌、干练,句子可以长一些",
        "if_example":False,
        "examples":[]
    }
}

random_finalprompt_sentence = [
    '', #默认情况
    '回答中可以不出现对象称谓和场景信息，也不用出现“愿你”“祝你”（对自己的长辈需要出现对象称谓和祝你），',
    '回答中可以不出现对象称谓和场景信息，',
    '回答中不用出现“愿你”“祝你”',
]
final_prompt = """
该祝福语字数小于 {} 字。 \n
请根据对象称谓及场景，写出符合对象的身份和场景气氛的祝福文案。要求的风格是：{} \n，注意不要有标题混在其中，对象称谓是：{}，祝福场景是：{}。 \n
{} 根据不同对象用不同的语气（尊敬、诙谐搞笑、亲近），请直接返回祝福文本，不要说任何其他话：
"""

if __name__ == "__main__":
    ##### 此处配置 #####
    roop_count = 2
    now_count = 0
    stylename = "小红书" # 小红书、正常、严肃
    output_number_limit = 50 # 限制回答输出长度，严肃的100，普通的小于20
    ##### 此处配置 #####

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
                    print("当前生成数目：", now_count)
                    if stylename == '正常':
                        # 默认不加风格指定
                        _input_prompt = f"祝{name}{scene}"
                    else:
                        _input_prompt = f"祝{name}{scene},{stylename}风格"
                    print("input:",_input_prompt)

                    conversation = {
                        "conversation": [
                            {
                                "system": "你现在是一个送祝福大师，帮我针对不同人和事情、节日送对应的祝福",
                                "src_input":input_prompt,
                                "style_name":stylename,
                                "input": _input_prompt,
                                "output": str(response).replace('\"','')
                            }
                        ]
                    }

                    # 将对话加入到列表中
                    conversations.append(conversation)
                except Exception as e:
                    print(e)
                    continue

        now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        file_path = f"./wishes_{stylename}_{now_time}.json"
        with open(file_path, "w", encoding='utf8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=4)

```

**注意**，这里为什么需要把 input 替换成 `f"祝{name}{scene}"` 的格式——是因为这里的input需要尽可能模拟人的输入，而不能是制造数据时候的输入。此外，我们在此设定了三种风格：正常、小红书、严肃；期待当加上风格名触发后可以有预期的输出。

这只是最简单的示例，如何根据祝福语的长短不同而生成后的语法风格要对应变化、如何更接近真人的风格化；这些都需要高质量的数据 + 良好的数据制造方式才可获得。

举个例子，如果在这里我们想要控制祝福语长短和语法风格，首先前者我们就要在之前的制造条件中（比如之前的对象、场景是一个条件）加入新的祝福语长短控制条件（比如我这里的小红书和正常风），而且此时few shot参考的句子也应当有所不同，这样才能保证我们造数据的llm可以返回预期长度的结果。如果是控制语法风格，我们就需要爬取大量文艺书籍、小红书等真正人类写的文案进行清洗，然后利用这些作为few shot得到严格的返回，需要用严格的提示词让模型写出类似的语句或者单独微调一个属于该文艺范畴的模型版本用于制造对应数据（有时候few shot指令跟随不那么有用）。

💡代码中有 `random_xxxxxxx_sentence` 的部分表明这是一个随机性注入列表，我们可以维护一些语句用于提高随机性（比如附加条件的修改），让大模型返回的结果更具特色。

若成功运行，你将看到类似如下输出结果，等待片刻后将得到属于本地的json文件 `wishes_0501_5000.json` ：

```
同学 家庭和睦 response: "烟火年年，暖意洋洋，🏡❤️家是心之所向。"
当前生成数目： 914
同学 比赛取得好成绩 response: "灿若星河，前程似锦🌟，所得皆所愿！"
当前生成数目： 915
同学 发财 response: "春风得意马蹄疾，财源滚滚至君前🎉💰"
当前生成数目： 916
同学 工作升职 response: "升职之光，照亮星河，未来灿烂如霞。🌟🌈"
当前生成数目： 917
同学 康复祝福 response: "挥别病痛，如花开坚强。🌱✨愿你前程，云蒸霞蔚，身心俱灿。"
当前生成数目： 918
```

💡注意，这里只是粗暴的进行所有角色和场景的遍历，但**并非所有角色都适配所有场景**（很多是不合适的），这里为了改进应该做一个heatmap进行映射，若不合适生产该数据，将直接跳过；又或者是在得到数据后做一个匹配，如果同时满足不合适的角色+场景就去除该数据QA对。

#### 数据合并

因为我们之前的数据都是跑完一轮存一次（以防前功尽弃），所以可能你有多个json需要组合，这里提供了一个脚本合并一个文件夹中的所有json，并把json格式清洗成和训练脚本一致适配的格式：

```bash
import os
import json

def extract_and_merge_conversations(folder_path, output_file):
    all_conversations = []

    # 遍历指定文件夹
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)

            # 打开并读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # 提取需要的字段
                for item in data:
                    for conversation in item['conversation']:
                        extracted = {
                            'system': conversation['system'],
                            'input': conversation['input'],
                            'output': conversation['output']
                        }
                        # 将每个对话包装在一个 'conversation' 键中，并作为独立对象加入列表
                        all_conversations.append({'conversation': [extracted]})

    # 将合并后的所有对话数据写入一个新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(all_conversations, file, ensure_ascii=False, indent=4)

# 使用示例
folder_path = 'tianji_wishes_datasets'  # 要扫描的文件夹路径
output_file = 'tianji-wishes-chinese-v0.1.json'     # 输出文件的名称和路径
extract_and_merge_conversations(folder_path, output_file)
```

合并后就是我们所需要的微调数据集。

#### 二次清洗

得到最初数据后，可能还存在一些奇怪的东西，比如句子长度返回错误，不是回答而是很短的一句 `当前祝福语如下` ，加入语气助词后容易出现 `！啦~` `。哦！` 如此在前面出现标点的奇怪现象，故我们需要利用清洗脚本对数据进行一定的筛选，由于比较冗长（还没有优雅的迭代），清洗脚本放在 [https://github.com/SocialAI-tianji/Tianji](https://github.com/SocialAI-tianji/Tianji)，可以自行查看

当然，如果你有时间和精力，最重要的还是可以用人工进行清洗。当然，**最重要的还是最好在数据制造的时候就记录比较完好的信息和做出比较好的output，能大大节约二次清洗的时间。**

#### 直接下载

为了方便大家使用，这里提供了已制造数据的下载地址，大家可以自行从huggingface上获取：

[https://huggingface.co/datasets/sanbu/tianji-wishes-chinese/blob/main/tianji-wishes-chinese-v0.1.json](https://huggingface.co/datasets/sanbu/tianji-wishes-chinese/blob/main/tianji-wishes-chinese-v0.1.json)

镜像站下载：

[https://hf-mirror.com/datasets/sanbu/tianji-wishes-chinese](https://hf-mirror.com/datasets/sanbu/tianji-wishes-chinese)

### 环境准备

接下来我们准备微调的环境，由于时间关系（标准流程是一样的），这里只做省略快速操作，详细操作请参考 xtuner的官方教程 [https://github.com/InternLM/Tutorial/tree/main/xtuner](https://github.com/InternLM/Tutorial/tree/main/xtuner)，或者是self-llm 项目中关于Xtuner Qlora的部分：[https://github.com/datawhalechina/self-llm/blob/master/InternLM2/04-InternLM2-7B-chat Xtuner Qlora 微调.md](https://github.com/datawhalechina/self-llm/blob/master/InternLM2/04-InternLM2-7B-chat%20Xtuner%20Qlora%20%E5%BE%AE%E8%B0%83.md)

⚠ 以下基于 **python 3.10** 构建全过程，请注意版本

首先创建一个虚拟环境，随后安装如下依赖

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

安装xtuner

```
git clone -b v0.1.18 https://github.com/InternLM/xtuner
cd xtuner && pip install -e '.[all]'

# 验证成功
xtuner version
```

模型下载

找地方执行下列python文件

```python

from modelscope import snapshot_download

model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b', cache_dir='./model_temp', revision='master')
```

完成上述过程后就可以正式进入微调阶段。

### 正式微调

为了微调自己的数据集配置，我们需要修改xtuner的config，首先查看有哪些配置：

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
# 新建一个用于微调工作的文件夹
mkdir /home/finetune
# 复制配置文件
cd /home/finetune && xtuner copy-cfg internlm2_chat_7b_qlora_oasst1_e3 ./
```

接下来我们需要修改配置文件，简单来说你要做这几处修改：

```python
# 修改模型为本地路径
- pretrained_model_name_or_path = 'internlm2/internlm2-chat-7b'
+ pretrained_model_name_or_path = '/home/model_temp/Shanghai_AI_Laboratory/internlm2-chat-7b'

# 修改训练数据集为本地路径
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = '/home/merged_data.json'

# 修改Evaluate
-
evaluation_freq = 500
SYSTEM = ''
evaluation_inputs = [
    '请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai'
]
+
evaluation_freq = 50
SYSTEM = '你现在是一个送祝福大师，帮我针对不同人和事情、节日送对应的祝福'
evaluation_inputs = [
    '祝姐姐生日快乐', '祝妹妹谈判顺利','祝大家元宵节快乐'
]

# 修改数据集加载
- dataset=dict(type=load_dataset, path=data_path),
+ dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path)),
```

以下是修改后的结果，你可以直接复制（只要修改模型路径和训练集路径，以及Evaluate的input，就可以把他变为你自己的配置文件开始训练。）
或者，你也可以在 tianji 的主仓库中获得所有使用 xtuner 的微调配置 `https://github.com/SocialAI-tianji/Tianji/tree/main/tianji/finetune/xtuner`。

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
SYSTEM = '你现在是一个送祝福大师，帮我针对不同人和事情、节日送对应的祝福'
evaluation_inputs = [
    '祝姐姐生日快乐','祝姐姐生日快乐，严肃风格','祝姐姐生日快乐,小红书风格', '祝妹妹谈判顺利，小红书风格','祝大家元宵节快乐','祝领导春节快乐，严肃风格'
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

接下来对新的配置直接开始训练(如果你显存不够，可以切换成 `--deepspeed deepspeed_zero3` )

```bash
xtuner train ./internlm2_chat_7b_qlora_oasst1_e3_copy.py  --deepspeed deepspeed_zero2
```

训练结束后，所有权重文件放置在训练目录下的work_dirs中，目录大致为：

```python
drwxr-xr-x 3 root root       4096 May  2 12:23 20240502_122337/
-rw-r--r-- 1 root root       6413 May  2 12:24 internlm2_chat_7b_qlora_oasst1_e3_copy.py
-rw-r--r-- 1 root root 1886589762 May  2 12:43 iter_500.pth
-rw-r--r-- 1 root root 1886601474 May  2 12:50 iter_657.pth
-rw-r--r-- 1 root root         76 May  2 12:50 last_checkpoint
```

值得注意的是，这里通常只需要微调一轮就好，原因是llm通常是过目不忘（有很多相关研究）容易过拟合。

- 如果你想查看更多超参数对结果带来的变动，请参考：

LoRA和QLoRA微调语言大模型：数百次实验后的见解 - OneFlow的文章 - 知乎
[https://zhuanlan.zhihu.com/p/664912829](https://zhuanlan.zhihu.com/p/664912829)

- 如果你想使用全量微调，对于 internlm2-7B 请至少准备 2xA100 80G 的显卡使用下列命令启用微调。（建议你使用“万”级别的数据再进行全量微调，目前天机相关数据还不足以支持好的全量微调）推荐混入更多正常对话数据来确保全量微调效果正常。NPROC_PER_NODE 的值表示使用几张显卡进行微调,此时双卡每张卡显存占用接近 79G。

```bash
NPROC_PER_NODE=2 xtuner train ./全量微调配置.py  --deepspeed deepspeed_zero3
```

### 效果验证

首先我们需要转换格式为hf，再与原模型合并，这里提供了统一脚本：

```bash
HF_OUTPUT_DIR="./hf" # lora转为hf格式后的输出地址
MERGE_OUTPUT_DIR="./merge" # 与原模型合并后的输出地址
SCRIPT_PATH="./internlm2_chat_7b_qlora_oasst1_e3_copy.py" # 训练配置文件
SRC_MODEL_PATH="/home/model_temp/Shanghai_AI_Laboratory/internlm2-chat-7b" # 原模型地址
WEIGHTS_PATH="/home/finetune/work_dirs/internlm2_chat_7b_qlora_oasst1_e3_copy/iter_150.pth" # lora权重地址

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

如果这步报错，请检查 WEIGHTS_PATH 是否正确。

当然，你也可以不合并（合并后就可以上传权重），而是转换后直接加载lora，对应脚本如下：

```bash
HF_OUTPUT_DIR="./hf" # lora转为hf格式后的输出地址
SCRIPT_PATH="./internlm2_chat_7b_qlora_oasst1_e3_copy.py" # 训练配置文件
SRC_MODEL_PATH="/home/model_temp/Shanghai_AI_Laboratory/internlm2-chat-7b"
WEIGHTS_PATH="/home/finetune/work_dirs/internlm2_chat_7b_qlora_oasst1_e3_copy/iter_150.pth"

rm -rf $HF_OUTPUT_DIR
rm -rf $MERGE_OUTPUT_DIR
mkdir -p $HF_OUTPUT_DIR

xtuner convert pth_to_hf "${SCRIPT_PATH}" "${WEIGHTS_PATH}" "${HF_OUTPUT_DIR}"

xtuner chat "${SRC_MODEL_PATH}" --adapter "${HF_OUTPUT_DIR}" --prompt-template internlm2_chat --system "你现在是一个送祝福大师，帮我针对不同人和事情、节日送对应的祝福" --temperature 0.7
```

启动对话：

```python
# 如果想要多样性更强加上 --temperature 1
xtuner chat ./merge --prompt-template internlm2_chat --system "你现在是一个送祝福大师，帮我针对不同人和事情、节日送对应的祝福" --temperature 0.7
```

此时你会看到如下显示，你只需要恰当的输入之前的prompt和触发风格即可验证。这里比较不方便的是为了去除上下文影响，我们需要反复输入RESET进行重置。

```bash
double enter to end input (EXIT: exit chat, RESET: reset history) >>> 祝弟弟工作升职 ,小红书风格

弟弟呀，你的才华如同春日里的花朵，绽放出绚烂的光彩。工作如鱼得水，升职之路顺顺利利，愿你继续闪耀，成为职场的明星！加油哦！🌟🌈<|im_end|>

double enter to end input (EXIT: exit chat, RESET: reset history) >>> RESET

Log: History responses have been removed!

double enter to end input (EXIT: exit chat, RESET: reset history) >>> 我想送哥哥中秋节祝福,小红书风格

🌕🎉 哥，中秋月圆人团圆，愿你事业如月圆，家庭幸福似团圆，快乐满怀，幸福满满！🌟<|im_end|>

double enter to end input (EXIT: exit chat, RESET: reset history) >>> RESET

Log: History responses have been removed!

double enter to end input (EXIT: exit chat, RESET: reset history) >>> 我想送赵老师生日祝福,严肃风格

尊敬的赵老师，值此佳辰，恭祝您福寿安康，事业蒸蒸日上。愿您在未来的岁月里，如同春日之花，绽放出更加绚烂的光彩，为莘莘学子树立起崇高的榜样。愿您的生活如同诗篇般美好，每一天都充满着智慧与喜悦。愿您的工作如同乐章般动听，每一个音符都承载着对教育事业的热爱与执着。在此，衷心祝愿您生日快乐，万事胜意！<|im_end|>
```

### 网页demo

为了方便测试，我们还可以使用streamlit的demo，使用方法如下：

`pip install streamlit==1.39.0`

```
# 找地方clone仓库
git clone https://github.com/InternLM/InternLM.git
```

将 `/root/code/InternLM/web_demo.py` 中 29 行和 33 行的模型路径更换为Merge后存放参数的路径 `/home/finetune/merge`

按照如下命令运行 `/root/personal_assistant/code/InternLM` 目录下的 `web_demo.py` 文件即可。

```python
streamlit run /root/personal_assistant/code/InternLM/web_demo.py --server.address 127.0.0.1 --server.port 6006
```

通过以上步骤，你就可以成功地微调出最初版本的天机送祝福模块并成功运行推理。

但由于制造数据的质量仍有改善空间、回答随机性注入、输入input的随机性加强仍有很多方式可以实现，送祝福完全可以变得更好、更智能、更通用，从而真正更接近中文下的使用场景，让llm对话更富有“人情味”。

## 感谢每一位贡献者

由于贡献者人数众多无法一一列举，我们由衷的感谢每一位为天机项目做出贡献的小伙伴，是他们让这个项目变得更好。我们也期待你的加入。相信我，你也可以变成光！

https://github.com/SocialAI-tianji/Tianji/blob/main/docs/contributor.md
