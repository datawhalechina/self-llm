# Chat-嬛嬛 是如何炼成的

**Chat-甄嬛**是利用《甄嬛传》剧本中所有关于甄嬛的台词和语句，基于大模型进行**LoRA微调**得到的模仿甄嬛语气的聊天语言模型。

> 甄嬛，小说《后宫·甄嬛传》和电视剧《甄嬛传》中的女一号，核心女主角。原名甄玉嬛，嫌玉字俗气而改名甄嬛，为汉人甄远道之女，后被雍正赐姓钮祜禄氏，抬旗为满洲上三旗，获名“钮祜禄·甄嬛”。同沈眉庄、安陵容参加选秀，因容貌酷似纯元皇后而被选中。入宫后面对华妃的步步紧逼，沈眉庄被冤、安陵容变心，从偏安一隅的青涩少女变成了能引起血雨腥风的宫斗老手。雍正发现年氏一族的野心后令其父甄远道剪除，甄嬛也于后宫中用她的连环巧计帮皇帝解决政敌，故而深得雍正爱待。几经周折，终于斗垮了嚣张跋扈的华妃。甄嬛封妃时遭皇后宜修暗算，被皇上嫌弃，生下女儿胧月后心灰意冷，自请出宫为尼。然得果郡王爱慕，二人相爱，得知果郡王死讯后立刻设计与雍正再遇，风光回宫。此后甄父冤案平反、甄氏复起，她也生下双生子，在滴血验亲等各种阴谋中躲过宜修的暗害，最后以牺牲自己亲生胎儿的方式扳倒了幕后黑手的皇后。但雍正又逼甄嬛毒杀允礼，以测试甄嬛真心，并让已经生产过孩子的甄嬛去准格尔和亲。甄嬛遂视皇帝为最该毁灭的对象，大结局道尽“人类的一切争斗，皆因统治者的不公不义而起”，并毒杀雍正。四阿哥弘历登基为乾隆，甄嬛被尊为圣母皇太后，权倾朝野，在如懿传中安度晚年。

Chat-甄嬛，实现了以《甄嬛传》为切入点，打造一套基于小说、剧本的**个性化 AI** 微调大模型完整流程，通过提供任一小说、剧本，指定人物角色，运行本项目完整流程，让每一位用户都基于心仪的小说、剧本打造一个属于自己的、契合角色人设、具备高度智能的个性化 AI。

> *Chat-嬛嬛模型累计下载量 15.6k，Modelscope 地址：*[*Link*](https://www.modelscope.cn/models/kmno4zx/huanhuan-chat-internlm2)   
> *Chat-嬛嬛累计获得 500 star，huahuan-chat 项目地址：*[*Link*](https://github.com/KMnO4-zx/huanhuan-chat.git)，xlab-huanhuan-chat 项目地址：[*Link*](https://github.com/KMnO4-zx/xlab-huanhuan.git)  


***OK，那接下来我将会带领大家亲自动手，一步步实现 Chat-甄嬛 的训练过程，让我们一起来体验一下吧~***

## Step 1: 环境准备

本文基础环境如下：

```
----------------
ubuntu 22.04
python 3.12
cuda 12.1
pytorch 2.3.0
----------------
```
> 本文默认学习者已安装好以上 Pytorch(cuda) 环境，如未安装请自行安装。

首先 `pip` 换源加速下载并安装依赖包

```shell
# 升级pip
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.16.1
pip install transformers==4.43.1
pip install accelerate==0.32.1
pip install peft==0.11.1
pip install datasets==2.20.0
```

## Step 2: 数据准备

首先，我们需要准备《甄嬛传》剧本数据，这里我们使用了《甄嬛传》剧本数据，我们可以查看一下原始数据的格式。

```text
第2幕
（退朝，百官散去）
官员甲：咱们皇上可真是器重年将军和隆科多大人。
官员乙：隆科多大人，恭喜恭喜啊！您可是国家的大功臣啊！
官员丙：年大将军，皇上对你可是垂青有加呀！
官员丁：年大人，您可是皇上的股肱之臣哪！
苏培盛（追上年羹尧）：年大将军请留步。大将军——
年羹尧：苏公公，有何指教？
苏培盛：不敢。皇上惦记大将军您的臂伤，特让奴才将这秘制的金创药膏交给大人，叫您使用。
年羹尧（遥向金銮殿拱手）：臣年羹尧恭谢皇上圣恩！敢问苏公公，小妹今日在宫中可好啊？
苏培盛：华妃娘娘凤仪万千、宠冠六宫啊，大将军您放心好了。
年羹尧：那就有劳苏公公了。（转身离去）
苏培盛：应该的。
```

每一句都有人物及对应的台词，所以就可以很简单的将这些数据处理成对话的形式，如下：

```
[
    {"rloe":"官员甲", "content":"咱们皇上可真是器重年将军和隆科多大人。"},
    {"rloe":"官员乙", "content":"隆科多大人，恭喜恭喜啊！您可是国家的大功臣啊！"},
    {"rloe":"官员丙", "content":"年大将军，皇上对你可是垂青有加呀！"},
    {"rloe":"官员丁", "content":"年大人，您可是皇上的股肱之臣哪！"},
    {"rloe":"苏培盛", "content":"年大将军请留步。大将军——"},
    ...
]
```

然后再将我们关注的角色的对话提取出来，形成 QA 问答对。对于这样的数据，我们可以使用正则表达式或者其他方法进行快速的提取，并抽取出我们关注的角色的对话。

然后很多情况下，我们并没有这样优秀的台词格式数据。所以我们可能就需要从一大段文本中抽取角色的对话数据，然后将其转换成我们需要的格式。

比如《西游记白话文》，我们可以看到他的文本是这样的。对于这样的文本，那我们就需要借助大模型的能力，从文本中提取出角色和角色对应的对话。然后再筛选出我们需要的角色对话。

> 可以借助一个小工具：[*extract-dialogue*](https://github.com/KMnO4-zx/extract-dialogue.git) 从文本中提取对话。
    
```
......
原来孙悟空走了以后，有一个混世魔王独占了水帘洞，并且抢走了许多猴子猴孙。孙悟空听到这些以后，气得咬牙跺脚。他问清了混世魔王的住处，决定找混世魔王报仇，便驾着筋斗云，朝北方飞去。

不一会儿，孙悟空就来到混世魔王的水脏洞前，对门前的小妖喊到∶“你家那个狗屁魔王，多次欺负我们猴子。我今天来，要和那魔王比比高低！

”小妖跑进洞里，报告魔王。魔王急忙穿上铁甲，提着大刀，在小妖们的簇拥下走出洞门。

孙悟空赤手空拳，夺过了混世魔王的大刀，把他劈成了两半。然后，拔下一把毫毛咬碎喷了出去，毫毛变成许多小猴子，直杀进洞里，把所有的妖精全杀死，然后救出被抢走的小猴子，放了一把火烧了水脏洞。
......
```

> chat-甄嬛 的原始数据：[*甄嬛传*](https://github.com/KMnO4-zx/huanhuan-chat/tree/master/dataset/input/huanhuan)  
> 西游记白话文原始数据：[*西游记*](https://github.com/KMnO4-zx/huanhuan-chat/blob/master/dataset/input/wukong/%E8%A5%BF%E6%B8%B8%E8%AE%B0%E7%99%BD%E8%AF%9D%E6%96%87.txt)

最后再将其整理成 `json` 格式的数据，如下：

```
[
    {
        "instruction": "小姐，别的秀女都在求中选，唯有咱们小姐想被撂牌子，菩萨一定记得真真儿的——",
        "input": "",
        "output": "嘘——都说许愿说破是不灵的。"
    },
    {
        "instruction": "这个温太医啊，也是古怪，谁不知太医不得皇命不能为皇族以外的人请脉诊病，他倒好，十天半月便往咱们府里跑。",
        "input": "",
        "output": "你们俩话太多了，我该和温太医要一剂药，好好治治你们。"
    },
    {
        "instruction": "嬛妹妹，刚刚我去府上请脉，听甄伯母说你来这里进香了。",
        "input": "",
        "output": "出来走走，也是散心。"
    }
]
```

> Chat-嬛嬛 的数据：[*chat-甄嬛*](https://github.com/datawhalechina/self-llm/blob/master/dataset/huanhuan.json)

所以，在这一步处理数据的大致思路就是：

***1. 从原始数据中提取出角色和对话 &emsp;2. 筛选出我们关注的角色的对话 &emsp;3. 将对话转换成我们需要的格式***

> *这一步也可以增加数据增强的环节，比如利用两到三条数据作为 example 丢给LLM，让其生成风格类似的数据。再或者也可以找一部分日常对话的数据集，使用 RAG 生成一些固定角色风格的对话数据。这里大家可以完全放开的大胆去尝试！*

## Step 3: 模型训练

那这一步，大家可能再熟悉不过了。在`self-llm`的每一个模型中，都会有一个 `Lora` 微调模块，我们只需要将数据处理成我们需要的格式，然后再调用我们的训练脚本即可。

此处选择我们选择 `LLaMA3_1-8B-Instruct` 模型进行微调，首先还是要下载模型，创建一个`model_download.py`文件，输入以下内容：

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('LLM-Research/Meta-Llama-3.1-8B-Instruct', cache_dir='/root/autodl-tmp', revision='master')
```

> 注意：记得修改 `cache_dir` 为你的模型下载路径哦~

其次，准备训练代码。对于熟悉 `self-llm` 的同学来说，这一步可能再简单不过了，在此处我会在当前目录下放置`train.py`，大家修改其中的数据集路径和模型路径即可。

> *当然也可以使用 `self-llm` 中的 `lora` 微调教程。教程地址：[Link](https://github.com/datawhalechina/self-llm/blob/master/models/LLaMA3/04-LLaMA3-8B-Instruct%20Lora%20%E5%BE%AE%E8%B0%83.md)*

在命令行运行以下指令：
    
```shell
python train.py
```

> *注意：记得修改 `train.py` 中的数据集路径和模型路径哦~*

训练大概会需要 *20 ~ 30* 分钟的时间，训练完成之后会在`output`目录下生成`lora`模型。可以使用以下代码进行测试：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = './LLM-Research/Meta-Llama-3___1-8B-Instruct'
lora_path = './output/llama3_1_instruct_lora/checkpoint-699' # 这里改称你的 lora 输出对应 checkpoint 地址

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

prompt = "嬛嬛你怎么了，朕替你打抱不平！"

messages = [
        {"role": "system", "content": "假设你是皇帝身边的女人--甄嬛。"},
        {"role": "user", "content": prompt}
]

input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# print(input_ids)

model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=512)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print('皇上：', prompt)
print('嬛嬛：',response)
```

```
皇上： 嬛嬛你怎么了，朕替你打抱不平！
嬛嬛： 皇上，臣妾不是故意的。
```

接下来，我们就可以使用这个嬛嬛模型进行对话了~   
有兴趣的同学可以尝试使用 `self-llm` 中的其他模型进行微调，检验你的学习成果！

## 写在最后

*Chat-嬛嬛是在去年LLM刚火起来的时候，我们觉得如果不做点什么的话，可能会错过很多有趣的事情。于是就和几个小伙伴一起，花了很多时间，做了这个项目。在这个项目中，我们学到了很多，也遇到了很多问题，但是我们都一一解决了。并且Chat-嬛嬛也获得了奖项，让项目得到了很多人的关注。所以，我觉得这个项目是非常有意义的，也是非常有趣的。*

- *2023 讯飞星火杯人认知大模型场景创新赛 Top50*
- *2024 书生·浦语大模型挑战赛（春季赛）创意应用奖 Top12*

### Chat-嬛嬛贡献者

- [宋志学](https://github.com/KMnO4-zx)（Datawhale成员-中国矿业大学(北京)）
- [邹雨衡](https://github.com/logan-zou)（Datawhale成员-对外经济贸易大学）
- [王熠明](https://github.com/Bald0Wang)（Datawhale成员-宁夏大学）
- [邓宇文](https://github.com/GKDGKD)（Datawhale成员-广州大学）
- [杜森](https://github.com/coderdeepstudy)（Datawhale成员-南阳理工学院）
- [肖鸿儒](https://github.com/Hongru0306)（Datawhale成员-同济大学）
