# CharacterGLM-6B Transformers部署调用

## 环境准备

在autodl平台中租一个3090等24G显存的显卡机器，如下图所示镜像选择PyTorch-->2.0.0-->3.8(ubuntu20.04)-->11.8

![image](https://github.com/suncaleb1/self-llm/assets/155936975/fc4c6323-d338-4d66-a244-bbefe7da3746)

接下来打开刚刚租用服务器的JupyterLab，并且打开其中的终端开始环境配置、模型下载和运行demo。

pip换源和安装依赖包

```python
#升级pip
python -m pip install --upgrade pip
#更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope
pip install transformers
pip install sentencepiece
```

## 模型下载

使用 modelscope 中的snapshot_download函数下载模型，第一个参数为模型名称，参数cache_dir为模型的下载路径。

在 /root/autodl-tmp 路径下新建 download.py 文件并在其中输入以下内容，粘贴代码后记得保存文件，如下图所示。并运行 python /root/autodl-tmp/download.py执行下载，模型大小为 12 GB，下载模型大概需要 10~15 分钟

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('THUCoAI/CharacterGLM-6B', cache_dir='/root/autodl-tmp', revision='master')
```

## 代码准备

```python
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
# 使用模型下载到的本地路径以加载
model_dir = '/root/autodl-tmp/THUCoAI/CharacterGLM-6B'
# 分词器的加载，本地加载，trust_remote_code=True设置允许从网络上下载模型权重和相关的代码
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# 模型加载，本地加载，使用AutoModelForCausalLM类
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
# 将模型移动到GPU上进行加速（如果有GPU的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 使用模型的评估模式来产生对话
model.eval()
session_meta = {'user_info': '我是陆星辰，是一个男性，是一位知名导演，也是苏梦远的合作导演。我擅长拍摄音乐题材的电影。苏梦远对我的态度是尊敬的，并视我为良师益友。', 'bot_info': '苏梦远，本名苏远心，是一位当红的国内女歌手及演员。在参加选秀节目后，凭借独特的嗓音及出众的舞台魅力迅速成名，进入娱乐圈。她外表美丽动人，但真正的魅力在于她的才华和勤奋。苏梦远是音乐学院毕业的优秀生，善于创作，拥有多首热门原创歌曲。除了音乐方面的成就，她还热衷于慈善事业，积极参加公益活动，用实际行动传递正能量。在工作中，她对待工作非常敬业，拍戏时总是全身心投入角色，赢得了业内人士的赞誉和粉丝的喜爱。虽然在娱乐圈，但她始终保持低调、谦逊的态度，深得同行尊重。在表达时，苏梦远喜欢使用“我们”和“一起”，强调团队精神。', 'bot_name': '苏梦远', 'user_name': '陆星辰'}
# 第一轮对话
response, history = model.chat(tokenizer, session_meta,"你好呀，小苏", history=[])
print(response)
# 第二轮对话
response, history = model.chat(tokenizer, session_meta,"最近对音乐有什么新的想法吗", history=history)
print(response)
# 第三轮对话
response, history = model.chat(tokenizer,session_meta, "那我们商量一下下一部音乐电影的拍摄，好嘛？", history=history)
print(response)
```

## 部署

在终端输入以下命令运行trans.py，即实现CharacterGLM-6B的Transformers部署调用

```python
cd /root/autodl-tmp
python trans.py
```

观察命令行中loading checkpoint表示模型正在加载，等待模型加载完成产生对话，如下图所示

![image](https://github.com/suncaleb1/self-llm/assets/155936975/f9d65275-fa89-4039-95c5-7cc0615753e2)

