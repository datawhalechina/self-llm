# DeepSeek-MoE-16b-chat Transformers 部署调用

## DeepSeek-MoE-16b-chat 介绍

DeepSeek MoE目前推出的版本参数量为160亿，实际激活参数量大约是28亿。与自家的7B密集模型相比，二者在19个数据集上的表现各有胜负，但整体比较接近。而与同为密集模型的Llama 2-7B相比，DeepSeek MoE在数学、代码等方面还体现出来明显的优势。但两种密集模型的计算量都超过了180TFLOPs每4k token，DeepSeek MoE却只有74.4TFLOPs，只有两者的40%。

## 环境准备
在autodl平台中租一个**双卡3090等24G（共计48G）**显存的显卡机器，如下图所示镜像选择PyTorch-->2.1.0-->3.10(ubuntu22.04)-->12.1
接下来打开刚刚租用服务器的JupyterLab， 图像 并且打开其中的终端开始环境配置、模型下载和运行演示。 
![Alt text](images/image-6.png)

接下来打开刚刚租用服务器的`JupyterLab`，并且打开其中的终端开始环境配置、模型下载和运行`demo`。

pip换源和安装依赖包

```shell
# 因为涉及到访问github因此最好打开autodl的学术镜像加速
source /etc/network_turbo
# 升级pip
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install modelscope transformers sentencepiece accelerate
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## 模型下载

使用 `modelscope` 中的`snapshot_download`函数下载模型，第一个参数为模型名称，参数`cache_dir`为模型的下载路径。

在 `/root/autodl-tmp` 路径下新建 `download.py` 文件并在其中输入以下内容，粘贴代码后记得保存文件，如下图所示。并运行 `python /root/autodl-tmp/download.py`执行下载，模型大小为 30 GB，下载模型大概需要 10~20 分钟

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('deepseek-ai/deepseek-moe-16b-chat', cache_dir='/root/autodl-tmp', revision='master')
```

## 代码准备

在/root/autodl-tmp路径下新建trains.py文件并在其中输入以下内容
```python
import torch  # 导入torch库，用于深度学习相关操作
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig  # 从transformers库导入所需的类

# 将模型路径设置为刚刚下载的模型路径
model_name = "/root/autodl-tmp/deepseek-ai/deepseek-moe-16b-chat"

# 加载分词器，trust_remote_code=True允许加载远程代码
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 加载语言模型，设置数据类型为bfloat16以优化性能（以免爆显存），并自动选择GPU进行推理
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

# 加载并设置生成配置，使用与模型相同的设置
model.generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)

# 将填充令牌ID设置为与结束令牌ID相同，用于生成文本的结束标记
model.generation_config.pad_token_id = model.generation_config.eos_token_id

# 定义输入消息，模型使用apply_chat_template进行消息输入，模拟用户与模型的交互
messages = [
    {"role": "user", "content": "你是谁"}
]

# 处理输入消息，并添加生成提示
input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

# 使用模型生成回应，设置max_new_tokens数量为100（防止爆显存）也可以将max_new_tokens设置的更大，但可能爆显存
outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

# 模型输出，跳过特殊令牌以获取纯文本结果
result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)

# 显示生成的回答
print(result)
```
### 部署

在终端输入以下命令运行trains.py，即实现DeepSeek-MoE-16b-chat的Transformers部署调用

```shell
cd /root/autodl-tmp
python trains.py
```
观察命令行中loading checkpoint表示模型正在加载，等待模型加载完成产生对话，如下图所示
![image](images/image-7.png)
