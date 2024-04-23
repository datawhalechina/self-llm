# XVERSE-MoE-A4.2B Transformers 部署调用
## XVERSE-MoE-A4.2B介绍
**XVERSE-MoE-A4.2B** 是由深圳元象科技自主研发的支持多语言的大语言模型（Large Language Model），使用混合专家模型（MoE，Mixture-of-experts）架构，模型的总参数规模为 258 亿，实际激活的参数量为 42 亿，本次开源的模型为底座模型 **XVERSE-MoE-A4.2B**，主要特点如下：

- **模型结构**：XVERSE-MoE-A4.2B 为 Decoder-only 的 Transformer 架构，将密集模型的 FFN 层扩展为专家层，不同于传统 MoE 中每个专家的大小与标准 FFN 相同（如Mixtral 8x7B ），使用了更细粒度的专家，每个专家是标准 FFN 大小的 1/4，并设置了共享专家（Shared Expert）和非共享专家（Non-shared Expert）两类，共享专家在计算时始终被激活，非共享专家通过 Router 选择性激活。
- **训练数据**：构建了 2.7 万亿 token 的高质量、多样化的数据对模型进行充分训练，包含中、英、俄、西等 40 多种语言，通过精细化设置不同类型数据的采样比例，使得中英两种语言表现优异，也能兼顾其他语言效果；模型使用 8K 长度的训练样本进行训练。
- **训练框架**：针对 MoE 模型中独有的专家路由和权重计算逻辑，进行了深入定制优化，开发出一套高效的融合算子，以提升计算效率。同时，为解决 MoE 模型显存占用和通信量大的挑战，设计了计算、通信和 CPU-Offload 的 Overlap 处理方式，从而提高整体吞吐量。

**XVERSE-MoE-A4.2B** 的模型大小、架构和学习率如下：

| total params | activated params | n_layers | d_model | n_heads | d_ff | n_non_shared_experts | n_shared_experts | top_k |   lr   |
| :----------: | :--------------: | :------: | :-----: | :-----: | :--: | :------------------: | :--------------: | :---: | :----: |
|    25.8B     |       4.2B       |    28    |  2560   |   32    | 1728 |          64          |        2         |   6   | 3.5e−4 |

但是 XVERSE 的仓库并没有更新更多的实践案例，还是需要大家丰富一下的，我有时间也会分享更多案例的。
有关 XVERSE-MoE-A4.2B 模型的相关报告可以看：[元象首个MoE大模型开源：4.2B激活参数，效果堪比13B模型](https://mp.weixin.qq.com/s/U_ihKmhRD6Xc0cZ8hMJ1SQ)

## 讲讲显存计算
显存计算的考虑会随着模型类型不同，任务不同而变化

这里的Transformers部署调用是推理任务，因而只需要考虑模型参数、KV Cache、中间结果和输入数据。这里的模型为MoE模型，考虑完整模型参数（25.8B）；使用了bf16加载，再考虑中间结果、输入数据和KV Cache等，大概是`2x1.2x25.8`的显存需求，所以我们后面会选择三卡共72G显存，显存要求还是挺大的大家根据自己条件自行尝试吧。

更完整的显存计算参照这个blog：[【Transformer 基础系列】手推显存占用](https://zhuanlan.zhihu.com/p/648924115)
## 环境准备
在autodl平台中租一个**三卡3090等24G（共计72G）显存**的机器，如下图所示镜像选择PyTorch-->2.1.0-->3.10(ubuntu22.04)-->12.1
接下来打开刚刚租用服务器的JupyterLab， 图像 并且打开其中的终端开始环境配置、模型下载和运行演示。 
![Alt text](images/1.png)
pip换源和安装依赖包
```shell
# 因为涉及到访问github因此最好打开autodl的学术镜像加速
source /etc/network_turbo
# 升级pip
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 从transformers的github仓库中安装包含XVERSE-MoE的新版本
# 如果安装不上可以使用 pip install git+https://github.moeyy.xyz/https://github.com/huggingface/transformers
pip install git+https://github.com/huggingface/transformers
# 安装需要的python包
pip install modelscope sentencepiece accelerate fastapi uvicorn requests streamlit transformers_stream_generator
# 安装flash-attention
# 这个也是不行使用 pip install https://github.moeyy.xyz/https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
## 模型下载
使用ModelScope下载模型
```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('xverse/XVERSE-MoE-A4.2B', cache_dir='/root/autodl-tmp', revision='master')
```
## 代码准备
在/root/autodl-tmp路径下新建trains.py文件并在其中输入以下内容
```python
import torch  # 导入torch库，用于深度学习相关操作
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig  # 三个类分别用于加载分词器、加载因果语言模型和加载生成配置

# 将模型路径设置为刚刚下载的模型路径
model_name = "/root/autodl-tmp/xverse/XVERSE-MoE-A4.2B"

# 加载语言模型，设置数据类型为bfloat16即混合精度格式以优化性能并减少显存使用，将推理设备设置为`auto`自动选择最佳的设备进行推理，如果没有可用的GPU，它可能会回退到CPU
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 定义input字符串
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
# 使用分词器的apply_chat_template方法来处理messages，转换格式
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True # 在消息前添加生成提示
)
# 将text变量中的文本转换为模型输入的格式，指定返回的张量为PyTorch张量（"pt"）
model_inputs = tokenizer([text], return_tensors="pt").to(device)
# 使用模型的generate方法来生成文本
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
# 从生成的ID中提取出除了原始输入之外的新生成的token
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
# 使用分词器的batch_decode方法将生成的token ID转换回文本
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# 显示生成的回答
print(response)
```