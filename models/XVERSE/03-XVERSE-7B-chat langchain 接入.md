# XVERSE-7B-chat langchain 接入

XVERSE-7B-Chat为[XVERSE-7B](https://huggingface.co/xverse/XVERSE-7B)模型对齐后的版本。

XVERSE-7B 是由深圳元象科技自主研发的支持多语言的大语言模型（Large Language Model），参数规模为 70 亿，主要特点如下：

- 模型结构：XVERSE-7B 使用主流 Decoder-only 的标准 Transformer 网络结构，支持 8K 的上下文长度（Context Length），能满足更长的多轮对话、知识问答与摘要等需求，模型应用场景更广泛。
- 训练数据：构建了 2.6 万亿 token 的高质量、多样化的数据对模型进行充分训练，包含中、英、俄、西等 40 多种语言，通过精细化设置不同类型数据的采样比例，使得中英两种语言表现优异，也能兼顾其他语言效果。
- 分词：基于 BPE（Byte-Pair Encoding）算法，使用上百 GB 语料训练了一个词表大小为 100,534 的分词器，能够同时支持多语言，而无需额外扩展词表。
- 训练框架：自主研发多项关键技术，包括高效算子、显存优化、并行调度策略、数据-计算-通信重叠、平台和框架协同等，让训练效率更高，模型稳定性强，在千卡集群上的峰值算力利用率可达到 58.5%，位居业界前列。


## 环境准备  

在 Autodl 平台中租赁一个 3090 等 24G 显存的显卡机器，如下图所示镜像选择 PyTorch-->2.1.0-->3.10(ubuntu22.04)-->12.1（11.3 版本以上的都可以）。  

![机器配置选择](images/1.png)
接下来打开刚刚租用服务器的 JupyterLab，并且打开其中的终端开始环境配置、模型下载和运行 demo。

pip 换源加速下载并安装依赖包，为了方便大家进行环境配置，在 code 文件夹里面给大家提供了 requirement.txt 文件，大家直接使用下面的命令安装即可。如果你使用的是 [autodl](https://www.autodl.com/) 部署模型的话，我们有制作好的镜像供大家使用：[XVERSE-7B-Chat](https://www.codewithgpu.com/i/datawhalechina/self-llm/XVERSE-7B-Chat)

```shell
# 升级pip
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install -r requirement.txt
```  

## 模型下载  

使用 modelscope 中的 snapshot_download 函数下载模型，第一个参数为模型名称，参数 cache_dir 为模型的下载路径。

在 /root/autodl-tmp 路径下新建 [model_download.py](code/model_download.py) 文件并在其中输入以下内容，粘贴代码后请及时保存文件，如下图所示。并运行 `python /root/autodl-tmp/model_download.py` 执行下载，模型大小为 14GB，下载模型大概需要 2 分钟。

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('xverse/XVERSE-7B-Chat', cache_dir='/root/autodl-tmp', revision='master')
```  

## 代码准备

> 为了方便大家部署，在 code 文件夹里面已经准备好了代码，大家可以将仓库 clone 到服务器上直接运行。

为便捷构建 LLM 应用，我们需要基于本地部署的 XVERSE-LLM，自定义一个 LLM 类，将 XVERSE 接入到 LangChain 框架中。完成自定义 LLM 类之后，可以以完全一致的方式调用 LangChain 的接口，而无需考虑底层模型调用的不一致。

基于本地部署的 XVERSE 自定义 LLM 类并不复杂，我们只需从 LangChain.llms.base.LLM 类继承一个子类，并重写构造函数与 _call 函数即可，下面创建 [LLM.py](code/LLM.py) 文件，内容如下：

```python
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LlamaTokenizerFast
import torch

class XVERSE_LLM(LLM):
    # 基于本地 XVERSE-7B-Chat 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
        
    def __init__(self, mode_name_or_path :str):

        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
        self.model = self.model.eval()
        print("完成本地模型的加载")
        
    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):

        # 构建消息
        history = [{"role": "user", "content": prompt}]

        response = self.model.chat(self.tokenizer, history)
        
        return response
    @property
    def _llm_type(self) -> str:
        return "XVERSE_LLM"
```

在上述类定义中，我们分别重写了构造函数和 _call 函数：对于构造函数，我们在对象实例化的一开始加载本地部署的 XVERSE-7B 模型，从而避免每一次调用都需要重新加载模型带来的时间过长；_call 函数是 LLM 类的核心函数，LangChain 会调用该函数来调用 LLM，在该函数中，我们调用已实例化模型的 chat 方法，从而实现对模型的调用并返回调用结果。

在整体项目中，我们将上述代码封装为 [LLM.py](code/LLM.py)，后续将直接从该文件中引入自定义的 LLM 类。

## 代码运行

然后就可以像使用任何其他的langchain大模型功能一样使用了。

```python
from LLM import XVERSE_LLM

llm = XVERSE_LLM(mode_name_or_path = "/root/autodl-tmp/xverse/XVERSE-7B-Chat")
llm("你好呀，你能帮助我干什么啊？")
```

运行结果如下：
```json
'当然可以，我可以帮助您做很多事情。例如：\n\n1. 回答问题：无论是关于天气、新闻、历史、科学、数学、艺术等各种领域的问题，我都会尽力提供准确的信息。\n2. 设置提醒和闹钟：告诉我你需要提醒或设定的时间，我会在指定的时间通知你。\n3. 发送消息：如果你想给某人发送消息，只需告诉我收件人的名字或者联系方式，我就可以帮你完成。\n4. 查询路线：输入你的目的地，我可以为你规划出最佳的行驶路线。\n5. 播放音乐和电影：只要告诉我你想听哪首歌或者看哪部电影，我就可以为你播放。\n6. 学习新知识：如果你有任何学习需求，比如学习新的语言、技能等，我也可以提供相关的学习资源和建议。\n7. 进行简单的计算：如果你需要进行复杂的数学运算，我可能无法直接解答，但我可以尝试提供一些基本的计算方法。\n8. 提供新闻更新：我可以为你提供实时的新闻更新，让你随时了解世界大事。\n9. 其他许多任务：只要是你需要的，我都会尽力去做。'
```
![](images/4.png)