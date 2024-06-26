# Yuan2.0-M32 接入 LangChain 搭建知识库助手

## 环境准备

在 Autodl 平台中租赁一个 RTX 3090/24G 显存的显卡机器。如下图所示，镜像选择 PyTorch-->2.1.0-->3.10(ubuntu22.04)-->12.1。

![开启机器配置选择](images/01-1.png)

接下来，我们打开刚刚租用服务器的 JupyterLab，如下图所示。

![开启JupyterLab](images/01-2.png)

然后打开其中的终端，开始环境配置、模型下载和运行演示。  

![开启终端](images/01-3.png)

## 环境配置

Yuan2-M32-HF-INT4是由原始的Yuan2-M32-HF经过auto-gptq量化而来的模型。

通过模型量化，部署Yuan2-M32-HF-INT4对显存和硬盘的要求都会显著减低。

注：由于pip版本的auto-gptq目前还不支持Yuan2.0 M32，因此需要编译安装

```shell
# 升级pip
python -m pip install --upgrade pip

# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 拉取Yuan2.0-M32项目
git clone https://github.com/IEIT-Yuan/Yuan2.0-M32.git

# 进入AutoGPTQ
cd  Yuan2.0-M32/3rd_party/AutoGPTQ

# 安装autogptq
pip install --no-build-isolation -e .

# 安装 einops langchain modelscope
pip install einops langchain modelscope
```

> 考虑到部分同学配置环境可能会遇到一些问题，我们在AutoDL平台准备了Yuan2.0-M32的镜像，点击下方链接并直接创建Autodl示例即可。
> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/Yuan2.0-M32***


## 模型下载  

使用 modelscope 中的 snapshot_download 函数下载模型，第一个参数为模型名称，参数 cache_dir 为模型的下载路径。

这里可以先进入autodl平台，初始化机器对应区域的的文件存储，文件存储路径为'/root/autodl-fs'。
该存储中的文件不会随着机器的关闭而丢失，这样可以避免模型二次下载。

![autodl-fs](images/autodl-fs.png)

然后运行下面代码，执行模型下载。

```python
from modelscope import snapshot_download
model_dir = snapshot_download('YuanLLM/Yuan2-M32-HF-INT4', cache_dir='/root/autodl-fs')
``` 

## 模型合并

下载后的模型为多个文件，需要将其进行合并。

```shell
cat /root/autodl-fs/YuanLLM/Yuan2-M32-HF-INT4/gptq_model-4bit-128g.safetensors*  > /root/autodl-fs/YuanLLM/Yuan2-M32-HF-INT4/gptq_model-4bit-128g.safetensors
```

## 代码准备

为便捷构建 LLM 应用，我们需要基于本地部署的 Yuan2，自定义一个 LLM 类，将 Yuan2 接入到 LangChain 框架中。

完成自定义 LLM 类之后，可以以完全一致的方式调用 LangChain 的接口，而无需考虑底层模型调用的不一致。

基于本地部署的 Yuan2 自定义 LLM 类并不复杂，我们只需从 LangChain.llms.base.LLM 类继承一个子类，并重写构造函数与 _call 函数即可：

```python
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from auto_gptq import AutoGPTQForCausalLM
from transformers import LlamaTokenizer
import torch

class Yuan2_LLM(LLM):
    # 基于本地 Yuan2 自定义 LLM 类
    tokenizer: LlamaTokenizer = None
    model: AutoGPTQForCausalLM = None

    def __init__(self, mode_name_or_path :str):
        super().__init__()

        # 加载预训练的分词器和模型
        print("Creat tokenizer...")
        self.tokenizer = LlamaTokenizer.from_pretrained(mode_name_or_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
        self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

        print("Creat model...")
        self.model = AutoGPTQForCausalLM.from_quantized(mode_name_or_path, trust_remote_code=True).cuda()

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):

        prompt += "<sep>"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=256)
        output = self.tokenizer.decode(outputs[0])
        response = output.split("<sep>")[-1]

        return response

    @property
    def _llm_type(self) -> str:
        return "Yuan2_LLM"
```

在上述类定义中，我们分别重写了构造函数和 _call 函数：对于构造函数，我们在对象实例化的一开始加载本地部署的 Yuan2 模型，从而避免每一次调用都需要重新加载模型带来的时间过长；_call 函数是 LLM 类的核心函数，LangChain 会调用该函数来调用 LLM，在该函数中，我们调用已实例化模型的 generate 方法，从而实现对模型的调用并返回调用结果。

在整体项目中，我们将上述代码封装为 LLM.py，后续将直接从该文件中引入自定义的 LLM 类。


## 调用

然后就可以像使用任何其他的langchain大模型功能一样使用了。

```python
from LLM import Yuan2_LLM
llm = Yuan2_LLM('/root/autodl-fs/YuanLLM/Yuan2-M32-HF-INT4')
print(llm("你是谁"))
```

![alt text](./images/02-0.png)
