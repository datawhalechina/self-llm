# GLM-4-9b-Chat 接入 LangChain 

## 环境准备

在 `01-ChatGLM4-9B-chat FastApi 部署调用` 的 `环境准备`和`模型下载`基础上，我们还需要安装 `langchain` 包。如果不需要使用fastapi相关功能，则可以不安装 `fastapi、uvicorn、requests`。

```bash
pip install langchain==0.2.1
```
注意langchain这里使用2024年5月新发布的v0.2版本, 但本教程代码经过测试，也兼容langchain的0.1.15版本，下载方式如下：
```bash
pip install langchain==0.1.15
```

> 考虑到部分同学配置环境可能会遇到一些问题，我们在 AutoDL 平台准备了 GLM-4 的环境镜像，该镜像适用于本教程需要 GLM-4 的部署环境。点击下方链接并直接创建 AutoDL 示例即可。（vLLM 对 torch 版本要求较高，且越高的版本对模型的支持更全，效果更好，所以新建一个全新的镜像。） **https://www.codewithgpu.com/i/datawhalechina/self-llm/GLM-4**

## 代码准备

为便捷构建 LLM 应用，我们需要基于本地部署的 Chat，自定义一个 LLM 类，将 ChatGLM4 接入到 LangChain 框架中。完成自定义 LLM 类之后，可以以完全一致的方式调用 LangChain 的接口，而无需考虑底层模型调用的不一致。

基于本地部署的 ChatGLM4 自定义 LLM 类并不复杂，我们只需从 Langchain.llms.base.LLM 类继承一个子类，并重写构造函数与 _call 函数即可：

```python
from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ChatGLM4_LLM(LLM):
    # 基于本地 ChatGLM4 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    gen_kwargs: dict = None
        
    def __init__(self, mode_name_or_path: str, gen_kwargs: dict = None):
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            mode_name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            mode_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        print("完成本地模型的加载")
        
        if gen_kwargs is None:
            gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        self.gen_kwargs = gen_kwargs
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:
        messages = [{"role": "user", "content": prompt}]
        model_inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        generated_ids = self.model.generate(**model_inputs, **self.gen_kwargs)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs['input_ids'], generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回用于识别LLM的字典,这对于缓存和跟踪目的至关重要。"""
        return {
            "model_name": "glm-4-9b-chat",
            "max_length": self.gen_kwargs.get("max_length"),
            "do_sample": self.gen_kwargs.get("do_sample"),
            "top_k": self.gen_kwargs.get("top_k"),
        }

    @property
    def _llm_type(self) -> str:
        return "glm-4-9b-chat"
```
在上述类定义中，我们分别重写了构造函数和 _call 函数： 对于构造函数，我们在对象实例化的一开始加载本地部署的 ChatGLM4 模型，从而避免每一次调用都需要重新加载模型带来的时间浪费; _call 函数是 LLM 类的核心函数，Langchain 会调用改函数来调用LLM，在改函数中，我们调用已实例化模型的 generate 方法，从而实现对模型的调用并返回调用结果。

此外，在实现自定义 LLM 类时，按照 langchain 框架的要求，我们需要定义 _identifying_params 属性。这个属性的作用是返回一个字典，该字典包含了能够唯一标识这个 LLM 实例的参数。这个功能对于缓存和追踪非常重要，因为它能够帮助系统识别不同的模型配置，从而进行有效的缓存管理和日志追踪。

在整体项目中，我们将上诉代码封装为 LLM.py，后续将直接从该文件中引入自定义的 ChatGLM4_LLM 类

## 调用

然后就可以像使用任何其他的langchain大模型功能一样使用了。  

```python
from LLM import ChatGLM4_LLM
gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
llm = ChatGLM4_LLM(mode_name_or_path="/root/autodl-tmp/ZhipuAI/glm-4-9b-chat", gen_kwargs=gen_kwargs)
print(llm.invoke("你是谁"))
```

![模型返回回答效果](images/image02-1.png)

