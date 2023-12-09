# DeepSeek-7B-chat langchain 接入

这篇主要讲 `DeepSeek-7B-chat` 如何对接`Langchain`中 `langchain.llms.base` 的 `LLM` 模块，其他关于如何对接向量数据库和`gradio`的部分请参考[internLM langchain](../InternLM/06-InternLM接入LangChain搭建知识库助手.md)模块。

## 安装依赖

除了需要安装模型的运行依赖之外，还需要安装 langchain 依赖。

```bash
pip install langchain==0.0.292
```

## DeepSeek-7B-chat 接入 LangChain

为便捷构建 LLM 应用，我们需要基于本地部署的 DeepSeek-7B-chat，自定义一个 LLM 类，将 DeepSeek-7B-chat 接入到 LangChain 框架中。完成自定义 LLM 类之后，可以以完全一致的方式调用 LangChain 的接口，而无需考虑底层模型调用的不一致。

基于本地部署的 DeepSeek-7B-chat 自定义 LLM 类并不复杂，我们只需从 LangChain.llms.base.LLM 类继承一个子类，并重写构造函数与 `_call` 函数即可：

```python
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

class DeepSeek_LLM(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path :str):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,torch_dtype=torch.bfloat16,  device_map="auto")
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.model = self.model.eval()
        print("完成本地模型的加载")

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        # 重写调用函数
        messages = [
            {"role": "user", "content": prompt}
        ]
        # 构建输入     
        input_tensor = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        # 通过模型获得输出
        outputs = self.model.generate(input_tensor.to(self.model.device), max_new_tokens=100)
        response = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return response
        
    @property
    def _llm_type(self) -> str:
        return "DeepSeek_LLM"
```

## 调用

然后就可以像使用任何其他的langchain大模型功能一样使用了。

```python
llm = DeepSeek_LLM('/root/autodl-tmp/deepseek-ai/deepseek-llm-7b-chat')

llm('你好')
```

如下图所示：

![Alt text](images/image-4.png)