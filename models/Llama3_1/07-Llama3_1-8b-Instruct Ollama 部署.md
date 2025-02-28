
#llm #ollama #LoRA部署 

## 引言 

在上面的[[04-LLaMA3-8B-Instruct Lora 微调]]中, 介绍了构建大语言模型微调的数据集格式,数据集预处理, Q-LoRA 微调大模型, 本章我们将来到**大模型的部署**环节.
在模型的部署场景我们会希望部署后的模型服务可以提供一个标准的API, 同时部署模型的框架还要对Cuda 保持兼容性,**Ollama**就成为了很好的选择.
本文将通过**模型导入到Ollama**完成对大模型+LoRA的本地部署.

## Ollama 介绍

Ollama是一个用于简化大语言模型本地部署和运行的工具,它提供了一个轻量\易于扩展的框架, 让开发者可以在本地部署和管理大模型. 通过Ollama,开发者可以访问和管理原始pre-trained 大模型,也可以导入自己的大模型, 而无需关注底层细节.

Ollama 主要支持的是**GGUF**格式的模型文件. GGUF（GPT-Generated Unified Format）是专为高效运行 LLM 设计的二进制格式，由 `llama.cpp` 项目引入，取代了旧版 GGML 格式。可以从Hugging face 下载预转换的GGUF 原始模型, 或者使用`llama.cpp` 自行转换

### 基于GGUF文件创建模型

如果你有一个基于 GGUF 的模型或适配器，可以将其导入 Ollama。你可以通过以下方式获取 GGUF 模型或适配器：

- 使用 Llama.cpp 中的 `convert_hf_to_gguf.py` 脚本将 Safetensors 模型转换为 GGUF 模型；
- 使用 Llama.cpp 中的 `convert_lora_to_gguf.py` 脚本将 Safetensors 适配器转换为 GGUF 适配器；或
- 从 HuggingFace 等地方下载模型或适配器

具体可参考本文 [[06-Llama3_1-8b-Instruct LoRA GGUF 转换]]
`Modelfile`是一个模型的配置文件, 需要包含如下信息:

*   `FROM`(必须):模型的GGUF 文件地址
*  `ADAPTER`: 模型的LORA 适配器地址, 也需要转化为GGUF 文件
*   `TEMPLATE`: 模型的提示模板, **建议直接沿用基模型的TEMPLATE**

```Modelfile
FROM /path/to/model.gguf
ADAPTER /path/to/adapter.gguf
TEMPLATE 基础模型的template
```

创建`my_lora_adapter.Modelfile` 文件, 并按照上述模板填写信息, 完成创建后通过下面的语句创建模型

```shell
ollama create <modelname> -F my_lora_adapter.Modelfile
```
再次执行`ollama list` 即可看到自己的模型啦

![07-1.png](./images/07-1.png)

# 总结

本文首先介绍了ollama以及Modelfile 文件, 基于这些命令可以快速使用ollama 部署微调后的Lora 适配器