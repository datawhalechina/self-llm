#llm #GGUF #llamacpp 
## 什么是GGUF

GGUF 格式的全名为（GPT-Generated Unified Format），提到 GGUF 就不得不提到它的前身 GGML（GPT-Generated Model Language）。GGML 是专门为了机器学习设计的张量库，最早可以追溯到 2022/10。其目的是为了有一个单文件共享的格式，并且易于在不同架构的 GPU 和 CPU 上进行推理。但在后续的开发中，遇到了灵活性不足、相容性及难以维护的问题。

根据上面Ollama 部分的介绍我们可以发现, pytorch训练后的模型文件不能直接导入到ollama中, 需要将其预先转为`.gguf` 文件. 本节将讲述如何将训练后的模型导出为gguf 文件.

## 动手转化基础模型和LoRA 文件到GGUF

### 从huggingface download 基础模型
最直觉是用 git clone 来下载模型，但是因为 LLM 每个一部分都按 GB 来计算，避免出现 OOM Error 的情况，简单用 Python 写一个 download.py 比较简单

```shell
#安装huggingface_hub
pip install huggingface_hub
```

执行下面的代码

```python
from huggingface_hub import snapshot_download
model_id="unsloth/Llama-3.2-1B-Instruct"
snapshot_download(repo_id=model_id, local_dir="yout_path",
                          local_dir_use_symlinks=False, revision="main")
```

### 转换模型到GGUF
#### 克隆llama.cpp
在控制台中部署下面的代码,clone 项目到本地

```shell
git clone https://github.com/ggerganov/llama.cpp.git

cd llama.cpp
```

直接从[release](https://github.com/ggerganov/llama.cpp/releases/tag/b3878) 中下载最新的预编译好的文件到本地并解压缩到llama.cpp目录下. ==**注意: 请下载带有CUDA 编译的版本, 请根据自己的Cuda版本进行选择**==
![[models/Llama3_1/images/06-1.png]]
#### 创建虚拟环境
##### 使用conda

```shell
conda create -n llamacpp
conda activate llamacpp
```

##### 使用venv

```shell
python -m venv llamacpp
.\venv\Scripts\activate
```

#### 安装环境依赖

```shell
pip install requirement.txt
```

#### 转换基础模型

运行`convert-hf-to-gguf.py`, 此脚本用于将 Hugging Face 格式的模型转换为 GGUF 格式。请按照以下步骤操作：

1. 确保您已经在`llama.cpp`目录下，并且已经激活了 Python 虚拟环境。

2. 使用以下命令运行脚本：
```shell
python convert_hf_to_gguf.py --outfile <gguf_path> <base_model_path>
```
- `--outfile` 参数指定转换后的 GGUF 文件的输出路径。
- 最后的参数是您要转换的 Hugging Face 模型的路径。

请确保在运行这些脚本之前，所有路径和文件名都是正确的，并且所需的依赖项已经安装完毕。

通过日志可以看到导出成功
![[models/Llama3_1/images/06-2.png]]
#### 转换LoRA 适配器

继续使用下面的命令运行脚本
```shell
python .\convert_lora_to_gguf.py --outfile <lora_gguf_path> --base <base_model_path>   <lora_model_path>
```

- `--outfile`参数指定转换后的GGUF 文件的输出路径
- `--base` 参数指定转换gguf 前的基础模型的地址
- 最后的参数是您要转换的LoRA 模型的路径
---

## 总结

将模型和LoRA适配器转换为GGUF格式需以下步骤：下载LLM至本地，准备环境并安装依赖，使用脚本将Hugging Face模型转为GGUF格式，再将LoRA适配器转为GGUF。后续可以按照 [[07-Llama3_1-8b-Instruct Ollama 部署]]完成gguf 的部署工作
