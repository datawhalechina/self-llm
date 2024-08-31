# Qwen1.5-7B-Chat vLLM 部署调用

## vLLM 简介
vLLM 框架是一个高效的大型语言模型（LLM）**推理和部署服务系统**，具备以下特性：

- **高效的内存管理**：通过 PagedAttention 算法，vLLM 实现了对 KV 缓存的高效管理，减少了内存浪费，优化了模型的运行效率。

- **高吞吐量**：vLLM 支持异步处理和连续批处理请求，显著提高了模型推理的吞吐量，加速了文本生成和处理速度。

- **易用性**：vLLM 与 HuggingFace 模型无缝集成，支持多种流行的大型语言模型，简化了模型部署和推理的过程。兼容 OpenAI 的 API 服务器。

- **分布式推理**：框架支持在多 GPU 环境中进行分布式推理，通过模型并行策略和高效的数据通信，提升了处理大型模型的能力。

- **开源**：vLLM 是开源的，拥有活跃的社区支持，便于开发者贡献和改进，共同推动技术发展。

## 环境准备  
在 Autodl 平台中租赁一个 3090 等 24G 显存的显卡机器，如下图所示镜像选择 PyTorch-->2.1.0-->3.10(ubuntu22.04)-->12.1

接下来打开刚刚租用服务器的 JupyterLab，并且打开其中的终端开始环境配置、模型下载和运行演示。  

![开启机器配置选择](images/Qwen1.5-vllm-gpu-select.png)

pip 换源加速下载并安装依赖包

```shell
# 升级pip
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.11.0
pip install openai==1.17.1
pip install torch==2.1.2+cu121
pip install tqdm==4.64.1
pip install transformers==4.39.3
pip install flash-attn==2.5.7 --no-build-isolation
pip install vllm==0.4.0.post1
```  


直接安装 vLLM 会安装 CUDA 12.1 版本。
```shell
pip install vllm
```

如果我们需要在 CUDA 11.8 的环境下安装 vLLM，可以使用以下命令，指定 vLLM 版本和 python 版本下载。
```shell
export VLLM_VERSION=0.4.0
export PYTHON_VERSION=38
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

> 考虑到部分同学配置环境可能会遇到一些问题，我们在 AutoDL 平台准备了 vLLM 的环境镜像，该镜像适用于任何需要 vLLM 的部署环境。点击下方链接并直接创建 AutoDL 示例即可。（vLLM 对 torch 版本要求较高，且越高的版本对模型的支持更全，效果更好，所以新建一个全新的镜像。）
> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/self-llm-vllm***


## 模型下载  

使用 modelscope 中的 snapshot_download 函数下载模型，第一个参数为模型名称，参数 cache_dir 为模型的下载路径。

在 /root/autodl-tmp 路径下新建 model_download.py 文件并在其中输入以下内容，粘贴代码后请及时保存文件，如下图所示。并运行 `python /root/autodl-tmp/model_download.py` 执行下载，模型大小为 14GB，下载模型大概需要 2 分钟。

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('qwen/Qwen1.5-7B-Chat', cache_dir='/root/autodl-tmp', revision='master')
```  

## 代码准备  

### python 文件
在 /root/autodl-tmp 路径下新建 vllm_model.py 文件并在其中输入以下内容，粘贴代码后请及时保存文件。下面的代码有很详细的注释，大家如有不理解的地方，欢迎提出 issue。  


首先从 vLLM 库中导入 LLM 和 SamplingParams 类。`LLM`类是使用vLLM引擎运行离线推理的主要类。`SamplingParams`类指定采样过程的参数，用于控制和调整生成文本的随机性和多样性。

vLLM 提供了非常方便的封装，我们直接传入模型名称或模型路径即可，不必手动初始化模型和分词器。

我们可以通过这个 demo 熟悉下 vLLM 引擎的使用方式。被注释的部分内容可以丰富模型的能力，但不是必要的，大家可以按需选择。


```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json

# 自动下载模型时，指定使用modelscope。不设置的话，会从 huggingface 下载
# os.environ['VLLM_USE_MODELSCOPE']='True'

def get_completion(prompts, model, tokenizer=None, max_tokens=512, temperature=0.8, top_p=0.95, max_model_len=2048):
    # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    # 初始化 vLLM 推理引擎
    llm = LLM(model=model, tokenizer=tokenizer, max_model_len=max_model_len)
    outputs = llm.generate(prompts, sampling_params)
    return outputs


if __name__ == "__main__":    
    # 初始化 vLLM 推理引擎
    model='/root/autodl-tmp/qwen/Qwen1.5-7B-Chat' # 指定模型路径
    # model="Qwen/Qwen1.5-MoE-A2.7B-Chat" # 指定模型名称，自动下载模型
    tokenizer = None
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False) # 加载分词器后传入vLLM 模型，但不是必要的。
    
    text = ["给我介绍一下大型语言模型。",
           "告诉我如何变强。"]
    # messages = [
    #     {"role": "system", "content": "你是一个有用的助手。"},
    #     {"role": "user", "content": prompt}
    # ]
    # 作为聊天模板的消息，不是必要的。
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )

    outputs = get_completion(text, model, tokenizer=tokenizer, max_tokens=512, temperature=1, top_p=1, max_model_len=2048)

    # 输出是一个包含 prompt、生成文本和其他信息的 RequestOutput 对象列表。
    # 打印输出。
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        
```
![vLLM 模型部署](images/Qwen1.5-vLLM.png)

### 创建兼容 OpenAI API 接口的服务器

Qwen 兼容 OpenAI API 协议，所以我们可以直接使用 vLLM 创建 OpenAI API 服务器。vLLM 部署实现 OpenAI API 协议的服务器非常方便。默认会在 http://localhost:8000 启动服务器。服务器当前一次托管一个模型，并实现列表模型、completions 和 chat completions 端口。

- completions：是基本的文本生成任务，模型会在给定的提示后生成一段文本。这种类型的任务通常用于生成文章、故事、邮件等。
- chat completions：是面向对话的任务，模型需要理解和生成对话。这种类型的任务通常用于构建聊天机器人或者对话系统。

在创建服务器时，我们可以指定模型名称、模型路径、聊天模板等参数。
- --host 和 --port 参数指定地址。
- --model 参数指定模型名称。
- --chat-template 参数指定聊天模板。
- --served-model-name 指定服务模型的名称。
- --max-model-len 指定模型的最大长度。

这里指定 `--max-model-len=2048` 是因为 Qwen1.5-7B-Chat 模型的最大长度过长 32768，导致 vLLM 初始化 KV 缓存时消耗资源过大。

```bash
python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/qwen/Qwen1.5-7B-Chat  --served-model-name Qwen1.5-7B-Chat --max-model-len=2048
```

1. 通过 curl 命令查看当前的模型列表。
```bash
curl http://localhost:8000/v1/models
```

得到的返回值如下所示：
```json
{"object":"list","data":[{"id":"Qwen1.5-7B-Chat","object":"model","created":1713201531,"owned_by":"vllm","root":"Qwen1.5-7B-Chat","parent":null,"permission":[{"id":"modelperm-b676428b47cb4ca19187876663da5eb3","object":"model_permission","created":1713201531,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
```

2. 使用 curl 命令测试 OpenAI Completions API 。
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen1.5-7B-Chat",
        "prompt": "你好",
        "max_tokens": 7,
        "temperature": 0
    }'
```

得到的返回值如下所示：
```json
{"id":"cmpl-ca4722e3c92a4e578da8f1f8fe378b35","object":"text_completion","created":1713201551,"model":"Qwen1.5-7B-Chat","choices":[{"index":0,"text":"，我有一个问题需要帮助解决","logprobs":null,"finish_reason":"length","stop_reason":null}],"usage":{"prompt_tokens":1,"total_tokens":8,"completion_tokens":7}}
```

也可以用 python 脚本请求 OpenAI Completions API 。
```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123", # 随便设，只是为了通过接口参数校验
)

completion = client.chat.completions.create(
  model="Qwen1.5-7B-Chat",
  messages=[
    {"role": "user", "content": "你好"}
  ]
)

print(completion.choices[0].message)
```
得到的返回值如下所示：
```
ChatCompletionMessage(content='你好！有什么我能帮助你的吗？', role='assistant', function_call=None, tool_calls=None)
```

3. 用 curl 命令测试 OpenAI Chat Completions API 。
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen1.5-7B-Chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你好"}
        ]
    }'
```

得到的返回值如下所示：
```json
{"id":"cmpl-6f002a1ddfa2420e83808032ed912809","object":"chat.completion","created":1713201596,"model":"Qwen1.5-7B-Chat","choices":[{"index":0,"message":{"role":"assistant","content":"你好！很高兴为你提供帮助。有什么问题或者需要咨询的吗？"},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":20,"total_tokens":36,"completion_tokens":16}}
```

也可以用 python 脚本请求 OpenAI Chat Completions API 。

```python
from openai import OpenAI
openai_api_key = "EMPTY" # 随便设，只是为了通过接口参数校验
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_outputs = client.chat.completions.create(
    model="Qwen1.5-7B-Chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"},
    ]
)
print(chat_outputs)
```

得到的返回值如下所示：
```
ChatCompletion(id='cmpl-1889c8c4e11240e3a6cab367b26d32b5', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='你好！很高兴能为你提供帮助。有什么问题或需要咨询的吗？', role='assistant', function_call=None, tool_calls=None), stop_reason=None)], created=1713201854, model='Qwen1.5-7B-Chat', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=17, prompt_tokens=20, total_tokens=37))
```

在处理请求时 API 后端也会打印一些日志和统计信息。
![](images/Qwen1.5-vllm-api-stat.png)

## 速度测试  

既然说 vLLM 是一个高效的大型语言模型推理和部署服务系统，那么我们就来测试一下模型的生成速度。看看和原始的速度有多大的差距。这里直接使用 vLLM 自带的 benchmark_throughput.py 脚本进行测试。可以将当前文件夹 benchmark_throughput.py 脚本放在 /root/autodl-tmp/ 下。或者大家可以自行[下载脚本](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_throughput.py)。

下面是一些 benchmark_throughput.py 脚本的参数说明：
- --model 参数指定模型路径或名称。
- --backend 推理后端，可以是 vllm、hf 和 mii。分布对应 vLLM、HuggingFace 和 Mii 推理后端。
- --input-len 输入长度
- --output-len 输出长度
- --num-prompts 生成的 prompt 数量
- --seed 2024 随机种子
- --dtype float16 浮点数精度
- --max-model-len 模型最大长度
- --hf_max_batch_size transformers 库的最大批处理大小（只有 hf 推理后端有效，且必须）
- --dataset 数据集路径。（未设置会自动生成数据）

测试 vLLM 的速度：
```bash
python benchmark_throughput.py \
	--model /root/autodl-tmp/qwen/Qwen1.5-7B-Chat \
	--backend vllm \
	--input-len 64 \
	--output-len 128 \
	--num-prompts 25 \
	--seed 2024 \
    --dtype float16 \
    --max-model-len 512
```
得到的结果如下所示：
```
Throughput: 6.34 requests/s, 1216.34 tokens/s
```
测试原始方式（使用 hunggingface 的 transformers 库）的速度：
```bash
python benchmark_throughput.py \
	--model /root/autodl-tmp/qwen/Qwen1.5-7B-Chat \
	--backend hf \
	--input-len 64 \
	--output-len 128 \
	--num-prompts 25 \
	--seed 2024 \
	--dtype float16 \
    --hf-max-batch-size 25
```
得到的结果如下所示：
```
Throughput: 4.03 requests/s, 773.74 tokens/s
```

对比两者的速度，在本次测试中 vLLM 的速度要比原始的速度快 **50%** 左右（本次测试相对比较随意，仅供本 case 参考，不对其他 case 有参考意义）。
| 推理框架 | Throughput | tokens/s |
| :---: | :---: | :---: |
| vllm | 6.34 requests/s | 1216.34 tokens/s |
| hf | 4.03 requests/s | 773.74 tokens/s |
| diff | 57.32% | 57.10% |
