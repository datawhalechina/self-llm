
# 基于vLLM的Yuan 2.0推理服务部署

## 1. 配置vLLM环境
环境要求：torch2.1.2 cuda12.1

vLLM环境配置主要分为以下两步，拉取Yuan-2.0项目，以及安装vllm运行环境

注：由于pip版本 vllm目前还不支持Yuan 2.0，因此需要编译安装

### Step 1. 拉取Yuan-2.0项目

```shell
# 拉取项目
git clone https://github.com/IEIT-Yuan/Yuan-2.0.git
```

### Step 2. 安装vLLM运行环境

```shell
# 进入vLLM项目
cd Yuan-2.0/3rdparty/vllm

# 安装依赖
pip install -r requirements.txt

# 安装setuptools
# vllm对setuptools的版本有要求, 参考 https://github.com/vllm-project/vllm/issues/4961
vim pyproject.toml # 修改为setuptools == 69.5.1
pip install setuptools == 69.5.1

# 安装vllm
pip install -e .
```

## 2. Yuan2.0-2B模型基于vLLM的推理和部署

以下是如何使用vLLM推理框架对Yuan2.0-2B模型进行推理和部署的示例

### Step 1. 模型下载

使用 modelscope 中的 snapshot_download 函数下载模型，第一个参数为模型名称，参数 cache_dir 为模型的下载路径。

这里可以先进入autodl平台，初始化机器对应区域的的文件存储，文件存储路径为'/root/autodl-fs'。
该存储中的文件不会随着机器的关闭而丢失，这样可以避免模型二次下载。

![autodl-fs](images/autodl-fs.png)

然后运行下面代码，执行模型下载。模型大小为 4.5GB，下载大概需要 5 分钟。

```python
from modelscope import snapshot_download
model_dir = snapshot_download('YuanLLM/Yuan2-2B-Mars-hf', cache_dir='/root/autodl-fs')
```

### Step 2. 基于vllm推理Yuan2.0-2B

基于vllm推理Yuan2.0-2B首先需要加载模型，然后进行推理

#### 1. 加载模型

```python
from vllm import LLM, SamplingParams
import time

# 配置参数
sampling_params = SamplingParams(max_tokens=300, temperature=1, top_p=0, top_k=1, min_p=0.0, length_penalty=1.0, repetition_penalty=1.0, stop="<eod>", )

# 加载模型
llm = LLM(model="/root/autodl-fs/YuanLLM/Yuan2-2B-Mars-hf", trust_remote_code=True)
```

#### 2. 推理
推理支持单个prompt和多个prompt

##### Option 1. 单个prompt推理

```python
prompts = ["给我一个python打印helloword的代码<sep>"]

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print("Prompt:", prompt)
    print("Generated text:", generated_text)
    print()

print("inference_time:", (end_time - start_time))
```

##### Option 2. 多个prompt推理

```python
prompts = ["给我一个python打印helloword的代码<sep>", "给我一个c++打印helloword的代码<sep>"]

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print("Prompt:", prompt)
    print("Generated text:", generated_text)
    print()

print("inference_time:", (end_time - start_time))
```

### Step 3. 基于vllm.entrypoints.api_server部署Yuan2.0-2B
基于api_server部署Yuan2.0-2B的步骤包括推理服务的发起和调用

#### 1. 服务发起

```shell 
# 请在命令行运行以下命令，不用直接在jupyter中使用!python运行
python -m vllm.entrypoints.api_server --model=/root/autodl-fs/YuanLLM/Yuan2-2B-Mars-hf --trust-remote-code
```

![](images/05-0.png)

#### 2. 服务调用
服务调用有以下两种方式：第一种是通过命令行直接调用；第二种方式是通过运行脚本批量调用。

##### Option 1. 基于命令行调用服务

```shell
!curl http://localhost:8000/generate -d '{"prompt": "给我一个python打印helloword的代码<sep>", "use_beam_search": false,  "n": 1, "temperature": 1, "top_p": 0, "top_k": 1,  "max_tokens":256, "stop": "<eod>"}'
```

##### Option 2. 基于命令脚本调用服务

```python
import requests
import json

prompt = "给我一个python打印helloword的代码<sep>"
raw_json_data = {
        "prompt": prompt,
        "logprobs": 1,
        "max_tokens": 256,
        "temperature": 1,
        "use_beam_search": False,
        "top_p": 0,
        "top_k": 1,
        "stop": "<eod>",
        }
json_data = json.dumps(raw_json_data)
headers = {
        "Content-Type": "application/json",
        }
response = requests.post(f'http://localhost:8000/generate',
                     data=json_data,
                     headers=headers)
output = response.text
output = json.loads(output)
print(output)
```

### Step 4. 基于vllm.entrypoints.openai.api_server部署Yuan2.0-2B
基于openai的api_server部署Yuan2.0-2B的步骤和step 3的步骤类似，发起服务和调用服务的方式如下：

#### 1. 服务发起

```shell 
# 请在命令行运行以下命令，不用直接在jupyter中使用!python运行
python -m vllm.entrypoints.openai.api_server --model=/root/autodl-fs/YuanLLM/Yuan2-2B-Mars-hf --trust-remote-code
```

![](images/05-1.png)


#### 2. 服务调用
服务调用有以下两种方式：第一种是通过命令行直接调用；第二种方式是通过运行脚本批量调用。

##### Option 1. 基于命令行调用服务

```shell
!curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/root/autodl-fs/YuanLLM/Yuan2-2B-Mars-hf", "prompt": "给我一个python打印helloword的代码<sep>", "max_tokens": 300, "temperature": 1, "top_p": 0, "top_k": 1, "stop": "<eod>"}'
```

##### Option 2. 基于命令脚本调用服务

```python
import requests
import json

prompt = "给我一个python打印helloword的代码<sep>"
raw_json_data = {
        "model": "/root/autodl-fs/YuanLLM/Yuan2-2B-Mars-hf",
        "prompt": prompt,
        "max_tokens": 256,
        "temperature": 1,
        "use_beam_search": False,
        "top_p": 0,
        "top_k": 1,
        "stop": "<eod>",
        }
json_data = json.dumps(raw_json_data, ensure_ascii=True)
headers = {
        "Content-Type": "application/json",
        }
response = requests.post(f'http://localhost:8000/v1/completions',
                     data=json_data,
                     headers=headers)
output = response.text
output = json.loads(output)
print(output)
```