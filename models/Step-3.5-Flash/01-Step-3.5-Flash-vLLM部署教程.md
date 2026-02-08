# Step-3.5-Flash vLLM 部署调用及 Docker 镜像

## vLLM 简介

`vLLM` 框架是一个高效的大语言模型**推理和部署服务系统**，具备以下特性：

- **高效的内存管理**：通过 `PagedAttention` 算法，`vLLM` 实现了对 `KV` 缓存的高效管理，减少了内存浪费，优化了模型的运行效率。
- **高吞吐量**：`vLLM` 支持异步处理和连续批处理请求，显著提高了模型推理的吞吐量，加速了文本生成和处理速度。
- **易用性**：`vLLM` 与 `HuggingFace` 模型无缝集成，支持多种流行的大型语言模型，简化了模型部署和推理的过程。兼容 `OpenAI` 的 `API` 服务器。
- **分布式推理**：框架支持在多 `GPU` 环境中进行分布式推理，通过模型并行策略和高效的数据通信，提升了处理大型模型的能力。
- **开源共享**：`vLLM` 由于其开源的属性，拥有活跃的社区支持，这也便于开发者贡献和改进，共同推动技术发展。

## 环境准备

本文基础环境如下：

```
----------------
ubuntu 22.04
python 3.12
cuda 12.4
pytorch 2.5.1
----------------
```

> 本文默认学习者已配置好以上 `Pytorch (cuda)` 环境，如未配置请先自行安装。

首先 `pip` 换源加速下载并安装依赖包

```bash
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

> 考虑到部分同学配置环境可能会遇到一些问题，我们在 AutoDL 平台准备了 Step-3.5-Flash 的环境镜像，点击下方链接并直接创建 Autodl 示例即可。https://www.autodl.art/i/datawhalechina/self-llm/step-3.5-flash-vllm
> ******

## 模型下载

使用 `modelscope` 中的 `snapshot_download` 函数下载模型，第一个参数为模型名称，参数 `cache_dir` 为模型的下载路径。

新建 `model_download.py` 文件并在其中输入以下内容，粘贴代码后记得保存文件。

```python
# model_download.py
from modelscope import snapshot_download

model_dir = snapshot_download('stepfun-ai/Step-3.5-Flash', cache_dir='/root/autodl-tmp', revision='master')
print(f"模型下载完成，保存路径为：{model_dir}")
```

然后在终端中输入 `python model_download.py` 执行下载，这里需要耐心等待一段时间直到模型下载完成。

> 注意：记得修改 `cache_dir` 为你的模型下载路径哦~

## 模型简介

`Step-3.5-Flash` 是阶跃星辰（StepFun）推出的新一代**稀疏混合专家（Sparse MoE）**大语言模型，专为极致效率与智能体应用打造。其核心特性包括：

- **极致推理速度：** 引入创新的 **MTP-3（多Token预测）** 技术，大幅突破传统Transformer的生成瓶颈，推理吞吐量显著提升，实现毫秒级极速响应。
- **Agent原生智能：** 具备卓越的“Think-and-Act”协同能力，在复杂逻辑推理、多步任务规划及工具调用上表现稳定，专为构建高性能智能体而生。
- **高效长程记忆：** 支持 **256k 超长上下文**，结合滑动窗口注意力机制，在处理海量文档与长代码库时，兼顾精准度与计算效率。
- **全能应用生态：** 凭借1960亿参数（激活约110亿）的架构优势，在数理逻辑、代码生成及跨模态任务编排中展现出媲美闭源顶尖模型的实力。



## Transformers 推理（验证模型）

在使用 vLLM 部署之前，我们先用 Transformers 验证模型能否正常推理。新建 `transformers_inference.py` 文件：

```python
# transformers_inference.py
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型路径：修改为你的本地路径
MODEL_PATH = "/root/autodl-tmp/stepfun-ai/Step-3.5-Flash"

# 1) 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
)

# 2) 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    local_files_only=True,
)

# 3) 构造对话消息
messages = [{"role": "user", "content": "Explain the significance of the number 42."}]

# 4) 应用 chat 模板
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# 5) 生成文本
generated_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

# 6) 解码输出
gen_ids = generated_ids[0][inputs.input_ids.shape[1]:]
output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
print(output_text)
```

运行代码：

```bash
python transformers_inference.py
```

输出示例：

```
Hmm, the user is asking about the significance of the number 42...
```

## vLLM 服务部署

### 安装 vLLM

我们建议使用 vLLM 的最新 nightly 版本以获得最佳性能。

```bash
# 通过 pip 安装（nightly 版本）
pip install -U vllm --pre \
  --index-url https://pypi.org/simple \
  --extra-index-url https://wheels.vllm.ai/nightly
```

### 启动 vLLM 服务

vLLM 提供了兼容 OpenAI API 的服务接口，可以通过 `vllm serve` 命令快速启动。

> **注意**：vLLM 尚未完全支持 MTP3。我们正在积极开发 Pull Request 以集成此功能，预计将显著提升解码性能。

成功启动后，你将看到类似以下的输出：

![01-01](images\01-01.png)

#### FP8 模型启动

```bash
vllm serve /root/autodl-tmp/stepfun-ai/Step-3.5-Flash \
  --served-model-name step3p5-flash \
  --tensor-parallel-size 8 \
  --disable-cascade-attn \
  --reasoning-parser step3p5 \
  --enable-auto-tool-choice \
  --tool-call-parser step3p5 \
  --trust-remote-code \
  --quantization fp8
```

#### BF16 模型启动

```bash
vllm serve /root/autodl-tmp/stepfun-ai/Step-3.5-Flash \
  --served-model-name step3p5-flash \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --disable-cascade-attn \
  --reasoning-parser step3p5 \
  --enable-auto-tool-choice \
  --tool-call-parser step3p5 \
  --hf-overrides '{"num_nextn_predict_layers": 1}' \
  --speculative_config '{"method": "step3p5_mtp", "num_speculative_tokens": 1}' \
  --trust-remote-code
```

服务启动成功后将监听 `http://0.0.0.0:8000/v1`。

成功启动后，你将看到类似以下的输出：

![01-02](images\01-02.png)

> 由于模型较大，因此首次加载的过程时间较长，可能在半个小时以上。

此时的参考显存占用情况如图：

![01-03](images\01-03.jpg)
![01-04](images\01-04.jpg)
### 参数说明

- `--served-model-name`：对外暴露的模型名称，客户端调用时需使用此名称
- `--tensor-parallel-size`：张量并行大小，通常设置为 GPU 数量
- `--enable-expert-parallel`：启用专家并行（适用于 MoE 模型）
- `--disable-cascade-attn`：禁用级联注意力机制
- `--reasoning-parser`：推理内容解析器，用于解析模型的思考过程
- `--enable-auto-tool-choice`：启用自动工具选择
- `--tool-call-parser`：工具调用解析器
- `--trust-remote-code`：允许加载自定义代码（必需）
- `--quantization`：量化方式（如 fp8）
- `--hf-overrides`：覆盖 HuggingFace 配置参数
- `--speculative_config`：推测解码配置

## API 调用示例

### 使用 requests 库调用

新建 `test_requests.py` 文件：

```python
# test_requests.py
import requests

url = "http://0.0.0.0:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

data = {
    "messages": [
        {"role": "user", "content": "Explain the significance of the number 42."}
    ],
    "model": "step3p5-flash"
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    result = response.json()
    print("Response:", result['choices'][0]['message']['content'])
else:
    print("Error:", response.status_code, response.text)
```

运行代码：

```bash
python test_requests.py
```

输出示例（部分截取）：

```json
{
  "id": "chatcmpl-9f51bb9294e6712a",
  "object": "chat.completion",
  "created": 1770390777,
  "model": "step3p5-flash",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The number **42** is most famously known as the **\"Answer to the Ultimate Question of Life, the Universe, and Everything\"** in Douglas Adams' beloved sci-fi series *The Hitchhiker's Guide to the Galaxy*. Its significance is both a hilarious absurdist joke and a cultural phenomenon...",
        "reasoning_content": "Hmm, the user is asking about the significance of the number 42. This is a classic pop culture reference, so they're likely expecting an explanation tied to Douglas Adams' *The Hitchhiker's Guide to the Galaxy*..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 21,
    "total_tokens": 982,
    "completion_tokens": 961
  }
}
```

> 注意：响应中包含 `reasoning_content` 字段，展示了模型的思考过程。

### 使用 OpenAI SDK 调用（Chat Completions）

新建 `test_chat.py` 文件：

```python
# test_chat.py
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)

response = client.chat.completions.create(
    model="step3p5-flash",
    messages=[
        {"role": "user", "content": "Explain the significance of the number 42."}
    ]
)

print("Response:", response.choices[0].message.content)
print("\nReasoning:", response.choices[0].message.reasoning_content)
```

运行代码：

```bash
python test_chat.py
```

输出示例：

```
Response: The number **42** is most famously known as the **"Answer to the Ultimate Question of Life, the Universe, and Everything"** in Douglas Adams' beloved sci-fi series *The Hitchhiker's Guide to the Galaxy*...

Reasoning: Hmm, the user is asking about the significance of the number 42. This is a classic pop culture reference, so they're likely expecting an explanation tied to Douglas Adams' *The Hitchhiker's Guide to the Galaxy*...
```

另外，在以上所有的请求处理过程中，`API` 后端都会打印相对应的日志和统计信息。

### 使用 OpenAI SDK 调用（Completions）

新建 `test_completion.py` 文件：

```python
# test_completion.py
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)

response = client.completions.create(
    model="step3p5-flash",
    prompt="简要介绍一下 Step-3.5-Flash 模型。",
    max_tokens=500,
    top_p=0.95,
    temperature=0.2,
)

response_text = response.choices[0].text
print(response_text)
```

运行代码：

```bash
python test_completion.py
```

输出示例：

```
Step-3.5-Flash 是阶跃星辰（StepFun）推出的大语言模型，具备多模态推理能力，能够处理文本、图片等多种输入形式。相比之前的 Step 模型，Step-3.5-Flash 在响应速度、推理能力和多模态支持方面都有显著提升...
```

### 工具调用 (Tool Calling)

Step 3.5 Flash是阶跃星辰开源模型中功能最为强大的开源基础模型，旨在以卓越的效率提供前沿的推理能力与智能体能力。Step 3.5 Flash 专为智能体任务设计，集成了可扩展的强化学习（RL）框架，驱动模型持续自我提升。它在 SWE-bench Verified 评测中达到74.4% 的准确率，在 Terminal-Bench 2.0 中达到51.0%，充分证明了其处理复杂、长周期任务的卓越稳定性。

工具调用示例：

```python
# test_completion.py
import json
import requests
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)

# 使用 WeatherAPI 的天气查询函数
def get_weather(city: str):
    api_key = ""  # 替换为你自己的 WeatherAPI APIKEY
    base_url = "http://api.weatherapi.com/v1/current.json"
    params = {
        "key": api_key,
        "q": city,
        "aqi": "no"
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            weather = data["current"]["condition"]["text"]
            temperature = data["current"]["temp_c"]
            return f"The weather in {city} is {weather} with a temperature of {temperature}°C."
        return f"Could not retrieve weather information for {city}. Status code: {response.status_code}"
    except Exception as e:
        return f"Error retrieving weather for {city}: {str(e)}"

# 定义 OpenAI 的 function calling tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to query weather for, e.g., beijing, shanghai, new york.",
                    },
                },
                "required": ["city"],
            },
        }
    }
]

# 发送请求并处理 function calling
def function_call_playground(prompt):
    messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model="step3p5-flash",
        messages=messages,
        temperature=0.01,
        top_p=0.95,
        stream=False,
        tools=tools
    )

    tool_call = response.choices[0].message.tool_calls[0]
    func_name = tool_call.function.name
    func_args = json.loads(tool_call.function.arguments)

    print(f"Debug - Calling function: {func_name} with args: {func_args}")

    if func_name == "get_weather":
        func_out = get_weather(**func_args)
    else:
        func_out = f"Function {func_name} not found"

    print(f"Debug - Function output: {func_out}")

    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "content": f"{func_out}",
        "tool_call_id": tool_call.id
    })

    response = client.chat.completions.create(
        model="step3p5-flash",
        messages=messages,
        temperature=0.5,
        top_p=0.95,
        stream=False,
        tools=tools
    )

    return response.choices[0].message.content

# 示例使用
prompts = [
    "what's the weather like in shanghai?",
    "北京天气怎么样？",
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"Question: {prompt}")
    print(f"Answer: {function_call_playground(prompt)}")
    print(f"{'='*60}")
```

运行：

```bash
python test_tool_calling.py
```

输出结果：

```bash
============================================================
Question: what's the weather like in shanghai?
Debug - Calling function: get_weather with args: {'city': 'shanghai'}
Debug - Function output: The weather in shanghai is Partly cloudy with a temperature of 6.0°C.
Answer: The weather in Shanghai is partly cloudy with a temperature of 22°C (72°F). The humidity is 65%, and there is a light breeze at 10 km/h.
============================================================

============================================================
Question: 北京天气怎么样？
Debug - Calling function: get_weather with args: {'city': 'beijing'}
Debug - Function output: The weather in beijing is Overcast with a temperature of -5.7°C.
Answer: 北京的天气是多云转阴。
============================================================
```
