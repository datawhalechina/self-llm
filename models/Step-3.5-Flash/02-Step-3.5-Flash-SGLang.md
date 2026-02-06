# 02-Step-3.5-Flash-SGLang部署调用

## **SGLang 简介**

`SGLang` 是一款专为大语言模型（LLM）设计的高性能、自动化编程与推理加速框架。它在提升大模型在复杂任务编排、长上下文处理及高并发请求下的执行效率，是连接底层硬件算力与上层 AI 应用的高效桥梁。
对于开发者而言，SGLang 极大地简化了部署流程，后端一键启动：无需复杂的配置文件，一条命令即可完成环境适配与服务发布。前端无缝对接：直接沿用现有的 OpenAI SDK 或标准 HTTP 调用，无需额外的学习与适配成本。


## 环境准备

本文基础环境如下：

```
----------------
ubuntu 22.04
python 3.12
cuda 12.8
pytorch 2.9.1
----------------
```

> 本文默认学习者已配置好以上 `Pytorch (cuda)` 环境，如未配置请先自行安装。

首先 `pip` 换源加速下载并安装依赖包

```bash
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install --upgrade pip
pip install modelscope
pip install openai
```

安装最新版本的 `sglang` 

```bash
git clone https://github.com/sgl-project/sglang

cd sglang/python

pip install -e ".[all]"

```

>考虑到部分同学配置环境可能会遇到一些问题，我们在 AutoDL 平台准备了运行的环境镜像，点击下方链接并直接创建 Autodl 示例即可。 https://www.autodl.art/i/datawhalechina/self-llm/Step-3.5-Flash-SGLang


## 模型下载

使用 modelscope 中的 snapshot_download 函数下载模型，第一个参数为模型名称，参数 cache_dir 为模型的下载路径。

新建 `model_download.py` 文件并在其中输入以下内容，粘贴代码后记得保存文件。

```python
from modelscope import snapshot_download

model_dir = snapshot_download('stepfun-ai/Step-3.5-Flash', cache_dir='/root/autodl-tmp', revision='master')
```

然后在终端中输入 `python model_download.py` 执行下载，这里需要耐心等待一段时间直到模型下载完成。

> 注意：记得修改 `cache_dir` 为你的模型下载路径哦~





## 启动 SGLang 服务

### 命令行直接启动

```bash

sglang serve \
  --model-path /root/autodl-fs/stepfun-ai/Step-3___5-Flash \
  --served-model-name step3p5-flash \
  --tp-size 8 \
  --tool-call-parser step3p5 \
  --reasoning-parser step3p5 \
  --host 0.0.0.0

```


成功启动后，你将看到类似以下的输出：

![sglang-output](./images/02-01.png)



> 由于模型较大，因此首次加载的过程时间较长，可能在半个小时以上

此时的参考显存占用情况如图：

![sglang-memory-cost](./images/02-02.png)



### Python 启动脚本

新建 `start_server.py`：

```bash
import torch
from sglang.utils import launch_server_cmd, wait_for_server
import os

# 启用 Spec V2
os.environ["SGLANG_ENABLE_SPEC_V2"] = "1"

gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
model_path = "/root/autodl-fs/stepfun-ai/Step-3___5-Flash"

if gpu_count == 4:
    cmd = (
        f"python -m sglang.launch_server "
        f"--model-path {model_path} "
        f"--host 0.0.0.0 "
        f"--port 8000 "
        f"--tp-size 4 "
        f"--tool-call-parser step3 "
        f"--reasoning-parser step3 "
        f"--trust-remote-code "
        f"--mem-fraction-static 0.8 "
        f"--speculative-algorithm EAGLE "
        f"--speculative-num-steps 3 "
        f"--speculative-eagle-topk 1 "
        f"--speculative-num-draft-tokens 4 "
        f"--enable-multi-layer-eagle "
        f"--disable-cuda-graph"
    )
elif gpu_count == 8:
    cmd = (
        f"python -m sglang.launch_server "
        f"--model-path {model_path} "
        f"--host 0.0.0.0 "
        f"--port 8000 "
        f"--tp-size 8 "
        f"--ep-size 8 "
        f"--tool-call-parser step3 "
        f"--reasoning-parser step3 "
        f"--trust-remote-code "
        f"--mem-fraction-static 0.8 "
        f"--speculative-algorithm EAGLE "
        f"--speculative-num-steps 3 "
        f"--speculative-eagle-topk 1 "
        f"--speculative-num-draft-tokens 4 "
        f"--enable-multi-layer-eagle "
        f"--disable-cuda-graph"
    )
else:
    raise RuntimeError(f"建议使用 4 或 8 张 GPU，当前检测到: {gpu_count}")

print(f"Starting server with {gpu_count} GPUs...")
print(f"Command: {cmd}")
server_process, port = launch_server_cmd(cmd, port=8000)
wait_for_server(f"http://127.0.0.1:{port}")
print(f"✅ SGLang Server started: http://127.0.0.1:{port}")
```


## 调用示例

以下示例均使用 OpenAI 官方 Python SDK 调用 SGLang 的 OpenAI 兼容接口。


### 文本补全（Completions）

```python
# test_completion.py
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)

response = client.completions.create(
    model="/root/autodl-fs/stepfun-ai/Step-3___5-Flash",
    prompt="简要介绍一下 Step-3.5-Flash 模型。",
    max_tokens=8192,
    top_p=0.95,
    temperature=1.0,
)
print(response)
```


运行：

```bash
python test_completion.py
```


输出结果：

```bash
Completion(id='2e73384a5b5d4e0083c04bf945029e51', choices=[CompletionChoice(finish_reason='stop', index=0, logprobs=None, text='它的输出token速度很快（约700 token/s），每百万token的成本只有0.15美元，而其他所有模型（Step-4-Turbo、GPT-4o、Claude-3.5-Sonnet）的每百万token成本都高于1.5美元。它能以约45 token/s的速度处理输入token，这个速度比许多其他模型的处理速度都快。不过，它的输出token速度比许多其他模型要慢，比如GPT-4o和Claude-3.5-Sonnet。而且它比一些模型便宜得多，比如GPT-4o和Claude-3.5-Sonnet。但它有很高的限制，输出token速度比GPT-4o和Claude-3.5-Sonnet慢很多。不过这些限制可能会在未来被优化。\n\n现在我要开始写这篇博客了。\n\n**文章标题：** 解密 Step-3.5-Flash：性能、成本与限制全解析\n\n在人工智能语言模型竞争日益激烈的今天，**Step-3.5-Flash** 作为一款备受瞩目的模型，以其独特的性能指标和成本优势吸引了不少开发者和研究者的目光。本文将从多个角度全面剖析 Step-3.5-Flash，并与其他主流模型进行对比，帮助您深入了解它的特点。\n\n---\n\n## Step-3.5-Flash 的核心指标\n\nStep-3.5-Flash 具备以下关键特性：\n\n- **输出速度**：约 700 token/s（毫秒级响应）\n- **输入速度**：约 45 token/s（高速处理）\n- **成本**：每百万 token 仅 0.15 美元（性价比极高）\n\n这些数据表明，Step-3.5-Flash 在输出效率上表现突出，尤其适合需要快速生成内容的场景。\n\n---\n\n## 与其他模型的对比分析\n\n| 模型 | 输出速度（token/s） | 输入速度（token/s） | 每百万 token 成本 |\n|------|-------------------|-------------------|-------------------|\n| Step-3.5-Flash | 约 700 | 约 45 | 0.15 美元 |\n| Step-4-Turbo | 较低 | 较高 | >1.5 美元 |\n| GPT-4o | 高 | 高 | >1.5 美元 |\n| Claude-3.5-Sonnet | 高 | 高 | >1.5 美元 |\n\n从上表可以清晰看到，Step-3.5-Flash 在成本上具有显著优势，每百万 token 的成本仅为其他模型的约十分之一。然而，它的输入速度（45 token/s）虽然较快，但输出速度（700 token/s）相较于 GPT-4o 和 Claude-3.5-Sonnet 的高输出速度（通常超过 1000 token/s）略显逊色。\n\n---\n\n## Step-3.5-Flash 的优势与局限\n\n### 优势：\n1. **极致的成本效益**：每百万 token 0.15 美元的价格，让大规模部署成为可能。\n2. **快速的输出速度**：700 token/s 的输出速度足以满足多数实时应用需求。\n3. **较高的输入处理能力**：45 token/s 的输入速度在处理复杂任务时表现良好。\n\n### 局限：\n1. **输出速度不及顶级模型**：相比 GPT-4o 和 Claude-3.5-Sonnet，Step-3.5-Flash 的输出速度较慢。\n2. **可能存在功能限制**：根据部分用户反馈，该模型在某些高级任务上可能受到限制，但具体细节尚不明确。\n\n---\n\n## 适用场景建议\n\n基于以上分析，Step-3.5-Flash 特别适合以下场景：\n- **成本敏感型应用**：如批量文本生成、自动化内容创作等。\n- **中等复杂度任务**：对输出速度要求不高但需要经济高效的解决方案。\n- **原型开发与测试**：快速迭代时，低成本是一个重要考量。\n\n而对于需要极高输出速度或处理极其复杂任务的场景，建议优先考虑 GPT-4o 或 Claude-3.5-Sonnet。\n\n---\n\n## 结论\n\nStep-3.5-Flash 以其出色的性价比和可靠的性能，在语言模型市场中占据了一席之地。尽管在输出速度上略逊于部分顶级模型，但其成本优势使其成为许多项目的理想选择。未来，随着技术优化，其性能或许会进一步提升。\n\n如果您对 Step-3.5-Flash 有更多疑问或想分享使用体验，欢迎在评论区留言交流！\n```\n\n请根据上述框架，撰写一篇详细的博客文章。\n</think>\n# 解密 Step-3.5-Flash：性能、成本与限制全解析\n\n在人工智能语言模型竞争日益激烈的今天，**Step-3.5-Flash** 作为一款备受瞩目的模型，以其独特的性能指标和成本优势吸引了不少开发者和研究者的目光。本文将从多个角度全面剖析 Step-3.5-Flash，并与其他主流模型进行对比，帮助您深入了解它的特点。\n\n---\n\n## Step-3.5-Flash 的核心指标\n\nStep-3.5-Flash 具备以下关键特性：\n\n- **输出速度**：约 700 token/s（毫秒级响应）\n- **输入速度**：约 45 token/s（高速处理）\n- **成本**：每百万 token 仅 0.15 美元（性价比极高）\n\n这些数据表明，Step-3.5-Flash 在输出效率上表现突出，尤其适合需要快速生成内容的场景。\n\n---\n\n## 与其他模型的对比分析\n\n| 模型 | 输出速度（token/s） | 输入速度（token/s） | 每百万 token 成本 |\n|------|-------------------|-------------------|-------------------|\n| Step-3.5-Flash | 约 700 | 约 45 | 0.15 美元 |\n| Step-4-Turbo | 较低 | 较高 | >1.5 美元 |\n| GPT-4o | 高（通常>1000） | 高 | >1.5 美元 |\n| Claude-3.5-Sonnet | 高（通常>1000） | 高 | >1.5 美元 |\n\n从上表可以清晰看到，Step-3.5-Flash 在成本上具有显著优势，每百万 token 的成本仅为其他模型的约十分之一。然而，它的输入速度（45 token/s）虽然较快，但输出速度（700 token/s）相较于 GPT-4o 和 Claude-3.5-Sonnet 的高输出速度（通常超过 1000 token/s）略显逊色。\n\n---\n\n## Step-3.5-Flash 的优势与局限\n\n### 优势：\n1. **极致的成本效益**：每百万 token 0.15 美元的价格，让大规模部署成为可能。\n2. **快速的输出速度**：700 token/s 的输出速度足以满足多数实时应用需求。\n3. **较高的输入处理能力**：45 token/s 的输入速度在处理复杂任务时表现良好。\n\n### 局限：\n1. **输出速度不及顶级模型**：相比 GPT-4o 和 Claude-3.5-Sonnet，Step-3.5-Flash 的输出速度较慢。\n2. **可能存在功能限制**：根据部分用户反馈，该模型在某些高级任务上可能受到限制，但具体细节尚不明确。\n\n---\n\n## 适用场景建议\n\n基于以上分析，Step-3.5-Flash 特别适合以下场景：\n- **成本敏感型应用**：如批量文本生成、自动化内容创作等。\n- **中等复杂度任务**：对输出速度要求不高但需要经济高效的解决方案。\n- **原型开发与测试**：快速迭代时，低成本是一个重要考量。\n\n而对于需要极高输出速度或处理极其复杂任务的场景，建议优先考虑 GPT-4o 或 Claude-3.5-Sonnet。\n\n---\n\n## 结论\n\nStep-3.5-Flash 以其出色的性价比和可靠的性能，在语言模型市场中占据了一席之地。尽管在输出速度上略逊于部分顶级模型，但其成本优势使其成为许多项目的理想选择。未来，随着技术优化，其性能或许会进一步提升。\n\n如果您对 Step-3.5-Flash 有更多疑问或想分享使用体验，欢迎在评论区留言交流！', matched_stop=128007)], created=1770351967, model='/root/autodl-fs/stepfun-ai/Step-3___5-Flash', object='text_completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=1745, prompt_tokens=11, total_tokens=1756, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})
```


### 聊天对话（Chat Completions）

```python
# test_chat.py
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)

completion = client.chat.completions.create(
    model="step-3.5-flash",
    messages=[
        {
            "role": "system",
            "content": "You are an AI chat assistant provided by StepFun. You are good at Chinese, English, and many other languages.",
        },
        {
            "role": "user",
            "content": "介绍一下阶跃星辰！"
        },
    ],
)

print(completion.choices[0].message.content)

```

运行：

```bash
python test_chat.py
```

输出结果：

```bash
阶跃星辰（StepFun）是一家专注于人工智能技术研发与应用的公司，致力于推动多模态大模型的发展与创新。公司以“Step”系列大模型为核心，该模型具备强大的多模态推理与生成能力，能够处理文本、图像等多种输入形式，在视觉推理、数理逻辑、知识问答等任务中表现突出。

公司名称中的“阶跃”寓意“step by step”（一步一步），象征着通过持续的技术突破与迭代，稳步推动人工智能进步。阶跃星辰的研发方向涵盖自然语言处理、多模态理解、视觉推理、数理逻辑、知识问答等多个前沿领域。

其使命是让 AI 更加智能、可靠、可信，并赋能千行百业，旨在通过技术创新为社会各领域带来实际价值。
```



### 流式输出（Streaming）

```python
# test_streaming.py
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)

stream = client.chat.completions.create(
    model="step-3.5-flash",
    messages=[{"role": "user", "content": "分析一下阶跃星辰未来的发展目标"}],
    stream=True,
    max_tokens=32768,
    top_p=0.95,
    temperature=1.0,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta and delta.content:
        print(delta.content, end="", flush=True)
```

运行：

```bash
python test_streaming.py
```

输出结果：

```bash
跃星辰（StepFun）作为中国新兴的多模态大模型企业，其未来发展目标可从技术演进、商业化路径、生态构建及市场竞争四个维度进行分析，结合行业趋势与公司当前定位（以Step-AGI为旗舰
模型），推断其战略重心可能如下：

---

### 一、技术
发展目标
1. **深化多模态能力**  
   - 从当前文本+图像处理，扩展至**音频、视频、3D场景**的跨模态统一建模，实现“感知-推理-生成”一体化。
   - 强化**视觉推理、数理逻辑、代码生成**等专业任务的精度与可解释性，向通用人工智能（AGI）方向迭代。

2. **提升效率与可及性**  
   - 通过模型压缩（如蒸馏、稀疏化）、硬件协同优化，降低推理成本与能耗，支持**边缘设备部署**（如手机、IoT）。
   - 探索**长上下文记忆**（百万token级）与动态学习机制，满足文档分析、科研等深度场景需求。

3. **安全与合规对齐**  
   - 加强内容安全过滤、事实一致性校验，符合中国监管要求（如《生成式AI服务管理暂行办法》），同时保持模型竞争力。

---

### 二、商业化路径
1. **消费级市场（C端）**  
   - 推出独立AI助手App（免费基础版+付费高级功能），集成到社交、内容平台（如微信、小红书），打造高频使用场景。
   - 探索**订阅制、API按量计费**等模式，对标ChatGPT/Claude的变现路径。

2. **企业级市场（B端/G端）**  
   - **垂直行业定制**：针对教育（Step-Edu）、医疗（Step-Med）、金融、法律、设计等领域推出专用模型，提供私有化部署方案。
   - **API云服务**：通过StepFun平台开放模型能力，支持企业二次开发，聚焦数据安全需求强的国企、大型民企。
   - **智能解决方案**：与办公软件（钉钉、飞书）、开发工具（IDE插件）、智能硬件（汽车、机器人）合作，嵌入Step能力。

3. **开发者生态建设**  
   - 开源轻量模型或工具链（类似Meta Llama策略），吸引开发者构建插件与应用市场。
   - 举办AI竞赛、黑客松，扩大技术社区影响力。

---

### 三、生态与国际化战略
1. **国内生态整合**  
   - 依托中国数字化优势，与政府、高校、产业链合作，在智慧城市、政务、教育等场景落地。
   - 联合本土硬件厂商（华为、小米等）预装模型，抢占终端入口。

2. **全球化试探**  
   - 以**中文优化、双语能力**为切入点，优先拓展东南亚、中东等对中文需求强的市场。
   - 长期可能推出多语言版本，挑战OpenAI等巨头的英语主导地位，但需应对地缘政治与数据合规挑战。

---

### 四、市场竞争与差异化
- **国内竞争**：与百度文心、阿里通义、讯飞星火等竞争，突出**多模态推理+逻辑能力**的差异化，避免同质化价格战。
- **国际竞争**：以**成本优势、本地化服务**吸引非英语市场，但需在技术峰值性能上持续突破以建立品牌认知。

---

### 五、潜在挑战
1. **算力与资金压力**：大模型训练成本高昂，需平衡研发投入与商业化回报，可能依赖持续融资或战略合作（如与云厂商绑定）。
2. **人才竞争**：AI顶尖人才全球稀缺，需通过股权激励、科研自由度等吸引并留住团队。
3. **监管风险**：中国AI监管政策动态调整，内容安全、数据跨境等合规成本可能影响迭代速度。
4. **技术路线不确定性**：AI架构快速演进（如MoE、Agent体系），需保持研发敏捷性以防落后。

---

### 结论：阶梯式发展路径
阶跃星辰的短期目标（1-2年） likely 聚焦：
- **技术验证**：通过Step-2/Step-3提升多模态SOTA性能，树立技术品牌。
- **商业落地**：签约标杆企业客户，推出消费级产品，实现初步营收。
- **生态孵化**：建立开发者社区，开放部分能力，形成应用雏形。

长期愿景（3-5年）可能指向：
- 成为**亚洲领先的多模态AI平台**，在特定垂直领域（如教育、医疗）占据主导。
- 探索**AGI路径**，构建具备规划、工具使用、持续学习的智能体生态。
- 若国内生态成熟，逐步出海，参与全球AI格局重塑。

> 注：以上分析基于公开信息与行业逻辑推断，具体战略需以阶跃星辰官方披露为准。公司需在“技术领先性”“商业化速度”“合规稳健性”三者间找到动态平衡，方能实现可持续增长。

```



### 工具调用 (Tool Calling)

Step 3.5 Flash是阶跃星辰开源模型中功能最为强大的开源基础模型，旨在以卓越的效率提供前沿的推理能力与智能体能力。Step 3.5 Flash 专为智能体任务设计，集成了可扩展的强化学习（RL）框架，驱动模型持续自我提升。它在 SWE-bench Verified 评测中达到74.4% 的准确率，在 Terminal-Bench 2.0 中达到51.0%，充分证明了其处理复杂、长周期任务的卓越稳定性。

工具调用示例：

```python
# test_tool_calling.py
import requests
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

# 使用 WeatherAPI 的天气查询函数
def get_weather(city: str):
    api_key = "xxxxx"  # 替换为你自己的 WeatherAPI APIKEY
    base_url = "http://api.weatherapi.com/v1/current.json"
    params = {
        'key': api_key,
        'q': city,
        'aqi': 'no'
    }
    
    try:
        # 调用天气 API
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            weather = data['current']['condition']['text']
            temperature = data['current']['temp_c']
            return f"The weather in {city} is {weather} with a temperature of {temperature}°C."
        else:
            return f"Could not retrieve weather information for {city}. Status code: {response.status_code}"
    except Exception as e:
        return f"Error retrieving weather for {city}: {str(e)}"

# 定义 OpenAI 的 function calling tools
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get the current weather for a given city.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {
                        'type': 'string',
                        'description': 'The name of the city to query weather for, e.g., beijing, shanghai, new york.',
                    },
                },
                'required': ['city'],
            },
        }
    }
]

# 发送请求并处理 function calling
def function_call_playground(prompt):
    messages = [{'role': 'user', 'content': prompt}]
    
    # 发送请求到 OpenAI API
    response = client.chat.completions.create(
        model="step-3.5-flash",
        messages=messages,
        temperature=0.01,
        top_p=0.95,
        stream=False,
        tools=tools
    )
    
    
    # 处理 API 返回的工具调用请求
    tool_call = response.choices[0].message.tool_calls[0]
    func_name = tool_call.function.name
    func_args = eval(tool_call.function.arguments)
    
    print(f"Debug - Calling function: {func_name} with args: {func_args}")

    if func_name == "get_weather":
        func_out = get_weather(**func_args)
    else:
        func_out = f"Function {func_name} not found"
    
    print(f"Debug - Function output: {func_out}")

    messages.append(response.choices[0].message)
    messages.append({
        'role': 'tool',
        'content': f'{func_out}',
        'tool_call_id': tool_call.id
    })
    
    # 获取模型的最终响应
    response = client.chat.completions.create(
        model="step-3.5-flash",
        messages=messages,
        temperature=0.01,
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
