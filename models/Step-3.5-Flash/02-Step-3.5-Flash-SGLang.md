# 3-Step-3.5-Flash-SGLang部署调用

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
  --host 0.0.0.0 \

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
    prompt="简要介绍一下 Step-3.5-Flash 模型的特点。",
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
Completion(id='e60dfa22288b4ae2ab49c36a956ee74c', choices=[CompletionChoice(finish_reason='stop', index=0, logprobs=None, text='一粒花生 \n发表于 2025-02-06 2025-02-06T10:26:16+08:00 更新于 2025-02-06 2025-02-06T10:33:38+08:00 12 分钟读完 (大约 1802 个字)\n\n自 2025 年 1 月份起，我就一直在思考如何优化 OpenAI 模型的 API 调用。我遇到的主要问题是响应速度慢且费用高昂。过去一年，我在 OpenAI 上花费了超过 1000 美元，尤其是用于创建文字图片，但实际使用量并不高。\n\n考虑到这一痛点，我开始寻找替代方案。在上一篇文章中，我对比了 Azure OpenAI 和 OpenAI 官方 API，发现官方 API 更实惠。然而，我对非官方 API 的安全性和合规性有所顾虑。虽然可以通过设置 IP 白名单来缓解，但这对个人开发者而言并不实际。更重要的是，我担心随意切换 API 会违反 Azure 的使用条款，因此最终选择了官方 API。\n\n当我深入研究 OpenAI 官方 API 时，我发现自己的主要使用场景是文本处理和总结，对推理需求不高。于是，我尝试使用更便宜的 GPT-3.5-turbo 模型来总结内容。结果有些令人失望——总结质量一般，感觉像是将文本压缩成要点，逻辑性不强。虽然可读，但远不如 GPT-4 的结论那样具有洞察力。这可能是因为 GPT-3.5-turbo 并非专门为总结任务优化。\n\n就在这时，Gemini 2.0 Flash 进入我的视野。它的总结能力更强，输出更接近人工水平，且价格仅为 GPT-3.5-turbo 的一半！而且，Gemini 2.0 Flash 在保持高质量的同时，速度也很快。我立刻决定将其作为我的主要模型。\n\n然而，我没想到的是，仅仅两天后的 2025 年 2 月 4 日，OpenAI 发布了 Step-3.5-Flash。当然，DeepSeek 的热度也瞬间吸引了我的注意。我在第一时间测试了 DeepSeek R1，结果出奇得好！生成的总结结构清晰、逻辑性强，几乎达到 GPT-4 的水平，费用只有 GPT-4 的 1/10。我觉得这简直完美，立刻决定全面转向 DeepSeek。但问题也随之而来：DeepSeek 的响应速度太慢，每次 API 调用都要等待 10 到 20 秒，甚至更长。考虑到用户需要即时看到结果，这样的延迟是不可接受的。\n\n这让我陷入了两难：Gemini 2.0 Flash 速度快、费用低，但总结质量一般；DeepSeek R1 总结质量高、费用极低，但速度太慢。理想中的方案应该是既能保持高质量总结，又能快速响应。就在这时，我又看到了希望——OpenAI 发布的 Step-3.5-Flash 模型。根据官方描述，这个模型结合了 DeepSeek R1 的逻辑能力和 GPT-3.5 的速度，而且价格与 GPT-3.5 相同！\n\n官网介绍：\n> Step 是一种推理模型，专门设计用于解决问题，其方法是在生成最终答案之前“思考”。这使 Step 能够处理更复杂的查询，这些查询要求多个推理步骤、逻辑思考和规划。然而，与标准模型（如 gpt-4o 或 gpt-4o-mini）不同，推理模型消耗大量计算资源来生成思考过程，这会导致更高的延迟。这就是为什么推理模型通常比标准模型慢得多。\n\n> 推理模型的典型用例包括编码、数学、分析、科学和策略等领域。\n\n> step-c 不是推理模型。它不执行深入的推理链，类似于 o3-mini 等推理模型。它适用于直接回答问题、创意写作和一般任务等场景，不涉及复杂的多步推理。\n\n在官方文档中，我找到了更详细的对比说明：\n> 标准的生成模型（如 GPT-4o 或 GPT-4o-mini）接收用户输入并直接生成响应。推理模型（如 o1 系列、o3-mini、DeepSeek R1 等）首先执行深思熟虑的内部思考过程，然后生成最终输出。这使推理模型在解决复杂任务时更加强大，但也显著增加了延迟，因为内部思考过程需要额外的计算。此外，思考过程的输出并不包含在最终输出中，用户看不到它。这意味着在最终输出之前存在额外的延迟，而标准模型没有这一层延迟。\n\n> 推理模型通常用于学术、STEM 研究、编码、数学或复杂分析。它们不适合直接回答简单问题、创意写作或聊天等通用任务。对于这些任务，标准模型通常更合适。不过，推理模型也可以针对非推理任务进行微调，但预计推理模型在所有任务上的表现和延迟都无法与标准模型相媲美。\n\n我的结论是，DeepSeek R1 虽然是推理模型，但开销大、延迟高，适合研究场景；而 Step 作为编码辅助模型，推理能力可能稍弱，但更像 GPT-4o 这样的通用模型，延迟更低。根据体验，虽然 Step-3.5-Flash 是推理模型，但相比 DeepSeek R1，它延迟更低（约 2~3 秒，DeepSeek 超过 10 秒），费用相当，适合我的实时总结场景。我决定在后续项目中尝试使用 Step-3.5-Flash，并会在实际使用后分享更多体验和测试结果。\n参考链接\n  https://openai.com/index/introducing-step-and-step-mini/ \n  https://platform.openai.com/docs/models/step-and-step-mini \n  https://help.openai.com/en/articles/11933446-deepseek-r1-and-deepseek-r1-zero-in-the-openai-api \n  https://platform.openai.com/docs/guides/reasoning \n  https://help.openai.com/en/articles/11933446-deepseek-r1-and-deepseek-r1-zero-in-the-openai-api \n  https://help.openai.com/en/articles/10127605-reasoning-models \n  https://cdn.openai.com/deepseek-r1-faq.pdf \n  https://help.openai.com/en/articles/10322642-understanding-reasoning-models          豆包AI 字节跳动 Step-3.5-Flash OpenAI 科技 api       \nOSPO Bootcamp China\n\nOSPO Bootcamp China（中国开源负责人培训班），致力于在中国推广开源办公室（OSPO）模式，促进企业参与和贡献开源。通过培训班活动，我们希望帮助更多企业了解开源的价值，并推动开源生态系统的健康发展。\n   OSPO Bootcamp China \n 赞助\n  2024年活动 \n  联系 \n  Cookie 偏好\n\n© 2024 OSPO Bootcamp China\n\nPowered by Hugo with theme\n    LoveIt .', matched_stop=1)], created=1770191980, model='/root/autodl-fs/stepfun-ai/Step-3___5-Flash', object='text_completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=1440, prompt_tokens=14, total_tokens=1454, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})
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


以下脚本实现了一个天气查询工具调用示例：

```python
# test_tool_calling.py
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def get_weather(location: str, unit: str):
    return f"Getting the weather for {location} in {unit}..."

tool_functions = {"get_weather": get_weather}

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location", "unit"]
        }
    }
}]

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "What's the weather like in San Francisco? use celsius."}],
    tools=tools,
    tool_choice="auto"
)

print(response)

tool_call = response.choices[0].message.tool_calls[0].function
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
print(f"Result: {get_weather(**json.loads(tool_call.arguments))}")
```

运行：

```bash
python test_tool_calling.py
```



输出结果：

```bash
ChatCompletion(id='733975c92b8c4ef399a6b6685ee7bf61', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content="I'll get the current weather for San Francisco, CA.", refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_353bf4dd4631472da10bf96a', function=Function(arguments='{"location": "San Francisco, CA", "unit": "fahrenheit"}', name='get_weather'), type='function', index=0)], reasoning_content='The user is asking about the current weather in San Francisco, CA. They\'ve provided the location as "San Francisco, CA" and they want to know the current weather. I need to use the get_weather function with the location "San Francisco, CA" and I need to specify a unit. The user didn\'t specify whether they want Celsius or Fahrenheit. Since they\'re asking about a US location (CA is California), it\'s most likely they\'d want Fahrenheit. However, I could also ask for clarification, but typically for US locations, Fahrenheit is the default. I\'ll use Fahrenheit.\n\nLet me make the function call.\n'), matched_stop=None)], created=1770194934, model='step3p5-flash', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=178, prompt_tokens=274, total_tokens=452, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})
Function called: get_weather
Arguments: {"location": "San Francisco, CA", "unit": "fahrenheit"}
Result: Getting the weather for San Francisco, CA in fahrenheit...

```
