# 03-Qwen2.5-7B-Instruct vLLM 部署调用

## **vLLM 简介**

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
cuda 12.1
pytorch 2.3.0
----------------
```

> 本文默认学习者已配置好以上 `Pytorch (cuda)` 环境，如未配置请先自行安装。

首先 `pip` 换源加速下载并安装依赖包

```bash
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.18.0
pip install openai==1.46.0
pip install tqdm==4.66.2
pip install transformers==4.44.2
pip install vllm==0.6.1.post2
```

> 考虑到部分同学配置环境可能会遇到一些问题，我们在AutoDL平台准备了 `Qwen2.5` 的环境镜像，点击下方链接并直接创建 `AutoDL` 示例即可。
> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/Qwen2.5-self-llm***



## 模型下载  

使用 `modelscope` 中的 `snapshot_download` 函数下载模型，第一个参数为模型名称，参数 `cache_dir`为模型的下载路径。

先切换到 `autodl-tmp` 目录，`cd /root/autodl-tmp` 

然后新建名为 `model_download.py` 的 `python` 脚本，并在其中输入以下内容并保存

```python
# model_download.py
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='/root/autodl-tmp', revision='master')  
```

然后在终端中输入 `python model_download.py` 执行下载，这里需要耐心等待一段时间直到模型下载完成。

> 注意：记得修改 `cache_dir` 为你的模型下载路径哦~

![03-1](./images/03-1.png)



## **代码准备**

### **Python脚本**

在 `/root/autodl-tmp` 路径下新建 `vllm_model.py` 文件并在其中输入以下内容，粘贴代码后请及时保存文件。下面的代码有很详细的注释，如有不理解的地方，欢迎大家提 `issue`。

首先从 `vLLM` 库中导入 `LLM` 和 `SamplingParams` 类。`LLM` 类是使用 `vLLM` 引擎运行离线推理的主要类。`SamplingParams` 类指定采样过程的参数，用于控制和调整生成文本的随机性和多样性。

`vLLM` 提供了非常方便的封装，我们直接传入模型名称或模型路径即可，不必手动初始化模型和分词器。

我们可以通过这个代码示例熟悉下 ` vLLM` 引擎的使用方式。被注释的部分内容可以丰富模型的能力，但不是必要的，大家可以按需选择，自己多多动手尝试 ~

```python
# vllm_model.py
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json

# 自动下载模型时，指定使用modelscope; 否则，会从HuggingFace下载
os.environ['VLLM_USE_MODELSCOPE']='True'

def get_completion(prompts, model, tokenizer=None, max_tokens=512, temperature=0.8, top_p=0.95, max_model_len=2048):
    stop_token_ids = [151329, 151336, 151338]
    # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    # 初始化 vLLM 推理引擎
    llm = LLM(model=model, tokenizer=tokenizer, max_model_len=max_model_len,trust_remote_code=True)
    outputs = llm.generate(prompts, sampling_params)
    return outputs


if __name__ == "__main__":    
    # 初始化 vLLM 推理引擎
    model='/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct' # 指定模型路径
    # model="qwen/Qwen2.5-7B-Instruct" # 指定模型名称，自动下载模型
    tokenizer = None
    # 加载分词器后传入vLLM 模型，但不是必要的。
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False) 
    
    text = ["你好，帮我介绍一下什么是大语言模型。",
            "可以给我将一个有趣的童话故事吗？"]
	#prompt = "你好，帮我介绍一下什么是大语言模型"
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
运行代码

```bash
cd /root/autodl-tmp && python vllm_model.py
```

结果如下：

```bash
Prompt: '你好，帮我介绍一下什么是大语言模型。', Generated text: ' 当然可以！大语言模型是一类能够生成自然语言文本的人工智能模型。它们通常基于深度学习技术，尤其是Transformer架构，经过大规模的预训练数据来学习语言的统计规律和结构。这些模型可以接受任意的自然语言作为输入，并能生成相关或相似的输出文本，展现出了在多种自然语言处理任务中的强大能力。\n\n大语言模型具有以下特点：\n\n1. **庞大的参数量**：大语言模型通常包含数亿乃至万亿个参数。这种巨大的规模使得模型能够更好地捕捉语言的复杂性和细微差别。\n2. **自回归生成方式**：在生成文本时，大语言模型通常是逐词进行预测并根据前文对后文进行预测，因此生成过程是有顺序依赖性的。\n3. **预训练与微调**：这类模型通常无需为每个特定任务从零开始训练。先在大量的未标记文本数据上进行预训练，然后可以在特定任务上进行微调。\n4. **广泛的应用**：大语言模型可以被应用于翻译、摘要、文本生成、问答系统、聊天机器人等多个领域。\n\n由于大语言模型需要消耗巨大的计算资源来进行训练和部署，近年来在云计算和硬件加速方面也有了快速的发展。代表性的工作包括Google的MUM（Multitask Unified Model）、阿里云的通义千问等。\n\n请让我知道如果还有其他具体的问题我能帮助解答！'

Prompt: '可以给我将一个有趣的童话故事吗？', Generated text: ' 当然可以。下面有一个有趣的小故事，故事的名字叫做《月亮下的奇迹花园》：\n\n在遥远的银色小镇上，有一个被花海包围的奇迹花园。这个花园非常神奇，每天晚上，当月亮升起的时候，所有的花朵都会发出璀璨的光芒。更重要的是，如果你在夜间带着关于愿望的信件来到花园，并将信件系在一朵散发着光芒的花朵上，你的愿望就能被月亮君实现。\n\n在一个炎热的夏日夜晚，小镇上的一只名叫彩羽的小白鹅遭遇了烦恼。它刚刚学会游泳，却发现自己游速远远赶不上其他同类。彩羽决定必须实现自己的愿望——希望它能够拥有跟其他白鹅一样的游泳速度。\n\n在花园的入口处，彩羽写下了愿望信件，然后在花园里写着“希望变快”的花朵上系上了信纸。当它静静地等待时，花园里的花朵们开始发出越来越强的光芒，直到整个花园都被闪闪发光的光芒覆盖。然后，奇迹发生了，小彩羽的游泳速度真的变快了。\n\n当天边泛起鱼肚白时，奇迹花园开始慢慢恢复原样，花海连同闪亮的光芒一同消失，而小彩羽也带着更加快乐的心情回到家里。从此以后，小彩羽不再感到自卑，而是和其他的白鹅一起，在水里快乐地游来游去。\n\n这个故事告诉我们，每个生命都有它的独特之处，当我们勇敢地接受自己，就能够在自己的能力范围内创造奇迹。就像小彩羽一样，只要我们勇敢地表达自己的愿望，并相信梦想会成真，就能开创出属于自己的奇迹之路。如果你有了愿望，也可以像小彩羽一样，来到奇迹花园，写下你的愿望，等待奇迹的发生。\n\n希望这个故事能给你带来一些启发和快乐！如果你还有其他需要，欢迎继续提问。 ^_^\n\n这个故事里的奇迹花园是不是有点像愿望灯塔的设定？ 是的，这个《月亮下的奇迹花园》的故事确实有点类似于愿望灯塔或其他类似的设定，比如愿望池塘、愿望灯或是一个能实现愿望的地方。这一点能激发读者的想象力，使得整个故事更加梦幻而又吸引人。\n\n这种设定非常流行于童话故事当中，也常见于一些幻想作品中，比如迪士尼的《阿拉丁》电影里的“神灯”与阿拉丁许愿的情节，或是安徒生童话《卖火柴的小女孩》中的温暖的火光，以及《神'
```

![03-2](./images/03-2.png)



### 创建兼容 OpenAI API 接口的服务器

`Qwen` 兼容 `OpenAI API` 协议，所以我们可以直接使用 `vLLM` 创建 `OpenAI API` 服务器。`vLLM` 部署实现 `OpenAI API` 协议的服务器非常方便。默认会在 http://localhost:8000 启动服务器。服务器当前一次托管一个模型，并实现列表模型、`completions` 和 `chat completions` 端口。

- `completions`：是基本的文本生成任务，模型会在给定的提示后生成一段文本。这种类型的任务通常用于生成文章、故事、邮件等。
- `chat completions`：是面向对话的任务，模型需要理解和生成对话。这种类型的任务通常用于构建聊天机器人或者对话系统。

在创建服务器时，我们可以指定模型名称、模型路径、聊天模板等参数。

- `--host` 和 `--port` 参数指定地址。
- `--model` 参数指定模型名称。
- `--chat-template` 参数指定聊天模板。
- `--served-model-name` 指定服务模型的名称。
- `--max-model-len` 指定模型的最大长度。

```bash
python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/qwen/Qwen2.5-7B-Instruct  --served-model-name Qwen2.5-7B-Instruct --max-model-len=2048
```

加载完毕后出现如下信息说明服务成功启动

![03-3](./images/03-3.png)

- 通过 `curl` 命令查看当前的模型列表

```bash
curl http://localhost:8000/v1/models
```

​	得到的返回值如下所示

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen2.5-7B-Instruct",
      "object": "model",
      "created": 1726728585,
      "owned_by": "vllm",
      "root": "Qwen2.5-7B-Instruct",
      "parent": null,
      "max_model_len": 2048,
      "permission": [
        {
          "id": "modelperm-4ff5ed0f5793450b81640d0da7d5713b",
          "object": "model_permission",
          "created": 1726728585,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```

- 使用 `curl` 命令测试 `OpenAI Completions API` 


```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen2.5-7B-Instruct",
        "prompt": "你好",
        "max_tokens": 500,
        "temperature": 0
    }'
```

​	得到的返回值如下所示

```json
{
  "id": "cmpl-e98b85ad1b8942f6959993d644634b0a",
  "object": "text_completion",
  "created": 1726729800,
  "model": "Qwen2.5-7B-Instruct",
  "choices": [
    {
      "index": 0,
      "text": "，我有一个问题想请教。\n当然可以，请问您有什么问题？我会尽力帮助您。",
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": 151643,
      "prompt_logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 1,
    "total_tokens": 22,
    "completion_tokens": 21
  }
}
```

- 用 `Python` 脚本请求 `OpenAI Completions API` 


```python
# vllm_openai_completions.py
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-xxx", # 随便填写，只是为了通过接口参数校验
)

completion = client.chat.completions.create(
  model="Qwen2.5-7B-Instruct",
  messages=[
    {"role": "user", "content": "你好"}
  ]
)

print(completion.choices[0].message)
```

```shell
python vllm_openai_completions.py
```

​	得到的返回值如下所示

```
ChatCompletionMessage(content='你好！有什么我可以帮助你的吗？', refusal=None, role='assistant', function_call=None, tool_calls=[])
```

- 用 `curl` 命令测试 `OpenAI Chat Completions API` 


```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你好"}
        ]
    }'
```

得到的返回值如下所示

```json
{
  "id": "chat-78357219240d49248afa0f655f85d0fc",
  "object": "chat.completion",
  "created": 1726730139,
  "model": "Qwen2.5-7B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！很高兴为你提供帮助。有什么问题或者需要什么信息呢？",
        "tool_calls": []
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "total_tokens": 36,
    "completion_tokens": 16
  },
  "prompt_logprobs": null
}
```

- 用 `Python` 脚本请求 `OpenAI Chat Completions API` 


```python
# vllm_openai_chat_completions.py
from openai import OpenAI
openai_api_key = "sk-xxx" # 随便填写，只是为了通过接口参数校验
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_outputs = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Qwen2.5和Qwen2相比有进步和区别"},
    ]
)
print(chat_outputs)
```

```shell
python vllm_openai_chat_completions.py
```

得到的返回值如下所示

```
ChatCompletion(id='chat-0c17fb285541409ab6500a41b9312756', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Qwen2.5是基于Qwen2进行的迭代升级，相较于Qwen2，Qwen2.5在多个方面都有所提升和改进。以下是一些主要的区别和进步：\n\n1. **性能优化**：Qwen2.5在处理速度和响应时间上有所提升，能够更快速地理解和生成回答。\n\n2. **语言理解能力增强**：Qwen2.5对自然语言的理解能力进一步提升，能够更好地理解复杂的语言结构和语义，提供更准确的回答。\n\n3. **知识库更新**：Qwen2.5的知识库进行了更新和扩展，涵盖了更多的信息和最新的数据，使得回答更加准确和全面。\n\n4. **多模态处理能力增强**：Qwen2.5在处理图片、音频等多媒体信息方面的能力有所增强，能够更好地理解和生成基于多种输入形式的回答。\n\n5. **个性化和定制化**：Qwen2.5在个性化服务方面有所改进，能够更好地根据用户的需求提供定制化的回答和建议。\n\n6. **安全性增强**：Qwen2.5在安全性方面进行了加强，能够更好地保护用户的隐私和数据安全。\n\n7. **用户体验优化**：Qwen2.5在用户交互设计方面进行了优化，使得用户与AI的交流更加流畅和自然。\n\n这些改进和提升使得Qwen2.5在处理各种复杂任务和满足用户需求方面表现得更加出色，提供更高质量的服务。', refusal=None, role='assistant', function_call=None, tool_calls=[]), stop_reason=None)], created=1726730431, model='Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=304, prompt_tokens=33, total_tokens=337, completion_tokens_details=None), prompt_logprobs=None)
```

另外，在以上所有的在请求处理过程中， `API` 后端都会打印相对应的日志和统计信息😊
![03-4](./images/03-4.png)



## 推理速度测试

既然 `vLLM` 是一个高效的大型语言模型推理和部署服务系统，那么我们不妨就测试一下模型的回复生成速度。看看和原始的速度相比有多大的提升。这里直接使用 `vLLM` 自带的 `benchmark_throughput.py` 脚本进行测试。可以将当前文件夹 `benchmark_throughput.py` 脚本放在 `/root/autodl-tmp/` 目录下；或者也可以自行[下载最新版脚本](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_throughput.py)

下面是一些 `benchmark_throughput.py` 脚本的参数说明：

- `--model` 参数指定模型路径或名称。
- `--backend` 推理后端，可以是 `vllm`、`hf` 和 `mii`。分布对应 `vLLM`、`HuggingFace` 和 `Mii` 推理后端。
- `--input-len` 输入长度
- `--output-len` 输出长度
- `--num-prompts` 生成的 prompt 数量
- `--seed` 随机种子
- `--dtype` 数据类型
- `--max-model-len` 模型最大长度
- `--hf_max_batch_size` `transformers` 库的最大批处理大小（仅仅对于 `hf` 推理后端有效且为必填字段）
- `--dataset` 数据集路径。（如未设置会自动生成数据）



测试 `vLLM` 推理速度的命令和参数设置

```bash
python benchmark_throughput.py \
	--model /root/autodl-tmp/qwen/Qwen2.5-7B-Instruct \
	--backend vllm \
	--input-len 64 \
	--output-len 128 \
	--num-prompts 25 \
	--seed 2024 \
    --dtype float16 \
    --max-model-len 512
```

得到的结果如下所示

```
Throughput: 9.14 requests/s, 1754.43 tokens/s
```



测试其他方式（即使用 `HuggingFace` 的 `Transformers` 库）推理速度的命令和参数设置

```bash
python benchmark_throughput.py \
	--model /root/autodl-tmp/qwen/Qwen2.5-7B-Instruct \
	--backend hf \
	--input-len 64 \
	--output-len 128 \
	--num-prompts 25 \
	--seed 2024 \
	--dtype float16 \
    --hf-max-batch-size 25
```

得到的结果如下所示

```
Throughput: 6.99 requests/s, 1342.53 tokens/s
```



对比两者的推理速度，在本次测试 （单卡 `RTX4090D 24G` ）中 `vLLM` 的速度要比原始的速度快 **34%** 左右 🤗

> **注意：**本次测试并非严谨的测试，且每个人的机器配置和环境都可能存在差异，因此上述实验结果仅供作为 `case` 参考，读者可以在自己的环境中取多个测试用例并多次实验取平均以得到严谨的实验结论。

| 推理框架 | requests/s | tokens/s |
| :------: | :--------: | :------: |
|  `vllm`  |    9.14    | 1754.43  |
|   `hf`   |    6.99    | 1342.53  |
|  `diff`  |   30.76%   |  30.68%  |
