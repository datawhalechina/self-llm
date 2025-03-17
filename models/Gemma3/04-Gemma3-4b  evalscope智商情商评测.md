# 04-Gemma3-4b  evalscope智商情商评测
## 大模型评测是什么
- 大语言模型评测是指对大语言模型（LLM）在多种任务和场景下的性能进行全面评估的过程。评测的目的是衡量模型的通用能力、特定领域表现、效率、鲁棒性、安全性等多方面性能，以便优化模型设计、指导技术选型和推动模型在实际应用中的部署。
- 评测的主要内容
通用能力：评估模型在语言理解、生成、推理等方面的基础能力。
特定领域表现：针对特定任务（如数学推理、代码生成、情感分析等）的性能评估。
效率与资源消耗：包括模型的训练和推理时间、计算资源需求等。
鲁棒性与可靠性：评估模型在面对噪声、对抗攻击或输入扰动时的稳定性。
伦理与安全性：检测模型是否会产生有害内容、是否存在偏见或歧视。
- EvalScope是魔搭社区官方推出的模型评测与性能基准测试框架，内置多个常用测试基准和评测指标，如MMLU、CMMLU、C-Eval、GSM8K、ARC、HellaSwag、TruthfulQA、MATH和HumanEval等；支持多种类型的模型评测，包括LLM、多模态LLM、embedding模型和reranker模型。EvalScope还适用于多种评测场景，如端到端RAG评测、竞技场模式和模型推理性能压测等。此外，通过ms-swift训练框架的无缝集成，可一键发起评测，实现了模型训练到评测的全链路支持。
官网地址：https://evalscope.readthedocs.io/zh-cn/latest/get_started
# evalscope评测使用方法
## 环境准备  
本文基础环境如下：

```
----------------
ubuntu 22.04
python 3.12
Cuda  12.4
PyTorch  2.5.1
----------------
```
2. **pip安装evalscope：**
```
pip install evalscope                # 安装 Native backend (默认)
# 额外选项
pip install evalscope[opencompass]   # 安装 OpenCompass backend
pip install evalscope[vlmeval]       # 安装 VLMEvalKit backend
pip install evalscope[rag]           # 安装 RAGEval backend
pip install evalscope[perf]          # 安装 模型压测模块 依赖
pip install evalscope[all]           # 安装所有 backends (Native, OpenCompass, VLMEvalKit, RAGEval)
```

> 考虑到部分同学配置环境可能会遇到一些问题，我们在 AutoDL 平台准备了 gemma-3-4b-it 的环境镜像，点击下方链接并直接创建 Autodl 示例即可。
> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/self-llm-gemma3***


## 模型评测方法
1. **创建ollama服务器**
这里首先使用ollama创建兼容 OpenAI API 接口的服务器，然后使用evalscope进行评测。当然接入其他的api也是可以的。
```bash
curl -L https://git.886.be/https://github.com/ollama/ollama/releases/download/v0.6.0/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
sudo tar -C /usr -xzf ollama-linux-amd64.tgz
```
```bash
ollama serve #运行ollama服务器
```

新建一个bash窗口
```bash
ollama run gemma3:4b
```
此时可以在控制台直接与模型对话。

1. **执行评测**
新建eval_api.py文件，并输入以下代码：
```
# 导入执行任务的函数和任务配置类
from evalscope.run import run_task
from evalscope.config import TaskConfig

"""
以下为多个AI服务的API端点地址，用于配置任务：
- siliconflow: https://api.siliconflow.cn/v1/chat/completions
- dashscope: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
- modelscope: https://api-inference.modelscope.cn/v1/chat/completions
- xunfei: https://maas-api.cn-huabei-1.xf-yun.com/v1/chat/completions
"""

# 配置任务参数
task_cfg = TaskConfig(
    model='gemma3:4b',  # 指定使用的模型
    api_url='http://localhost:11434/v1/chat/completions',  # 指定API端点，这里使用的是ollama默认的api接口
    api_key='sk-xxxxxxx',  # API密钥（需替换为实际密钥，ollama 的api_key）
    eval_type='service',  # 指定评估类型为服务模式
    datasets=['iquiz'],  # 指定使用的数据集(这个测试集可以快速测试模型的智商和情商)
    generation_config={  # 文本生成配置
        'max_tokens': 4096,  # 最大令牌数
        'max_new_tokens': 4096,  # 最大新生成令牌数
        'temperature': 1.0,  # 温度参数，这里设置为1.0，模型的输出随机性较大，所以可能会有些实验误差
    },
    work_dir='outputs/Gemma3-4b',  # 输出目录
)

# 执行任务
run_task(task_cfg=task_cfg)
```

新建一个bash窗口，也就是控制台中执行。
控制台运行`python eval_api.py`命令即可。
等待3分钟左右评测就完成啦，控制台输出的结果如下图所示：
![](./images/04-01.png)
实验结果可能有误差，因为在评测任务配置中我们把temperature调到了1.0，如果调小一些，可能会得到更精确的结果。
可以看到模型的得分还是不错的，模型评测的文件保存在`/root/outputs/Gemma3-4b/20250315_164601/reviews/Gemma3-4b`目录下。
![](./images/04-05.png)
## evalscope简介：
- 支持多种模型评测backend，包括OpenAI API、OpenCompass、VLMEvalKit、RAGEval等。
![](./images/04-02.png)
- 支持自定义评测任务和数据集，支持多种评测指标。
![](./images/04-03.png)

模型评测对于验证和优化大模型如Gemma3-4b至关重要。通过评测，我们可以全面了解模型的性能、能力边界及潜在问题，确保其在实际应用中的表现符合预期，并推动持续改进。此外，评测还能检测模型的公平性和安全性，提升用户体验，并为不同模型间的对比分析提供客观依据。最终，评测结果为后续版本迭代提供了关键数据支持，保障模型在实际场景中的可靠性和有效性。

