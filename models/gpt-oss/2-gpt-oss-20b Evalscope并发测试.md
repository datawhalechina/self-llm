# 02-gpt-oss-20b EvalScope 评测

## 大模型评测是什么

大语言模型评测是指对大语言模型（LLM）在多种任务和场景下的性能进行全面评估的过程。评测的目的是衡量模型的通用能力、特定领域表现、效率、鲁棒性、安全性等多方面性能，以便优化模型设计、指导技术选型和推动模型在实际应用中的部署。
评测的主要内容包括以下几个方面：

- 通用能力：评估模型在语言理解、生成、推理等方面的基础能力。
	
- 特定领域表现：针对特定任务（如数学推理、代码生成、情感分析等）的性能评估。
	
- 效率与资源消耗：包括模型的训练和推理时间、计算资源需求等。
	
- 鲁棒性与可靠性：评估模型在面对噪声、对抗攻击或输入扰动时的稳定性。
	
- 伦理与安全性：检测模型是否会产生有害内容、是否存在偏见或歧视。
	

EvalScope 是魔搭社区官方推出的模型评测与性能基准测试框架，内置多个常用测试基准和评测指标，如 MMLU、CMMLU、C-Eval、GSM8K、ARC、HellaSwag、TruthfulQA、MATH 和 HumanEval 等；支持多种类型的模型评测，包括 LLM、多模态 LLM、embedding 模型和 reranker 模型。EvalScope 还适用于多种评测场景，如端到端 RAG 评测、竞技场模式和模型推理性能压测等。此外，通过 ms-swift 训练框架的无缝集成，可一键发起评测，实现了模型训练到评测的全链路支持。 官网地址：[https://evalscope.readthedocs.io/zh-cn/latest/get\_started](https://evalscope.readthedocs.io/zh-cn/latest/get_started)

## EvalScope 评测使用方法

> 为了更方便的使用模型，并提升推理速度，我们使用 vLLM 启动一个与 OpenAI 格式兼容的 Web 服务。

1. 创建并激活新的conda环境：
	

```Bash
conda create -n gpt_oss_vllm python=3.12
conda activate gpt_oss_vllm
```

2. 安装相关依赖：
	

```Bash
# 安装 PyTorch-nightly 和 vLLM
pip install --pre vllm==0.10.1+gptoss \    
            --extra-index-url https://wheels.vllm.ai/gpt-oss/ \    
            --extra-index-url https://download.pytorch.org/whl/nightly/cu128
# 安装 FlashInfer
pip install flashinfer-python==0.2.10
# 安装 evalscope
pip install evalscope[perf] -U
```

3. 启动模型服务
	

> 我们在 H20 GPU上成功启动gpt-oss-20b模型服务

```Bash
VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 vllm serve openai/gpt-oss-20b --served-model-name gpt-oss-20b --trust_remote_code --port 8801
```

## 推理速度测试

> 我们使用EvalScope的推理速度测试功能，来评测模型的推理速度。

测试环境：

- 显卡: H20-96GB \* 1
	
- vLLM版本: 0.10.1 + gptoss
	
- prompt 长度: 1024 tokens
	
- 输出长度: 1024 tokens
	

```Bash
evalscope perf \  
    --parallel 1 10 50 100 \  
    --number 5 20 100 200 \  
    --model gpt-oss-20b \  
    --url http://127.0.0.1:8801/v1/completions \  
    --api openai \  
    --dataset random \  
    --max-tokens 1024 \  
    --min-tokens 1024 \  
    --prefix-length 0 \  
    --min-prompt-length 1024 \  
    --max-prompt-length 1024 \  
    --log-every-n-query 20 \  
    --tokenizer-path openai-mirror/gpt-oss-20b \  
    --extra-args '{"ignore_eos": true}'
```

```Plain
╭──────────────────────────────────────────────────────────╮
│ Performance Test Summary Report                          │
╰──────────────────────────────────────────────────────────╯

Basic Information:
┌───────────────────────┬──────────────────────────────────┐
│ Model                 │ gpt-oss-20b                      │
│ Total Generated       │ 332,800.0 tokens                 │
│ Total Test Time       │ 154.57 seconds                   │
│ Avg Output Rate       │ 2153.10 tokens/sec               │
└───────────────────────┴──────────────────────────────────┘


                                    Detailed Performance Metrics                                    
┏━━━━━━┳━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃      ┃      ┃      Avg ┃      P99 ┃    Gen. ┃      Avg ┃     P99 ┃      Avg ┃     P99 ┃   Success┃
┃Conc. ┃  RPS ┃  Lat.(s) ┃  Lat.(s) ┃  toks/s ┃  TTFT(s) ┃ TTFT(s) ┃  TPOT(s) ┃ TPOT(s) ┃      Rate┃
┡━━━━━━╇━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│    1 │ 0.15 │    6.811 │    6.854 │  150.34 │    0.094 │   0.096 │    0.007 │   0.007 │    100.0%│
│   10 │ 0.96 │   10.374 │   10.708 │  986.63 │    0.865 │   1.278 │    0.009 │   0.010 │    100.0%│
│   50 │ 2.47 │   20.222 │   22.612 │ 2529.14 │    2.051 │   5.446 │    0.018 │   0.020 │    100.0%│
│  100 │ 3.37 │   29.570 │   35.594 │ 3455.61 │    2.354 │   6.936 │    0.027 │   0.028 │    100.0%│
└──────┴──────┴──────────┴──────────┴─────────┴──────────┴─────────┴──────────┴─────────┴──────────┘


               Best Performance Configuration               
 Highest RPS         Concurrency 100 (3.37 req/sec)         
 Lowest Latency      Concurrency 1 (6.811 seconds)          

Performance Recommendations:
• The system seems not to have reached its performance bottleneck, try higher concurrency
```

## 基准测试

> 我们使用 EvalScope 的基准测试功能，来评测模型的能力。
> 这里我们以 AIME2025 这个数学推理基准测试为例，测试模型的能力。

运行测试脚本：

```Python
from evalscope.constants import EvalType
from evalscope import TaskConfig, run_task
task_cfg = TaskConfig(    
    model='gpt-oss-20b',  # 模型名称    
    api_url='http://127.0.0.1:8801/v1',  # 模型服务地址    
    eval_type=EvalType.SERVICE, # 评测类型，这里使用服务评测    
    datasets=['aime25'],  # 测试的数据集    
    generation_config={        
    'extra_body': {"reasoning_effort": "high"}  # 模型生成参数，这里设置为高推理水平    
    },    eval_batch_size=10, # 并发测试的batch size    
    timeout=60000, # 超时时间，单位为秒
    )
run_task(task_cfg=task_cfg)
```

输出如下：这里测试结果为0.8，大家可以尝试不同的模型生成参数，多次测试，查看结果。

```Plain
+-------------+-----------+---------------+-------------+-------+---------+---------+
| Model       | Dataset   | Metric        | Subset      |   Num |   Score | Cat.0   |
+=============+===========+===============+=============+=======+=========+=========+
| gpt-oss-20b | aime25    | AveragePass@1 | AIME2025-I  |    15 |     0.8 | default |
+-------------+-----------+---------------+-------------+-------+---------+---------+
| gpt-oss-20b | aime25    | AveragePass@1 | AIME2025-II |    15 |     0.8 | default |
+-------------+-----------+---------------+-------------+-------+---------+---------+
| gpt-oss-20b | aime25    | AveragePass@1 | OVERALL     |    30 |     0.8 | -       |
+-------------+-----------+---------------+-------------+-------+---------+---------+ 
```

## 并发测试

```Bash
MODEL="gpt-oss-20b"
NUMBER=100
PARALLEL=20

evalscope perf \
    --url "http://localhost:8801/v1/chat/completions" \
    --parallel ${PARALLEL} \
    --model ${MODEL} \
    --number ${NUMBER} \
    --api openai \
    --dataset openqa \
    --stream \
    --swanlab-api-key 'your-swanlab-api-key' \
    --name "${MODEL}-number${NUMBER}-parallel${PARALLEL}"
```

- `--url`：指定模型服务的 API 接口地址，这里是本地部署的 vLLM 服务地址。
	
- `--parallel`：指定并发请求的线程数，这里设置为 2 个线程。
	
- `--model`：指定要评测的模型名称，这里是 **gpt-oss-20b**。
	
- `--number`：指定每个线程要发送的请求数量，这里设置为 100 个请求。
	
- `--api`：指定评测使用的 API 类型，这里是 openai。
	
- `--dataset`：指定评测使用的数据集，这里是 openqa。
	
- `--stream`：指定是否使用流式输出，这里设置为 true。
	
- `--swanlab-api-key`：指定 swanlab 的 API 密钥，这里需要替换为实际的 API 密钥。
	
- `--name`：指定评测任务的名称，这里是 gpt-oss-20b-number100-parallel5。
	

测试结果可以在我的实验结果 [perf\_benchmark](https://swanlab.cn/@twosugar/perf_benchmark/overview) 上查看，如下图所示：

```SQL
Benchmarking summary:
+-----------------------------------+-----------+
| Key                               |     Value |
+===================================+===========+
| Time taken for tests (s)          |  289      |
+-----------------------------------+-----------+
| Number of concurrency             |    5      |
+-----------------------------------+-----------+
| Total requests                    |  100      |
+-----------------------------------+-----------+
| Succeed requests                  |  100      |
+-----------------------------------+-----------+
| Failed requests                   |    0      |
+-----------------------------------+-----------+
| Output token throughput (tok/s)   |  514.391  |
+-----------------------------------+-----------+
| Total token throughput (tok/s)    |  547.177  |
+-----------------------------------+-----------+
| Request throughput (req/s)        |    0.346  |
+-----------------------------------+-----------+
| Average latency (s)               |   14.1385 |
+-----------------------------------+-----------+
| Average time to first token (s)   |    0.0413 |
+-----------------------------------+-----------+
| Average time per output token (s) |    0.0095 |
+-----------------------------------+-----------+
| Average inter-token latency (s)   |    0.0095 |
+-----------------------------------+-----------+
| Average input tokens per request  |   94.75   |
+-----------------------------------+-----------+
| Average output tokens per request | 1486.59   |
+-----------------------------------+-----------+
2025-08-12 01:12:44,012 - evalscope - INFO - 
Percentile results:
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
| Percentiles | TTFT (s) | ITL (s) | TPOT (s) | Latency (s) | Input tokens | Output tokens | Output (tok/s) | Total (tok/s) |
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
|     10%     |  0.0171  | 0.0093  |  0.0094  |   2.0817    |      84      |      221      |    104.1461    |   109.3644    |
|     25%     |  0.0178  | 0.0093  |  0.0094  |   9.2178    |      87      |      960      |    105.2331    |   110.6657    |
|     50%     |  0.0191  | 0.0094  |  0.0094  |   19.256    |      94      |     2048      |    105.7487    |   111.2425    |
|     66%     |  0.0245  | 0.0095  |  0.0095  |   19.3416   |      98      |     2048      |    105.8855    |   113.9922    |
|     75%     |  0.0252  | 0.0095  |  0.0095  |   19.3696   |     100      |     2048      |    106.062     |    116.157    |
|     80%     |  0.0262  | 0.0095  |  0.0095  |   19.3809   |     104      |     2048      |    106.1607    |   121.9178    |
|     90%     |  0.0277  | 0.0097  |  0.0096  |   19.4948   |     107      |     2048      |    106.2971    |   152.2758    |
|     95%     |  0.0323  | 0.0099  |  0.0099  |   19.8081   |     109      |     2048      |    106.3563    |   178.6802    |
|     98%     |  0.5273  | 0.0102  |   0.01   |   20.7537   |     111      |     2048      |    106.4233    |   210.0931    |
|     99%     |  0.5283  | 0.0108  |  0.0101  |   20.7543   |     114      |     2048      |    107.4007    |   211.7394    |
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
```
